//! A rig [`Tool`] that runs `grep` over files within the current working
//! directory.
//!
//! `grep` cannot itself modify files, but to keep the tool provably read-only
//! it (a) never goes through a shell — so there is no way to inject `>`, `|`,
//! `;`, `$()`, etc. — (b) confines the searched path to the CWD subtree, and
//! (c) rejects any flag that is not on an allow-list of known read-only grep
//! options. Anything that isn't provably side-effect-free is refused.

use std::process::Command;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::scratch;

/// How many output lines to echo back inline.
const PREVIEW_LINES: usize = 40;

/// Read-only grep flags that take no value.
const ALLOWED_FLAGS: &[&str] = &[
    "-i", "--ignore-case", "-n", "--line-number", "-r", "-R", "--recursive", "-l",
    "--files-with-matches", "-L", "--files-without-match", "-c", "--count", "-v",
    "--invert-match", "-w", "--word-regexp", "-x", "--line-regexp", "-E", "--extended-regexp",
    "-F", "--fixed-strings", "-G", "--basic-regexp", "-P", "--perl-regexp", "-o",
    "--only-matching", "-H", "--with-filename", "-h", "--no-filename", "-s", "--no-messages",
    "-a", "--text", "-I",
];

/// Read-only grep flags that take a value. Accepted as `-A3`, `-A=3`,
/// `--after-context=3`, or split across two tokens (`-A`, `3`). The value half,
/// when separate, must be a plain non-flag token (validated below).
const ALLOWED_VALUE_FLAGS: &[&str] = &[
    "-A", "--after-context", "-B", "--before-context", "-C", "--context", "-m", "--max-count",
    "--include", "--exclude", "--exclude-dir", "--color",
];

/// Arguments accepted by [`Grep`].
#[derive(Debug, Deserialize)]
pub struct GrepArgs {
    /// Pattern (regular expression) to search for.
    pub pattern: String,
    /// File or directory within the CWD to search. Defaults to ".".
    #[serde(default)]
    pub path: Option<String>,
    /// Extra grep flags. Only read-only flags are permitted; anything else is
    /// rejected. Use the `=` form for glob-valued flags, e.g. `--include=*.rs`.
    #[serde(default)]
    pub flags: Vec<String>,
}

/// What [`Grep`] returns to the model.
#[derive(Debug, Serialize)]
pub struct GrepOutput {
    /// Number of output lines grep produced.
    pub lines: usize,
    /// grep's exit code (0 = matches, 1 = no matches, 2 = error).
    pub exit_code: i32,
    /// Scratch file holding the full output, or `None` if there was none.
    pub scratch_file: Option<String>,
    /// The first lines of output (or grep's stderr if it failed).
    pub preview: String,
}

/// Errors that [`Grep`] can produce.
#[derive(Debug, thiserror::Error)]
pub enum GrepError {
    #[error("could not determine the current working directory: {0}")]
    Cwd(std::io::Error),
    #[error("'{0}' is outside the current working directory")]
    OutsideCwd(String),
    #[error("could not resolve '{path}': {source}")]
    Resolve {
        path: String,
        source: std::io::Error,
    },
    #[error("flag '{0}' is not allowed; grep is restricted to read-only options")]
    DisallowedFlag(String),
    #[error("failed to run grep: {0}")]
    Spawn(std::io::Error),
    #[error("could not write scratch file: {0}")]
    Scratch(std::io::Error),
}

impl From<super::PathError> for GrepError {
    fn from(e: super::PathError) -> Self {
        match e {
            super::PathError::Cwd(io) => GrepError::Cwd(io),
            super::PathError::OutsideCwd(p) => GrepError::OutsideCwd(p),
            super::PathError::Resolve { path, source } => GrepError::Resolve { path, source },
        }
    }
}

/// Runs read-only grep within the current working directory.
#[derive(Debug, Clone, Default)]
pub struct Grep;

impl Tool for Grep {
    const NAME: &'static str = "grep";

    type Error = GrepError;
    type Args = GrepArgs;
    type Output = GrepOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Search file contents with grep, within the current working directory. \
                          Read-only: only safe grep flags are accepted (e.g. -i, -n, -r, -l, -c, \
                          -A/-B/-C N, --include=GLOB); file-writing or non-grep usage is rejected. \
                          Pass `path` to scope the search (a file or directory, default '.'; use \
                          -r in `flags` to recurse into a directory). Full output is written to a \
                          scratch file you can read with read_file."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression to search for."
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory within the CWD to search. Defaults to '.'."
                    },
                    "flags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Optional read-only grep flags, e.g. [\"-r\", \"-n\", \"-i\"]."
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        validate_flags(&args.flags)?;

        let path_arg = args.path.as_deref().unwrap_or(".");
        let resolved = super::contain(path_arg)?;

        // No shell: arguments are passed directly to grep as an argv. `-e` keeps
        // a pattern starting with `-` from being read as a flag, and `--` ends
        // option parsing so the path is never treated as one.
        let output = Command::new("grep")
            .args(&args.flags)
            .arg("-e")
            .arg(&args.pattern)
            .arg("--")
            .arg(&resolved)
            .output()
            .map_err(GrepError::Spawn)?;

        let exit_code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines = if stdout.is_empty() {
            0
        } else {
            stdout.lines().count()
        };

        let (scratch_file, preview) = if stdout.is_empty() {
            // No matches: surface stderr (e.g. "Is a directory — use -r") so the
            // model gets a useful hint instead of silence.
            let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
            (None, stderr)
        } else {
            let path = scratch::write_unique("grep", "txt", &stdout).map_err(GrepError::Scratch)?;
            (
                Some(path.display().to_string()),
                super::preview(&stdout, PREVIEW_LINES),
            )
        };

        Ok(GrepOutput {
            lines,
            exit_code,
            scratch_file,
            preview,
        })
    }
}

/// Reject any flag that isn't a known read-only grep option.
fn validate_flags(flags: &[String]) -> Result<(), GrepError> {
    for flag in flags {
        if !is_allowed_flag(flag) {
            return Err(GrepError::DisallowedFlag(flag.clone()));
        }
    }
    Ok(())
}

fn is_allowed_flag(flag: &str) -> bool {
    if ALLOWED_FLAGS.contains(&flag) {
        return true;
    }
    for vf in ALLOWED_VALUE_FLAGS {
        if flag == *vf {
            return true; // value supplied as the next token
        }
        if let Some(rest) = flag.strip_prefix(vf) {
            // Attached value: `-A3`, `-A=3`, `--include=*.rs`.
            if rest.starts_with('=') || (!vf.starts_with("--") && !rest.is_empty()) {
                return true;
            }
        }
    }
    // A bare numeric token is the value half of a split context/max-count flag
    // (e.g. the "3" in `["-A", "3"]`). Digits can't name a path that escapes the
    // CWD, so this is safe.
    !flag.starts_with('-') && !flag.is_empty() && flag.chars().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_common_readonly_flags() {
        for f in ["-i", "-n", "-r", "--recursive", "-A3", "-A", "3", "--include=*.rs", "-C", "5"] {
            assert!(is_allowed_flag(f), "{f} should be allowed");
        }
    }

    #[test]
    fn rejects_unknown_or_unsafe_flags() {
        // Not real grep flags / could name files / not on the allow-list.
        for f in ["--foo", "-Z--evil", "/etc/passwd", "..", "-d", "--devices=read"] {
            assert!(!is_allowed_flag(f), "{f} should be rejected");
        }
    }

    #[tokio::test]
    async fn rejects_disallowed_flag_at_call() {
        let err = Grep
            .call(GrepArgs {
                pattern: "x".into(),
                path: None,
                flags: vec!["--write-me".into()],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, GrepError::DisallowedFlag(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn rejects_path_outside_cwd() {
        let err = Grep
            .call(GrepArgs {
                pattern: "root".into(),
                path: Some("/etc/hosts".into()),
                flags: vec![],
            })
            .await
            .unwrap_err();
        assert!(matches!(err, GrepError::OutsideCwd(_)), "got {err:?}");
    }
}
