//! A rig [`Tool`] that explores the directory tree under the current working
//! directory: either the immediate contents of a directory, or every file in
//! the subtree whose name matches a wildcard.
//!
//! The full listing is written to a scratch file; the tool returns its path
//! plus a short preview.

use glob::Pattern;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;
use walkdir::WalkDir;

use super::scratch;

/// Directories never worth descending into during a subtree walk.
const SKIP_DIRS: &[&str] = &[".git", "target", "node_modules", scratch::SCRATCH_DIR];

/// How many listing lines to echo back inline.
const PREVIEW_LINES: usize = 40;

/// Arguments accepted by [`Explore`].
#[derive(Debug, Deserialize)]
pub struct ExploreArgs {
    /// Directory to explore, relative to the CWD. Defaults to ".".
    #[serde(default)]
    pub path: Option<String>,
    /// Optional filename wildcard (e.g. "*.rs"). When set, the entire subtree
    /// under `path` is searched and every matching file is listed. When unset,
    /// only the immediate contents of `path` are listed.
    #[serde(default)]
    pub pattern: Option<String>,
}

/// What [`Explore`] returns to the model on success.
#[derive(Debug, Serialize)]
pub struct ExploreOutput {
    /// Number of entries found.
    pub count: usize,
    /// Scratch file holding the full newline-separated listing.
    pub scratch_file: String,
    /// The first lines of that listing.
    pub preview: String,
}

/// Errors that [`Explore`] can produce.
#[derive(Debug, thiserror::Error)]
pub enum ExploreError {
    #[error("could not determine the current working directory: {0}")]
    Cwd(std::io::Error),
    #[error("'{0}' is outside the current working directory")]
    OutsideCwd(String),
    #[error("'{0}' is not a directory")]
    NotADirectory(String),
    #[error("invalid wildcard pattern '{pattern}': {source}")]
    BadPattern {
        pattern: String,
        source: glob::PatternError,
    },
    #[error("failed while exploring '{path}': {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("could not write scratch file: {0}")]
    Scratch(std::io::Error),
}

impl From<super::PathError> for ExploreError {
    fn from(e: super::PathError) -> Self {
        match e {
            super::PathError::Cwd(io) => ExploreError::Cwd(io),
            super::PathError::OutsideCwd(p) => ExploreError::OutsideCwd(p),
            super::PathError::Resolve { path, source } => ExploreError::Io { path, source },
        }
    }
}

/// Lists directory contents, or finds files in the subtree by wildcard.
#[derive(Debug, Clone, Default)]
pub struct Explore;

impl Tool for Explore {
    const NAME: &'static str = "explore";

    type Error = ExploreError;
    type Args = ExploreArgs;
    type Output = ExploreOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "List a directory, or recursively find files matching a wildcard \
                          `pattern`. Output saved to a scratch file."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Directory, default '.'" },
                    "pattern": { "type": "string", "description": "Filename wildcard, e.g. *.rs" }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let dir_arg = args.path.as_deref().unwrap_or(".");

        let cwd = std::env::current_dir().map_err(ExploreError::Cwd)?;
        let cwd = cwd.canonicalize().map_err(ExploreError::Cwd)?;
        let base = super::contain(dir_arg)?;

        if !base.is_dir() {
            return Err(ExploreError::NotADirectory(dir_arg.to_string()));
        }

        let entries = match args.pattern.as_deref().filter(|p| !p.is_empty()) {
            // Subtree mode: walk everything, filter by wildcard on file name.
            Some(pat) => {
                let pattern = Pattern::new(pat).map_err(|source| ExploreError::BadPattern {
                    pattern: pat.to_string(),
                    source,
                })?;
                let mut out = Vec::new();
                let walker = WalkDir::new(&base)
                    .into_iter()
                    .filter_entry(|e| !is_skipped(e));
                for entry in walker {
                    let entry = entry.map_err(|e| ExploreError::Io {
                        path: dir_arg.to_string(),
                        source: std::io::Error::from(e),
                    })?;
                    if entry.file_type().is_file() {
                        if let Some(name) = entry.file_name().to_str() {
                            if pattern.matches(name) {
                                let rel = entry.path().strip_prefix(&cwd).unwrap_or(entry.path());
                                out.push(rel.display().to_string());
                            }
                        }
                    }
                }
                out.sort();
                out
            }
            // Listing mode: immediate children only, ls-style.
            None => {
                let mut out = Vec::new();
                let rd = std::fs::read_dir(&base).map_err(|source| ExploreError::Io {
                    path: dir_arg.to_string(),
                    source,
                })?;
                for entry in rd {
                    let entry = entry.map_err(|source| ExploreError::Io {
                        path: dir_arg.to_string(),
                        source,
                    })?;
                    let mut name = entry.file_name().to_string_lossy().into_owned();
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        name.push('/');
                    }
                    out.push(name);
                }
                out.sort();
                out
            }
        };

        let listing = entries.join("\n");
        let scratch_file = scratch::write_unique("explore", "txt", &listing)
            .map_err(ExploreError::Scratch)?;

        Ok(ExploreOutput {
            count: entries.len(),
            scratch_file: scratch_file.display().to_string(),
            preview: super::preview(&listing, PREVIEW_LINES),
        })
    }
}

/// Skip noisy/large directories (but never the walk root itself).
fn is_skipped(entry: &walkdir::DirEntry) -> bool {
    entry.depth() > 0
        && entry.file_type().is_dir()
        && entry
            .file_name()
            .to_str()
            .map(|n| SKIP_DIRS.contains(&n))
            .unwrap_or(false)
}
