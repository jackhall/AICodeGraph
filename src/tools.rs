//! Tools the LLM agent can call to work with files under the current working
//! directory.
//!
//! Every tool is confined to the CWD subtree (see [`contain`]). Tools that can
//! produce bulky output ([`Explore`], [`Grep`]) write the full result to a
//! unique file in the [`scratch`] directory and return its path, so the model
//! can re-read it with [`ReadFile`] or search it with [`Grep`].

pub mod explore;
pub mod grep;
pub mod read_file;
pub mod scratch;

pub use explore::Explore;
pub use grep::Grep;
pub use read_file::ReadFile;

use std::path::{Path, PathBuf};

/// Error resolving a user-supplied path against the current working directory.
#[derive(Debug, thiserror::Error)]
pub enum PathError {
    #[error("could not determine the current working directory: {0}")]
    Cwd(std::io::Error),
    #[error("'{0}' is outside the current working directory")]
    OutsideCwd(String),
    #[error("could not resolve '{path}': {source}")]
    Resolve {
        path: String,
        source: std::io::Error,
    },
}

/// Resolve `requested` (relative to, or within, the CWD) to a canonical path and
/// ensure it stays inside the CWD subtree.
///
/// Canonicalization resolves `..` and symlinks *before* the containment check,
/// so neither can be used to escape the working directory. The target must
/// exist (canonicalization requires it).
pub(crate) fn contain(requested: &str) -> Result<PathBuf, PathError> {
    let cwd = std::env::current_dir().map_err(PathError::Cwd)?;
    let base = cwd.canonicalize().map_err(PathError::Cwd)?;

    let req = Path::new(requested);
    let candidate = if req.is_absolute() {
        req.to_path_buf()
    } else {
        base.join(req)
    };

    let resolved = candidate.canonicalize().map_err(|source| PathError::Resolve {
        path: requested.to_string(),
        source,
    })?;

    if !resolved.starts_with(&base) {
        return Err(PathError::OutsideCwd(requested.to_string()));
    }
    Ok(resolved)
}

/// First `max_lines` lines of `text`, with a trailing note if it was truncated.
/// Used to give the model an inline glimpse of output that was spooled to a
/// scratch file.
pub(crate) fn preview(text: &str, max_lines: usize) -> String {
    let total = text.lines().count();
    let mut out: String = text
        .lines()
        .take(max_lines)
        .collect::<Vec<_>>()
        .join("\n");
    if total > max_lines {
        out.push_str(&format!("\n… ({} more lines)", total - max_lines));
    }
    out
}
