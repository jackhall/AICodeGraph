//! A rig [`Tool`] that reads a UTF-8 text file located within the current
//! working directory (containment enforced by [`super::contain`]).

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Arguments accepted by [`ReadFile`].
#[derive(Debug, Deserialize)]
pub struct ReadFileArgs {
    /// Path to the file, relative to (or within) the current working directory.
    pub path: String,
}

/// What [`ReadFile`] returns to the model on success.
#[derive(Debug, Serialize)]
pub struct ReadFileOutput {
    /// The resolved, absolute path that was read.
    pub path: String,
    /// The file's UTF-8 contents.
    pub contents: String,
}

/// Errors that [`ReadFile`] can produce.
#[derive(Debug, thiserror::Error)]
pub enum ReadFileError {
    #[error("could not determine the current working directory: {0}")]
    Cwd(std::io::Error),
    #[error("'{0}' is outside the current working directory")]
    OutsideCwd(String),
    #[error("'{0}' is not a regular file")]
    NotAFile(String),
    #[error("could not read '{path}': {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
}

impl From<super::PathError> for ReadFileError {
    fn from(e: super::PathError) -> Self {
        match e {
            super::PathError::Cwd(io) => ReadFileError::Cwd(io),
            super::PathError::OutsideCwd(p) => ReadFileError::OutsideCwd(p),
            // A non-existent path fails canonicalization; surface it as I/O.
            super::PathError::Resolve { path, source } => ReadFileError::Io { path, source },
        }
    }
}

/// Reads a text file located anywhere within the current working directory.
#[derive(Debug, Clone, Default)]
pub struct ReadFile;

impl Tool for ReadFile {
    const NAME: &'static str = "read_file";

    type Error = ReadFileError;
    type Args = ReadFileArgs;
    type Output = ReadFileOutput;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Read a text file in the working directory.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "File path, e.g. src/main.rs" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let resolved = super::contain(&args.path)?;

        if !resolved.is_file() {
            return Err(ReadFileError::NotAFile(args.path.clone()));
        }

        let contents = std::fs::read_to_string(&resolved).map_err(|source| ReadFileError::Io {
            path: args.path.clone(),
            source,
        })?;

        Ok(ReadFileOutput {
            path: resolved.display().to_string(),
            contents,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `cargo test` runs with the working directory set to the package root,
    // so `Cargo.toml` / `src` are inside the CWD and `/etc/hosts` is outside.
    async fn read(path: &str) -> Result<ReadFileOutput, ReadFileError> {
        ReadFile
            .call(ReadFileArgs {
                path: path.to_string(),
            })
            .await
    }

    #[tokio::test]
    async fn reads_a_file_inside_cwd() {
        let out = read("Cargo.toml").await.expect("should read Cargo.toml");
        assert!(out.contents.contains("[package]"));
    }

    #[tokio::test]
    async fn rejects_absolute_path_outside_cwd() {
        let err = read("/etc/hosts").await.unwrap_err();
        assert!(matches!(err, ReadFileError::OutsideCwd(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn rejects_traversal_escaping_cwd() {
        // Enough `..` to reach the filesystem root from any depth (extra `..` at
        // `/` stay at `/`), then back down to a file that exists outside the CWD.
        // Canonicalization resolves the `..` before the containment check.
        let depth = std::env::current_dir().unwrap().components().count();
        let path = format!("{}etc/hosts", "../".repeat(depth));
        let err = read(&path).await.unwrap_err();
        assert!(
            matches!(err, ReadFileError::OutsideCwd(_)),
            "got {err:?} for {path}"
        );
    }

    #[tokio::test]
    async fn rejects_a_directory() {
        let err = read("src").await.unwrap_err();
        assert!(matches!(err, ReadFileError::NotAFile(_)), "got {err:?}");
    }

    #[tokio::test]
    async fn missing_file_is_an_io_error() {
        let err = read("does-not-exist-xyz.txt").await.unwrap_err();
        assert!(matches!(err, ReadFileError::Io { .. }), "got {err:?}");
    }
}
