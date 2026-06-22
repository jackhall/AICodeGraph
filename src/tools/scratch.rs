//! The shared `ai-scratch/` directory where tools stash bulky output.
//!
//! It lives inside the current working directory, so files written here are
//! reachable by the CWD-confined [`ReadFile`](super::ReadFile) and
//! [`Grep`](super::Grep) tools. It is wiped when the shell starts and exits.

use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

/// Scratch directory name, relative to the current working directory.
pub const SCRATCH_DIR: &str = "ai-scratch";

/// Monotonic counter used to name scratch files within a single run.
static COUNTER: AtomicU64 = AtomicU64::new(0);

fn dir() -> PathBuf {
    PathBuf::from(SCRATCH_DIR)
}

/// Remove every scratch file and recreate an empty scratch directory.
///
/// Called when the shell starts and when it exits.
pub fn reset() -> io::Result<()> {
    let dir = dir();
    if dir.exists() {
        fs::remove_dir_all(&dir)?;
    }
    fs::create_dir_all(&dir)?;
    COUNTER.store(0, Ordering::Relaxed);
    Ok(())
}

/// Write `contents` to a uniquely-named `<prefix>-<n>.<ext>` file in the scratch
/// directory, returning its path relative to the CWD (suitable to hand to the
/// `read_file` / `grep` tools).
pub fn write_unique(prefix: &str, ext: &str, contents: &str) -> io::Result<PathBuf> {
    let dir = dir();
    fs::create_dir_all(&dir)?;
    loop {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{prefix}-{n}.{ext}"));
        // `create_new` fails if the file already exists, giving us atomic
        // uniqueness even if the counter collides with a leftover file.
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                file.write_all(contents.as_bytes())?;
                return Ok(path);
            }
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}
