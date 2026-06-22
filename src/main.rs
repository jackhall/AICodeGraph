//! A minimal shell that forwards every command to the calling shell, unless
//! the line is prefixed with `@`, in which case it is sent to a local LLM
//! (FastFlowLM / lemonade). All LLM interaction lives in [`llm::LLMClient`].

mod llm;
mod tools;

use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::Parser;
use tracing_subscriber::EnvFilter;

use llm::{LLMClient, LlmConfig};

/// A shell that forwards commands to your shell, or asks a local LLM when the
/// line is prefixed with `@`.
#[derive(Debug, Parser)]
#[command(name = "aicg", version, about)]
struct Cli {
    /// Base URL of the OpenAI-compatible LLM server (overrides $FLM_BASE_URL).
    #[arg(long)]
    base_url: Option<String>,

    /// Ask the LLM a single prompt, print the reply, and exit (no interactive shell).
    #[arg(short, long, value_name = "PROMPT")]
    ask: Option<String>,

    /// Increase log verbosity (-v for debug, -vv for trace).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    // Start fresh, and ensure the scratch dir is wiped on every exit path.
    clear_scratch();
    let _scratch_guard = ScratchGuard;

    // The model is whatever is currently loaded in fastflowlm.
    let config = LlmConfig::resolve(cli.base_url).await?;
    let llm = LLMClient::new(config).context("failed to build LLM client")?;

    // One-shot mode: answer a single prompt and exit without the interactive shell.
    if let Some(prompt) = cli.ask {
        tracing::debug!(prompt, "one-shot ask");
        let reply = llm.ask(&prompt).await.context("LLM request failed")?;
        println!("{reply}");
        return Ok(());
    }

    let shell = env::var("SHELL").unwrap_or_else(|_| "/bin/sh".to_string());
    let mut cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

    tracing::info!(
        %shell,
        model = llm.model(),
        "shell ready — '@ <text>' asks the LLM; Ctrl-D or 'exit' to quit"
    );


    let stdin = io::stdin();
    loop {
        print!("{} ❯ ", cwd.display());
        io::stdout().flush().ok();

        let mut line = String::new();
        match stdin.read_line(&mut line) {
            Ok(0) => {
                println!();
                break; // EOF (Ctrl-D)
            }
            Ok(_) => {}
            Err(e) => {
                tracing::error!("read error: {e}");
                break;
            }
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "exit" || line == "quit" {
            break;
        }

        // `@` -> ask the local LLM.
        if let Some(query) = line.strip_prefix('@') {
            let query = query.trim();
            if query.is_empty() {
                continue;
            }
            tracing::debug!(query, "asking LLM");
            match llm.ask(query).await {
                Ok(resp) => println!("{resp}"),
                Err(e) => tracing::error!("llm error: {e}"),
            }
            continue;
        }

        // `cd` is handled in-process so the directory persists across commands.
        if line == "cd" || line.starts_with("cd ") {
            let target = line[2..].trim();
            let new_dir = resolve_cd(&cwd, target);
            match env::set_current_dir(&new_dir) {
                Ok(()) => cwd = env::current_dir().unwrap_or(new_dir),
                Err(e) => tracing::warn!("cd: {}: {e}", new_dir.display()),
            }
            continue;
        }

        // Everything else is forwarded verbatim to the calling shell.
        tracing::debug!(command = line, "forwarding to shell");
        match Command::new(&shell)
            .arg("-c")
            .arg(line)
            .current_dir(&cwd)
            .status()
        {
            Ok(status) if !status.success() => {
                tracing::debug!(?status, "command exited non-zero");
            }
            Ok(_) => {}
            Err(e) => tracing::error!("failed to run command: {e}"),
        }
    }

    Ok(())
}

/// Wipe the scratch directory, logging (not failing) on error.
fn clear_scratch() {
    if let Err(e) = tools::scratch::reset() {
        tracing::warn!("could not reset {}: {e}", tools::scratch::SCRATCH_DIR);
    }
}

/// Clears the scratch directory when dropped, so it's wiped on every exit path
/// (normal return, `?` propagation, or unwinding panic).
struct ScratchGuard;

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        clear_scratch();
    }
}

/// Initialize tracing. `RUST_LOG` wins if set; otherwise `-v` flags raise our
/// crate's level while keeping noisy dependencies at `warn`.
fn init_tracing(verbose: u8) {
    let app_level = match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(format!("warn,aicg={app_level}")));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time()
        .init();
}

/// Resolve a `cd` target relative to `cwd`, expanding `~` / `~/...` via `$HOME`.
fn resolve_cd(cwd: &Path, target: &str) -> PathBuf {
    let home = env::var("HOME").ok();
    let expanded: PathBuf = if target.is_empty() || target == "~" {
        match &home {
            Some(h) => PathBuf::from(h),
            None => cwd.to_path_buf(),
        }
    } else if let Some(rest) = target.strip_prefix("~/") {
        match &home {
            Some(h) => PathBuf::from(h).join(rest),
            None => PathBuf::from(target),
        }
    } else {
        PathBuf::from(target)
    };

    if expanded.is_absolute() {
        expanded
    } else {
        cwd.join(expanded)
    }
}
