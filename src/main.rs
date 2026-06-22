//! A minimal shell that forwards every command to the calling shell, unless
//! the line is prefixed with `@`, in which case it is sent to a local LLM
//! (FastFlowLM / lemonade). All LLM interaction lives in [`llm::LLMClient`].

mod llm;
mod tools;

use std::env;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use clap::Parser;
use linefeed::{DefaultTerminal, Interface, ReadResult, Signal};
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
    let mut llm = LLMClient::new(config).context("failed to build LLM client")?;

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
        "shell ready — '@ <text>' asks the LLM; '/clear' resets it; Ctrl-D or 'exit' to quit"
    );

    // Measure the baseline (system prompt + tools) and context window for the bar.
    llm.prime().await;

    let history_path = env::var_os("HOME").map(|h| PathBuf::from(h).join(".aicg_history"));
    let reader = LineReader::new(history_path.as_deref())?;

    loop {
        // Show how full the conversation's context is, just above the prompt.
        // Skipped when stdout isn't a terminal (e.g. piped) to keep output clean.
        if io::stdout().is_terminal() {
            let (used, capacity) = llm.context_usage();
            println!("{}", context_bar(used, capacity));
        }

        let line = match reader.read(&format!("{} ❯ ", cwd.display())) {
            Some(line) => line,
            None => break, // EOF (Ctrl-D)
        };

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        reader.add_history(line);
        if line == "exit" || line == "quit" {
            break;
        }

        // `/clear` -> forget the conversation so far.
        if line == "/clear" {
            llm.clear();
            println!("(conversation cleared)");
            continue;
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

    if let Some(path) = &history_path {
        reader.save_history(path);
    }

    Ok(())
}

/// Line input: full readline honoring `~/.inputrc` on a terminal, or plain stdin
/// when piped/non-interactive (linefeed needs a real tty).
enum LineReader {
    Interactive(Interface<DefaultTerminal>),
    Plain(io::Stdin),
}

impl LineReader {
    fn new(history: Option<&Path>) -> Result<Self> {
        if !io::stdin().is_terminal() {
            return Ok(LineReader::Plain(io::stdin()));
        }
        // linefeed reads the standard readline config itself ($INPUTRC, then
        // ~/.inputrc): editing mode, key bindings, conditionals, and all.
        let iface = Interface::new("aicg").context("initializing line editor")?;
        iface.set_report_signal(Signal::Interrupt, true); // Ctrl-C cancels the line
        if let Some(path) = history {
            let _ = iface.load_history(path);
        }
        Ok(LineReader::Interactive(iface))
    }

    /// Read one line. `None` means EOF (exit); a Ctrl-C returns an empty line so
    /// the caller just shows a fresh prompt.
    fn read(&self, prompt: &str) -> Option<String> {
        match self {
            LineReader::Interactive(iface) => {
                iface.set_prompt(prompt).ok()?;
                match iface.read_line() {
                    Ok(ReadResult::Input(line)) => Some(line),
                    Ok(ReadResult::Signal(_)) => Some(String::new()),
                    Ok(ReadResult::Eof) => None,
                    Err(e) => {
                        tracing::error!("read error: {e}");
                        None
                    }
                }
            }
            LineReader::Plain(stdin) => {
                print!("{prompt}");
                io::stdout().flush().ok();
                let mut line = String::new();
                match stdin.read_line(&mut line) {
                    Ok(0) => None,
                    Ok(_) => Some(line),
                    Err(e) => {
                        tracing::error!("read error: {e}");
                        None
                    }
                }
            }
        }
    }

    fn add_history(&self, line: &str) {
        if let LineReader::Interactive(iface) = self {
            iface.add_history_unique(line.to_string());
        }
    }

    fn save_history(&self, path: &Path) {
        if let LineReader::Interactive(iface) = self {
            let _ = iface.save_history(path);
        }
    }
}

/// Render a context-fill bar: `ctx [████░░░░] 1234/4096 (30%)`, colored green →
/// yellow → red as it fills. If the window is unknown, just shows the count.
fn context_bar(used: u64, capacity: Option<u64>) -> String {
    const WIDTH: usize = 24;
    match capacity {
        Some(cap) if cap > 0 => {
            let frac = (used as f64 / cap as f64).clamp(0.0, 1.0);
            let filled = ((frac * WIDTH as f64).round() as usize).min(WIDTH);
            let color = if frac < 0.7 {
                "\x1b[32m" // green
            } else if frac < 0.9 {
                "\x1b[33m" // yellow
            } else {
                "\x1b[31m" // red
            };
            let bar = format!(
                "{color}{}\x1b[0m{}",
                "█".repeat(filled),
                "░".repeat(WIDTH - filled)
            );
            format!("ctx [{bar}] {used}/{cap} ({:.0}%)", frac * 100.0)
        }
        _ => format!("ctx [{used} tokens, window unknown]"),
    }
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
