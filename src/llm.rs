//! Encapsulates all interaction with the local LLM behind an [`LLMClient`].
//!
//! Callers don't need to know which rig client, endpoint, model, or credentials
//! are in play — they build one with [`LLMClient::from_env`] and call
//! [`LLMClient::ask`].

use std::env;

use anyhow::{Context, Result};
use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::agent::PromptRequest;
use rig::completion::{Chat, Message, PromptError};
use rig::providers::openai::{CompletionModel, CompletionsClient};

use crate::tools::{Explore, Grep, ReadFile};

/// How many tool-call/response turns a single prompt may take before stopping.
const MAX_TURNS: usize = 5;

/// OpenAI-compatible endpoint exposed by the local lemonade / FastFlowLM server.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8001/v1";
/// System prompt that shapes the assistant's behavior.
const PREAMBLE: &str = "You are an assistant embedded in a Unix shell. Be concise. \
     You can read text files within the current working directory with the `read_file` tool; \
     use it when answering questions about local files.";

/// Connection settings for the local LLM.
///
/// Build one with [`LlmConfig::resolve`], which layers CLI overrides, `FLM_*`
/// env vars, the model currently loaded in FastFlowLM, and built-in defaults.
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl LlmConfig {
    /// Resolve settings from CLI overrides, the environment, and the server.
    ///
    /// Base URL and key come from the override / `FLM_BASE_URL` / `FLM_API_KEY` /
    /// defaults. The model is always whatever is currently loaded in FastFlowLM
    /// (via `/api/ps`); if that can't be detected (server down, or nothing
    /// loaded), this fails rather than guessing.
    pub async fn resolve(base_url_override: Option<String>) -> Result<Self> {
        let base_url = base_url_override
            .or_else(|| env::var("FLM_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        // The local server ignores the key, but rig requires one.
        let api_key = env::var("FLM_API_KEY").unwrap_or_else(|_| "local".to_string());

        let model = running_model(&base_url).await?;
        tracing::debug!(%model, "using model currently loaded in fastflowlm");

        Ok(Self {
            base_url,
            model,
            api_key,
        })
    }
}

/// Ask FastFlowLM which model is currently loaded, via its Ollama-style
/// `/api/ps` endpoint (sibling of the `/v1` API root).
///
/// Fails if the server is unreachable or reports no loaded model.
async fn running_model(base_url: &str) -> Result<String> {
    // `/api/ps` lives at the host root, not under `/v1`.
    let host = base_url
        .strip_suffix("/v1")
        .or_else(|| base_url.strip_suffix("/v1/"))
        .unwrap_or(base_url)
        .trim_end_matches('/');
    let url = format!("{host}/api/ps");

    let json: serde_json::Value = reqwest::get(&url)
        .await
        .with_context(|| format!("could not reach FastFlowLM at {url} — is it running?"))?
        .json()
        .await
        .with_context(|| format!("unexpected response from {url}"))?;

    json["models"][0]["name"]
        .as_str()
        .map(String::from)
        .with_context(|| format!("no model is currently loaded in FastFlowLM (per {url})"))
}

/// A handle to the local LLM.
///
/// Hides the underlying rig [`CompletionsClient`] and all configuration
/// (endpoint, model, credentials, system preamble) behind a small surface.
///
/// Keeps the running conversation in `history` so the model remembers earlier
/// turns within a session.
pub struct LLMClient {
    agent: Agent<CompletionModel>,
    model: String,
    base_url: String,
    history: Vec<Message>,
    /// Fixed per-request overhead: the system preamble + tool schemas that are
    /// sent on every turn. This is the floor the context never drops below
    /// (`/clear` resets to it, not to zero). Measured once against the server.
    baseline: u64,
    /// The server's context window (`max_kv_token_capacity`), once detected.
    context_window: Option<u64>,
}

impl LLMClient {
    /// Build a client from the given [`LlmConfig`].
    pub fn new(config: LlmConfig) -> Result<Self> {
        let LlmConfig {
            base_url,
            model,
            api_key,
        } = config;

        tracing::debug!(%base_url, %model, "connecting to LLM");

        // FastFlowLM speaks the Chat Completions API (`/v1/chat/completions`), so
        // we use rig's Completions client rather than the default Responses-API
        // client.
        let client = CompletionsClient::builder()
            .api_key(api_key)
            .base_url(&base_url)
            .build()
            .with_context(|| format!("building LLM client for {base_url}"))?;

        let agent = client
            .agent(&model)
            .preamble(PREAMBLE)
            .tool(ReadFile)
            .tool(Explore)
            .tool(Grep)
            .default_max_turns(MAX_TURNS)
            .build();

        Ok(Self {
            agent,
            model,
            base_url,
            history: Vec::new(),
            baseline: 0,
            context_window: None,
        })
    }

    /// The model name this client is talking to.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a prompt to the LLM and return its reply.
    ///
    /// The prompt and the model's response (including any tool calls/results)
    /// are appended to the running conversation, so later prompts can refer back
    /// to earlier ones. The conversation's token usage is updated from the
    /// server's reported usage.
    pub async fn ask(&mut self, prompt: &str) -> Result<String, PromptError> {
        // `chat` feeds the prior history back in and appends this turn's prompt +
        // assistant/tool messages to it.
        self.agent.chat(prompt, &mut self.history).await
    }

    /// Estimated context usage (baseline + conversation) and the window if known.
    ///
    /// The conversation portion is estimated locally from the messages we hold
    /// (~4 chars/token), rather than from the server's per-turn token counts:
    /// FastFlowLM's prefix caching makes those report only deltas, which can't be
    /// turned into a reliable absolute total.
    pub fn context_usage(&self) -> (u64, Option<u64>) {
        (self.baseline + self.history_tokens(), self.context_window)
    }

    /// Rough token estimate of the held conversation history.
    fn history_tokens(&self) -> u64 {
        let chars: usize = self
            .history
            .iter()
            .filter_map(|m| serde_json::to_string(m).ok())
            .map(|s| s.len())
            .sum();
        (chars / 4) as u64
    }

    /// Forget the conversation so far. Usage falls back to the fixed baseline
    /// (the system prompt + tool schemas are still sent on every request).
    pub fn clear(&mut self) {
        self.history.clear();
    }

    /// Prepare the context meter: measure the fixed per-request overhead and the
    /// server's context window. Best-effort — failures leave the meter at zero /
    /// "window unknown". Also warms the server's prefix cache.
    pub async fn prime(&mut self) {
        // A throwaway call with empty history: its prompt is just the preamble +
        // tool schemas, so its input token count is our fixed baseline.
        let probe = PromptRequest::from_agent(&self.agent, ".")
            .with_history(Vec::<Message>::new())
            .extended_details()
            .await;
        if let Ok(response) = probe {
            self.baseline = response.usage.input_tokens;
        }

        self.context_window = context_capacity(&self.base_url, &self.model).await;
        tracing::debug!(
            baseline = self.baseline,
            capacity = ?self.context_window,
            "primed context meter"
        );
    }
}

/// Read the server's `max_kv_token_capacity` from a tiny streaming completion
/// (the field only appears in the streaming usage block).
async fn context_capacity(base_url: &str, model: &str) -> Option<u64> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": true,
    });
    let text = reqwest::Client::new()
        .post(url)
        .json(&body)
        .send()
        .await
        .ok()?
        .text()
        .await
        .ok()?;

    let key = "\"max_kv_token_capacity\":";
    let start = text.find(key)? + key.len();
    let rest = &text[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}
