//! Encapsulates all interaction with the local LLM behind an [`LLMClient`].
//!
//! Callers don't need to know which rig client, endpoint, model, or credentials
//! are in play — they build one with [`LLMClient::from_env`] and call
//! [`LLMClient::ask`].

use std::env;

use anyhow::{Context, Result};
use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::completion::message::{AssistantContent, ToolResultContent, UserContent};
use rig::completion::{Chat, Message, PromptError};
use rig::tool::Tool;
use rig::providers::openai::{CompletionModel, CompletionsClient};

use crate::tools::{Explore, Grep, ReadFile};

/// How many tool-call/response turns a single prompt may take before stopping.
const MAX_TURNS: usize = 5;

/// OpenAI-compatible endpoint exposed by the local lemonade / FastFlowLM server.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8001/v1";
/// System prompt that shapes the assistant's behavior.
const PREAMBLE: &str = "You are an assistant embedded in a Unix shell. Be concise. \
     Use the file tools to answer questions about files in the working directory.";

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
    /// rig's running conversation (for chat continuity / tool loop).
    history: Vec<Message>,
    /// OpenAI-format tool schemas, kept for the context probes.
    tools: Vec<serde_json::Value>,
    /// The full conversation (incl. tool calls/results) in OpenAI message form,
    /// converted from rig's history as it grows. The calibration probe replays
    /// this, so it mirrors what rig actually sends.
    transcript: Vec<serde_json::Value>,
    /// How many of `history`'s messages have already been folded into `transcript`.
    recorded: usize,
    /// Running character count of [`Self::transcript`].
    conv_chars: usize,
    /// Calibrated chars-per-token ratio for the conversation estimate.
    chars_per_token: f64,
    /// Turns since the last calibration probe.
    turns_since_calibration: u32,
    /// Fixed per-request overhead: the system preamble + tool schemas that are
    /// sent on every turn. This is the floor the context never drops below
    /// (`/clear` resets to it, not to zero). Measured once against the server.
    baseline: u64,
    /// The server's context window (`max_kv_token_capacity`), once detected.
    context_window: Option<u64>,
}

/// Run a calibration probe every this many turns.
const CALIBRATE_EVERY: u32 = 4;
/// Chars-per-token assumption until the first calibration refines it.
const DEFAULT_CHARS_PER_TOKEN: f64 = 3.0;

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
            tools: Vec::new(),
            transcript: Vec::new(),
            recorded: 0,
            conv_chars: 0,
            chars_per_token: DEFAULT_CHARS_PER_TOKEN,
            turns_since_calibration: 0,
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
        let reply = self.agent.chat(prompt, &mut self.history).await?;

        // Fold the new messages — including any tool calls/results rig added this
        // turn — into the transcript, so the meter accounts for tool context too.
        let new_messages = self.history.len() - self.recorded;
        for msg in &self.history[self.recorded..] {
            for om in message_to_openai(msg) {
                self.conv_chars += serde_json::to_string(&om).map(|s| s.len()).unwrap_or(0);
                self.transcript.push(om);
            }
        }
        self.recorded = self.history.len();

        // Recalibrate periodically, but also right after a tool-using turn (more
        // than the user+assistant pair was added) since those move context most.
        self.turns_since_calibration += 1;
        if self.turns_since_calibration >= CALIBRATE_EVERY || new_messages > 2 {
            self.calibrate().await;
            self.turns_since_calibration = 0;
        }

        Ok(reply)
    }

    /// Estimated context usage (baseline + conversation) and the window if known.
    ///
    /// The conversation portion is `conv_chars / chars_per_token`, where the ratio
    /// is recalibrated against the server every [`CALIBRATE_EVERY`] turns (see
    /// [`Self::calibrate`]). The baseline is exact (measured in [`Self::prime`]).
    pub fn context_usage(&self) -> (u64, Option<u64>) {
        let conv = (self.conv_chars as f64 / self.chars_per_token).round() as u64;
        (self.baseline + conv, self.context_window)
    }

    /// Forget the conversation so far. Usage falls back to the fixed baseline
    /// (the system prompt + tool schemas are still sent on every request). The
    /// calibrated ratio is kept.
    pub fn clear(&mut self) {
        self.history.clear();
        self.transcript.clear();
        self.recorded = 0;
        self.conv_chars = 0;
        self.turns_since_calibration = 0;
    }

    /// Refine `chars_per_token` against ground truth: probe the server with the
    /// recorded transcript and read the true KV occupancy (`active_kv_tokens`).
    async fn calibrate(&mut self) {
        if self.conv_chars == 0 {
            return;
        }
        let mut messages = vec![serde_json::json!({ "role": "system", "content": PREAMBLE })];
        messages.extend(self.transcript.iter().cloned());

        if let Some((_, active_kv)) = probe(&self.base_url, &self.model, &messages, &self.tools).await
        {
            if active_kv > self.baseline {
                let real_conv = (active_kv - self.baseline) as f64;
                self.chars_per_token = (self.conv_chars as f64 / real_conv).clamp(1.5, 8.0);
                tracing::debug!(
                    active_kv,
                    ratio = self.chars_per_token,
                    "calibrated context meter"
                );
            }
        }
    }

    /// Prepare the context meter: probe the server for its context window and the
    /// fixed per-request overhead (system preamble + tool schemas).
    ///
    /// We send a probe mirroring a real request (same preamble + tools) and read
    /// `active_kv_tokens` — the true KV occupancy, which is stable regardless of
    /// FastFlowLM's prefix caching (`prompt_tokens` is not).
    pub async fn prime(&mut self) {
        self.tools = [
            ReadFile.definition(String::new()).await,
            Explore.definition(String::new()).await,
            Grep.definition(String::new()).await,
        ]
        .iter()
        .filter_map(|d| serde_json::to_value(d).ok())
        .map(|d| serde_json::json!({ "type": "function", "function": d }))
        .collect();

        let messages = vec![
            serde_json::json!({ "role": "system", "content": PREAMBLE }),
            serde_json::json!({ "role": "user", "content": "hi" }),
        ];
        if let Some((capacity, baseline)) =
            probe(&self.base_url, &self.model, &messages, &self.tools).await
        {
            self.context_window = Some(capacity);
            self.baseline = baseline;
        }
        tracing::debug!(
            baseline = self.baseline,
            capacity = ?self.context_window,
            "primed context meter"
        );
    }
}

/// Convert one rig [`Message`] into the OpenAI chat message(s) it corresponds to,
/// preserving tool calls and tool results (which carry most of the token weight).
/// A single rig message can expand to several (e.g. a user turn bundling several
/// tool results), so this returns a `Vec`.
fn message_to_openai(msg: &Message) -> Vec<serde_json::Value> {
    use serde_json::json;
    match msg {
        Message::System { content } => vec![json!({ "role": "system", "content": content })],

        Message::User { content } => {
            let mut out = Vec::new();
            let mut text = Vec::new();
            for c in content.iter() {
                match c {
                    UserContent::Text(t) => text.push(t.text().to_string()),
                    // Tool results are their own `tool` role message in OpenAI form.
                    UserContent::ToolResult(tr) => {
                        let body = tr
                            .content
                            .iter()
                            .filter_map(|rc| match rc {
                                ToolResultContent::Text(t) => Some(t.text().to_string()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        out.push(json!({
                            "role": "tool",
                            "tool_call_id": tr.id,
                            "content": body,
                        }));
                    }
                    _ => {}
                }
            }
            if !text.is_empty() {
                out.insert(0, json!({ "role": "user", "content": text.join("\n") }));
            }
            out
        }

        Message::Assistant { content, .. } => {
            let mut text = Vec::new();
            let mut tool_calls = Vec::new();
            for c in content.iter() {
                match c {
                    AssistantContent::Text(t) => text.push(t.text().to_string()),
                    AssistantContent::ToolCall(tc) => tool_calls.push(json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": serde_json::to_string(&tc.function.arguments).unwrap_or_default(),
                        },
                    })),
                    _ => {}
                }
            }
            let mut m = json!({ "role": "assistant", "content": text.join("\n") });
            if !tool_calls.is_empty() {
                m["tool_calls"] = json!(tool_calls);
            }
            vec![m]
        }
    }
}

/// Probe FastFlowLM with the given messages + tools and read
/// `(max_kv_token_capacity, active_kv_tokens)` from the streaming usage block —
/// i.e. the context window and the request's true KV occupancy. Both fields only
/// appear when streaming. `max_tokens: 1` keeps the generation negligible.
async fn probe(
    base_url: &str,
    model: &str,
    messages: &[serde_json::Value],
    tools: &[serde_json::Value],
) -> Option<(u64, u64)> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    let body = serde_json::json!({
        "model": model,
        "messages": messages,
        "tools": tools,
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

    let capacity = parse_usage_field(&text, "max_kv_token_capacity")?;
    let baseline = parse_usage_field(&text, "active_kv_tokens").unwrap_or(0);
    Some((capacity, baseline))
}

/// Pull an integer `"key":N` value out of a raw SSE/JSON response body.
fn parse_usage_field(body: &str, key: &str) -> Option<u64> {
    let needle = format!("\"{key}\":");
    let start = body.find(&needle)? + needle.len();
    let rest = &body[start..];
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}
