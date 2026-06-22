//! Encapsulates all interaction with the local LLM behind an [`LLMClient`].
//!
//! Callers don't need to know which rig client, endpoint, model, or credentials
//! are in play — they build one with [`LLMClient::from_env`] and call
//! [`LLMClient::ask`].

use std::env;

use anyhow::{Context, Result};
use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::completion::{Prompt, PromptError};
use rig::providers::openai::{CompletionModel, CompletionsClient};

/// OpenAI-compatible endpoint exposed by the local lemonade / FastFlowLM server.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8001/v1";
/// Model to talk to. Overridable with `FLM_MODEL`.
const DEFAULT_MODEL: &str = "qwen3.5-9b-FLM";
/// System prompt that shapes the assistant's behavior.
const PREAMBLE: &str = "You are an assistant embedded in a Unix shell. Be concise.";

/// Connection settings for the local LLM.
///
/// Build the defaults (plus any `FLM_*` env overrides) with [`LlmConfig::from_env`],
/// then mutate fields to apply higher-priority overrides (e.g. CLI flags).
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
}

impl LlmConfig {
    /// Resolve settings from the environment, falling back to sensible defaults
    /// for the local lemonade / FastFlowLM server.
    ///
    /// Honors `FLM_BASE_URL`, `FLM_MODEL`, and `FLM_API_KEY`.
    pub fn from_env() -> Self {
        Self {
            base_url: env::var("FLM_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            model: env::var("FLM_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            // The local server ignores the key, but rig requires one.
            api_key: env::var("FLM_API_KEY").unwrap_or_else(|_| "local".to_string()),
        }
    }
}

/// A handle to the local LLM.
///
/// Hides the underlying rig [`CompletionsClient`] and all configuration
/// (endpoint, model, credentials, system preamble) behind a small surface.
pub struct LLMClient {
    agent: Agent<CompletionModel>,
    model: String,
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

        let agent = client.agent(&model).preamble(PREAMBLE).build();

        Ok(Self { agent, model })
    }

    /// The model name this client is talking to.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Send a prompt to the LLM and return its reply.
    pub async fn ask(&self, prompt: &str) -> Result<String, PromptError> {
        self.agent.prompt(prompt).await
    }
}
