use std::path::PathBuf;

use foundry_local_sdk::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestToolMessage, ChatCompletionRequestUserMessage, ChatCompletionTools,
    ChatToolChoice, FinishReason, FoundryLocalConfig, FoundryLocalError, FoundryLocalManager,
    FunctionCall, openai::ChatClient,
};
use serde_json::json;
use tokio::fs::{self};
use tokio::sync::{
    mpsc::{Sender, UnboundedSender},
    watch,
};
use tokio_stream::StreamExt;

use crate::web_search::WebSearchHandler;

const MODEL_NAME: &str = "qwen2.5-7b";
const MAX_TOKENS: u32 = 512;

pub struct ModelManager {
    client: ChatClient,
    search_handler: WebSearchHandler,
}

impl ModelManager {
    pub async fn new(
        ep_download: watch::Sender<f64>,
        model_download: watch::Sender<f64>,
    ) -> Result<Self, FoundryLocalError> {
        let manager = FoundryLocalManager::create(FoundryLocalConfig::new("modelchecker"))?;
        manager
            .download_and_register_eps_with_progress(None, move |_, progress| {
                let _ = ep_download.send(progress);
            })
            .await?;

        let model = manager.catalog().get_model(MODEL_NAME).await?;

        if !model.is_cached().await? {
            model
                .download(Some(move |progress| {
                    let _ = model_download.send(progress);
                }))
                .await?;

            // Currently a broken inference_model.json is downloaded
            let path = model
                .path()
                .await
                .unwrap()
                .join(PathBuf::from("inference_model.json"));

            for provider in manager.discover_eps().unwrap() {
                if provider.name == "QNNExecutionProvider" && provider.is_registered {
                    fs::write(&path, serde_json::to_string_pretty(&json!({
                        "Name": "qwen2.5-7b-instruct-qnn-npu:1",
                        "PromptTemplate": {
                            "system": "<|im_start|>system\nYou are Qwen, an AI assistant developed by Alibaba. {Content}<|im_end|>",
                            "user": "<|im_start|>user\n{Content}<|im_end|>",
                            "assistant": "<|im_start|>assistant\n{Content}<|im_end|>",
                            "prompt": "<|im_start|>user\n{Content}<|im_end|>\n<|im_start|>assistant"
                        }
                    })).unwrap()).await.unwrap();
                }
            }
        }

        let client = model
            .create_chat_client()
            .temperature(0.7)
            .top_p(0.9)
            .frequency_penalty(1.12)
            .presence_penalty(1.05)
            .max_tokens(MAX_TOKENS)
            .tool_choice(ChatToolChoice::Auto);

        model.load().await?;

        let search_handler = WebSearchHandler::new().await;

        Ok(Self {
            client,
            search_handler,
        })
    }

    pub async fn ask(
        &self,
        message: String,
        sender: UnboundedSender<String>,
    ) -> Result<(), FoundryLocalError> {
        let tools: Vec<ChatCompletionTools> = serde_json::from_value(json!([{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Perform a web search to get more information or get current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for"
                        }
                    },
                    "required": ["query"]
                }
            }
        }])).unwrap();

        let mut messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessage::from(
                "You are a helpful AI assistant. If necessary, you can use any provided tools to answer the question.",
            )
            .into(),
            ChatCompletionRequestUserMessage::from(message).into(),
        ];

        'generation: loop {
            let mut stream = self
                .client
                .complete_streaming_chat(&messages, Some(&tools))
                .await?;

            let mut current_tool_id = String::new();
            let mut current_tool_name = String::new();
            let mut current_tool_args = String::new();
            let mut tool_calls = vec![];

            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;

                if let Some(choice) = chunk.choices.first() {
                    if let Some(ref content) = choice.delta.content {
                        sender.send(content.clone()).unwrap();
                    }

                    if let Some(ref tc_deltas) = choice.delta.tool_calls {
                        for call in tc_deltas {
                            if let Some(id) = &call.id {
                                current_tool_id.push_str(id);
                            }
                            if let Some(f) = &call.function {
                                if let Some(name) = &f.name {
                                    current_tool_name.push_str(name);
                                }
                                if let Some(args) = &f.arguments {
                                    current_tool_args.push_str(args);
                                }
                            }
                        }
                    }

                    if choice.finish_reason == Some(FinishReason::ToolCalls) {
                        let call = json!({
                            "id": current_tool_id.clone(),
                            "type": "function",
                            "function": {
                                "name": current_tool_name.clone(),
                                "arguments": current_tool_args.clone(),
                            }
                        });
                        tool_calls.push(call);
                    } else if choice.finish_reason == Some(FinishReason::Stop) {
                        break 'generation;
                    }
                }
            }

            // If no tool calls were requested during this turn, generation is complete
            if tool_calls.is_empty() {
                break 'generation;
            }

            // 1. Push the single assistant message with all tool calls
            let assistant_msg: ChatCompletionRequestMessage = serde_json::from_value(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": tool_calls,
            }))?;
            messages.push(assistant_msg);

            // 2. Execute tools and push individual tool result messages
            for call in &tool_calls {
                let name = call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();
                let arguments = call["function"]["arguments"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();

                let result = self
                    .handle_tool_call(&FunctionCall { name, arguments })
                    .await;

                messages.push(
                    ChatCompletionRequestToolMessage {
                        content: result.to_string().into(),
                        tool_call_id: call["id"].as_str().unwrap_or_default().to_string(),
                    }
                    .into(),
                );
            }
        }

        Ok(())
    }

    async fn handle_tool_call(&self, call: &FunctionCall) -> serde_json::Value {
        match call.name.as_str() {
            "web_search" => {
                let paramters = serde_json::from_str::<serde_json::Value>(&call.arguments).unwrap();
                let query = paramters["query"].as_str().unwrap();
                let results = self.search_handler.query(query).await;

                println!("Making web search! Query: {query}");

                serde_json::to_value(results).unwrap()
            }
            _ => json!({"error": format!("Unknown function: {}", call.name)}),
        }
    }
}

pub enum BackendCommands {
    Query {
        query: String,
        /// Send the message back to the UI
        sender: UnboundedSender<String>,
    },
    _InitializeModel {
        ep_download: Sender<f64>,
        model_download: Sender<f64>,
    },
}
