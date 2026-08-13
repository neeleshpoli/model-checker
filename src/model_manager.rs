use foundry_local_sdk::{
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestUserMessage, FoundryLocalConfig, FoundryLocalError, FoundryLocalManager,
    openai::ChatClient,
};
use tokio::sync::mpsc::{Sender, UnboundedSender};
use tokio_stream::StreamExt;

const MODEL_NAME: &str = "phi-4-mini";
const MAX_TOKENS: u32 = 1024;

pub struct ModelManager {
    client: ChatClient,
}

impl ModelManager {
    pub async fn new(
        ep_download: impl FnMut(&str, f64) + Send + 'static,
        model_download: impl FnMut(f64) + Send + 'static,
    ) -> Result<Self, FoundryLocalError> {
        let manager = FoundryLocalManager::create(FoundryLocalConfig::new("modelchecker"))?;
        manager
            .download_and_register_eps_with_progress(None, ep_download)
            .await?;

        let model = manager.catalog().get_model(MODEL_NAME).await?;

        if !model.is_cached().await? {
            model.download(Some(model_download)).await?;
        }

        let client = model
            .create_chat_client()
            .temperature(0.7)
            .max_tokens(MAX_TOKENS);

        model.load().await?;
        Ok(Self { client })
    }

    pub async fn ask(&self, message: String, sender: UnboundedSender<String>) -> Result<(), FoundryLocalError> {
        let messages: Vec<ChatCompletionRequestMessage> = vec![
            ChatCompletionRequestSystemMessage::from("You are a helpful assistant.").into(),
            ChatCompletionRequestUserMessage::from(message).into(),
        ];

        let mut stream = self.client.complete_streaming_chat(&messages, None).await?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;

            if let Some(choice) = chunk.choices.first() {
                if let Some(ref content) = choice.delta.content {
                    print!("{content}");

                    sender.send(content.clone()).unwrap();
                }
            }
        }

        Ok(())
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
