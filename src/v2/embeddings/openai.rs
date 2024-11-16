use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::v2::commons::Embedding;

const OPENAI_EMBEDDINGS_ENDPOINT: &str = "https://api.openai.com/v1/embeddings";
const OPENAI_EMBEDDINGS_MODEL: &str = "text-embedding-3-small";

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    pub model: String,
    pub input: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
}

/// Represents the OpenAI Embeddings provider
pub struct OpenAIEmbeddings {
    config: OpenAIConfig
}

/// Defaults to the "text-embedding-3-small" model
/// The API key can be set in the OPENAI_API_KEY environment variable
pub struct OpenAIConfig {
    pub api_endpoint: String,
    pub api_key: String,
    pub model: String,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            api_endpoint: OPENAI_EMBEDDINGS_ENDPOINT.to_string(),
            api_key: std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY env is not set"),
            model: OPENAI_EMBEDDINGS_MODEL.to_string(),
        }
    }
}

impl OpenAIEmbeddings {
    pub fn new(config: OpenAIConfig) -> Self {
        Self { config }
    }

    async fn post<T: Serialize>(&self, json_body: T) -> anyhow::Result<Value> {
        let client = reqwest::Client::new();
        let res = client.post(&self.config.api_endpoint)
            .body("the exact body that is sent")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .json(&json_body)
            .send()
            .await?;

        match res.error_for_status() {
            Ok(res) => {
                Ok(res.json().await?)
            },
            Err(e) => {
                Err(e.into())
            }
        }
    }

    pub async fn embed(&self, docs: Vec<String>) -> anyhow::Result<Vec<Embedding>> {
        let mut embeddings = Vec::new();
        for doc in docs {
            let req = EmbeddingRequest {
                model: self.config.model.clone(),
                input: doc,
            };
            let res = self.post(req).await?;
            let body = serde_json::from_value::<EmbeddingResponse>(res)?;
            embeddings.push(body.data[0].embedding.clone());
        }

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use crate::v2::collection::CollectionEntries;
    use super::*;
    use crate::v2::ChromaClient;

    #[tokio::test]
    async fn test_openai_embeddings() {

        let client = ChromaClient::new(Default::default());
        let collection = client
            .get_or_create_collection("open-ai-test-collection".to_string(), None)
            .await
            .unwrap();
        let openai_embeddings = OpenAIEmbeddings::new(Default::default());

        let docs = vec![
            "Once upon a time there was a frog".to_string(),
            "Once upon a time there was a cow".to_string(),
            "Once upon a time there was a wolverine".to_string(),
        ];

        let collection_entries = CollectionEntries {
            ids: vec!["test1".to_string(), "test2".to_string(), "test3".to_string()],
            metadatas: None,
            documents: Some(docs),
            embeddings: None,
        };

        collection
            .upsert(
                collection_entries, 
                Some(openai_embeddings),
            )
            .await
            .unwrap();
    }
}
