use serde::{Deserialize, Serialize};
use windows::{Foundation::Uri, Web::Http::HttpClient as WindowsHttpClient, core::HSTRING};

const USER_AGENT: &str = concat!(env!("CARGO_PKG_NAME"), "/", env!("CARGO_PKG_VERSION"));
const SEARXNG_URL: &str = "http://localhost:8080/search?format=json&q=";

pub struct WebSearchHandler {
    #[cfg(windows)]
    client: WindowsHttpClient,
}

impl WebSearchHandler {
    #[cfg(windows)]
    pub async fn new() -> Self {
        let client = WindowsHttpClient::new().unwrap();

        let headers = client.DefaultRequestHeaders().unwrap();
        headers
            .UserAgent()
            .unwrap()
            .ParseAdd(&HSTRING::from(USER_AGENT))
            .unwrap();
        headers
            .Accept()
            .unwrap()
            .ParseAdd(&HSTRING::from("application/json"))
            .unwrap();

        Self { client }
    }

    pub async fn query(&self, query: &str) -> SearchResults {
        let encoded_query = Uri::EscapeComponent(&HSTRING::from(query)).unwrap();
        let full_url = format!("{SEARXNG_URL}{encoded_query}");

        let response = self
            .client
            .GetStringAsync(&Uri::CreateUri(&HSTRING::from(full_url)).unwrap())
            .unwrap()
            .await
            .unwrap()
            .to_string_lossy();

        serde_json::from_str::<SearchResults>(&response).unwrap()
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SearchResults {
    query: String,
    results: Vec<SearchResult>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct SearchResult {
    title: String,
    content: String,
    url: String,
}
