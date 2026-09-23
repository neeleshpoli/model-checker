use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::LazyLock,
};

pub mod allocator;

/// The base URL for downloading models.
const HUGGINGFACE_BASE_URL: &str = "https://huggingface.co";

/// A small model to test with.
const MODEL_REPO: &str = "microsoft/Phi-3-mini-4k-instruct-onnx";
const MODEL_REVISION: &str = "main";
const MODEL_SUBDIR: &str = "cpu_and_mobile/cpu-int4-rtn-block-32";

static CACHED_MODEL_DIR: LazyLock<PathBuf> = LazyLock::new(|| {
    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("test_models")
        .join("phi3-mini-4k-instruct-cpu-int4");

    if !target_dir.exists() {
        fs::create_dir_all(&target_dir).expect("Failed to create test models directory");
        println!("Downloading test model to {}", target_dir.display());

        let files_to_download = [
            "added_tokens.json",
            "genai_config.json",
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx",
            "phi3-mini-4k-instruct-cpu-int4-rtn-block-32.onnx.data",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ];

        for file_name in files_to_download {
            let url = format!(
                "{}/{}/resolve/{}/{}/{}",
                HUGGINGFACE_BASE_URL, MODEL_REPO, MODEL_REVISION, MODEL_SUBDIR, file_name
            );

            let dest_path = target_dir.join(file_name);
            download_file(&url, &dest_path).expect(&format!("Failed to download {}", file_name));
        }
    }

    target_dir
});

fn download_file(url: &str, dest_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let response = ureq::get(url).call()?;

    if response.status() != 200 {
        return Err(format!("Failed to download {}: HTTP {}", url, response.status()).into());
    }

    let mut dest_file = File::create(dest_path)?;
    let mut reader = response.into_body().into_reader();
    let mut buffer = [0; 8192];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        dest_file.write_all(&buffer[..bytes_read])?;
    }

    Ok(())
}

/// Gets the path to a downloaded test model.
pub fn get_test_model_dir() -> &'static Path {
    &CACHED_MODEL_DIR
}
