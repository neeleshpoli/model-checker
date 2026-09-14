use std::env;
use std::io::{Cursor, Read};
use std::path::PathBuf;

use bindgen::callbacks::ParseCallbacks;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Using the current NuGet package version
    let pkg_name = "microsoft.ml.onnxruntimegenai.winml";
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap();

    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let rid = match target_arch.as_str() {
        "x86_64" => "win-x64",
        "aarch64" => "win-arm64",
        _ => panic!("Unsupported architecture: {}", target_arch),
    };

    let pkg_dir = out_dir.join(format!("{}-{}", pkg_name, pkg_version));

    if !pkg_dir.exists() {
        println!(
            "cargo:warning=Downloading {} v{} for {}...",
            pkg_name, pkg_version, rid
        );

        let nupkg_url = format!(
            "https://api.nuget.org/v3-flatcontainer/{}/{}/{}.{}.nupkg",
            pkg_name, pkg_version, pkg_name, pkg_version
        );

        let resp = ureq::get(&nupkg_url)
            .call()
            .expect("Failed to download NuGet package");
        let mut bytes = Vec::new();

        resp.into_body()
            .into_reader()
            .read_to_end(&mut bytes)
            .expect("Failed to read HTTP response");

        let mut archive =
            zip::ZipArchive::new(Cursor::new(bytes)).expect("Failed to read zip archive");

        archive
            .extract(&pkg_dir)
            .expect("Failed to extract package");
    }

    // Generate Bindings
    let header_path = pkg_dir.join("build/native/include/ort_genai_c.h");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .parse_callbacks(Box::new(DocFormatter))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_dir.join("ort_genai_bindings.rs"))
        .expect("Couldn't write bindings!");

    // Link native libraries
    let lib_dir = pkg_dir.join(format!("runtimes/{}/native", rid));

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=onnxruntime-genai");

    println!("cargo:lib_dir={}", lib_dir.display());
}

#[derive(Debug)]
struct DocFormatter;

impl ParseCallbacks for DocFormatter {
    fn process_comment(&self, comment: &str) -> Option<String> {
        let formatted = comment
            .replace("\\brief", "")
            // 1. Target the specific "in/out" tags FIRST
            .replace("\\paramin", "\n* **[in]**")
            .replace("\\paramout", "\n* **[out]**")
            // (Optional) Catch standard Doxygen spacing just in case
            .replace("\\param [in]", "\n* **[in]**")
            .replace("\\param [out]", "\n* **[out]**")
            // 2. ONLY THEN target the generic \param tag
            .replace("\\param", "\n* **Param:**")
            // 3. Format the return statement
            .replace("\\return", "\n\n**Returns:**");

        Some(formatted)
    }
}
