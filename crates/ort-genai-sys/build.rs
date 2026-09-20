use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

use bindgen::callbacks::ParseCallbacks;

struct NugetPackage {
    name: &'static str,
    version: &'static str,
    custom_action: fn(out_dir: &Path, pkg_dir: &Path, rid: &str),
}

const PACKAGES: &[NugetPackage] = &[
    NugetPackage {
        name: "microsoft.ml.onnxruntimegenai.winml",
        version: env!("CARGO_PKG_VERSION"),
        custom_action: |out_dir, pkg_dir, rid| {
            let header_path = pkg_dir.join("build/native/include/ort_genai_c.h");

            let bindings = bindgen::Builder::default()
                .header(header_path.to_str().unwrap())
                .parse_callbacks(Box::new(DocFormatter))
                .generate()
                .expect("Unable to generate bindings");

            bindings
                .write_to_file(out_dir.join("ort_genai_bindings.rs"))
                .expect("Couldn't write bindings!");

            let lib_dir = pkg_dir.join(format!("runtimes/{}/native", rid));

            println!("cargo:rustc-link-search=native={}", lib_dir.display());
            println!("cargo:rustc-link-lib=dylib=onnxruntime-genai");
            println!("cargo:lib_dir={}", lib_dir.display());
        },
    },
    // Runtime dependency providing matching onnxruntime.dll
    NugetPackage {
        name: "microsoft.windows.ai.machinelearning",
        version: "2.3.42",
        custom_action: |_out_dir, _pkg_dir, _rid| {
            // DO NOT emit cargo:rustc-link-lib here!
            // No import library exists or is needed for this package.
        },
    },
];

fn copy_dlls_to_target(native_dir: &Path, out_dir: &Path) {
    if !native_dir.exists() {
        return;
    }

    // OUT_DIR is target/{profile}/build/{crate-name}-{hash}/out
    let target_profile_dir = match out_dir.ancestors().nth(3) {
        Some(dir) => dir,
        None => return,
    };

    let deps_dir = target_profile_dir.join("deps");

    if let Ok(entries) = fs::read_dir(native_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("dll") {
                if let Some(file_name) = path.file_name() {
                    let _ = fs::copy(&path, target_profile_dir.join(file_name));
                    let _ = fs::copy(&path, deps_dir.join(file_name));
                }
            }
        }
    }
}

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let rid = match target_arch.as_str() {
        "x86_64" => "win-x64",
        "aarch64" => "win-arm64",
        _ => panic!("Unsupported architecture: {}", target_arch),
    };

    for package in PACKAGES {
        let pkg_dir = out_dir.join(format!("{}-{}", package.name, package.version));

        if !pkg_dir.exists() {
            println!(
                "cargo:warning=Downloading {} v{} for {}...",
                package.name, package.version, rid
            );

            let nupkg_url = format!(
                "https://api.nuget.org/v3-flatcontainer/{}/{}/{}.{}.nupkg",
                package.name, package.version, package.name, package.version
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

        // Copy runtime DLLs (onnxruntime.dll, DirectML.dll, etc.) to target directories
        let native_dir = pkg_dir.join(format!("runtimes/{}/native", rid));
        copy_dlls_to_target(&native_dir, &out_dir);

        (package.custom_action)(&out_dir, &pkg_dir, rid);
    }
}

#[derive(Debug)]
struct DocFormatter;

impl ParseCallbacks for DocFormatter {
    fn process_comment(&self, comment: &str) -> Option<String> {
        let formatted = comment
            .replace("\\brief", "")
            .replace("\\paramin", "\n* **[in]**")
            .replace("\\paramout", "\n* **[out]**")
            .replace("\\param [in]", "\n* **[in]**")
            .replace("\\param [out]", "\n* **[out]**")
            .replace("\\param", "\n* **Param:**")
            .replace("\\return", "\n\n**Returns:**");

        Some(formatted)
    }
}