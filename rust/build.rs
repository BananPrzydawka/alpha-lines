use std::{env, path::PathBuf, process::Command};

fn generate_config() {
    let path = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("../config.json");
    println!("cargo:rerun-if-changed={}", path.display());
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).expect("cannot read project config.json"),
    ).expect("invalid project config.json");
    assert_eq!(config["game"]["height"].as_u64(), Some(10), "engine requires height 10");
    assert_eq!(config["game"]["width"].as_u64(), Some(16), "engine requires width 16");
    let mut generated = String::from("// Generated from config.json; do not edit.\n");
    for (section, key, ty) in [
        ("game", "height", "usize"), ("game", "width", "usize"),
        ("mcts", "k", "usize"),
        ("mcts", "g", "usize"), ("mcts", "b", "usize"), ("mcts", "t", "usize"),
        ("mcts", "s", "u32"), ("mcts", "node_capacity_factor", "usize"),
        ("mcts", "c_puct", "f32"), ("mcts", "alpha", "f32"),
        ("mcts", "epsilon", "f32"), ("mcts", "exp3_gamma", "f32"),
    ] {
        let value = &config[section][key];
        let literal = match ty {
            "&str" => format!("{:?}", value.as_str().expect("config value must be a string")),
            "f32" | "f64" => format!("{:?}", value.as_f64().expect("config value must be numeric")),
            _ => value.as_u64().expect("config value must be a nonnegative integer").to_string(),
        };
        generated.push_str(&format!("pub const {}_{}: {ty} = {literal};\n",
            section.to_uppercase(), key.to_uppercase()));
    }
    std::fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("config.rs"), generated)
        .expect("cannot write generated Rust config");
}

fn main() {
    generate_config();
    println!("cargo:rerun-if-changed=native/inference.cpp");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=PYTHON");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if env::var_os("CARGO_FEATURE_COMPILED_MODEL").is_none() { return; }
    let python = env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let info = Command::new(python).args(["-c", "import torch; from pathlib import Path; print(Path(torch.__file__).parent); print(int(torch.compiled_with_cxx11_abi())); from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME or '')"])
        .output().expect("cannot run PYTHON to locate PyTorch");
    assert!(info.status.success(), "cannot import PyTorch: {}", String::from_utf8_lossy(&info.stderr));
    let info = String::from_utf8(info.stdout).unwrap();
    let mut lines = info.lines();
    let torch = PathBuf::from(lines.next().unwrap());
    let abi = lines.next().unwrap();
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let lib = torch.join("lib");
    let cuda = lines.next().unwrap_or("");
    let mut compiler = Command::new(env::var("CXX").unwrap_or_else(|_| "c++".into()));
    if lib.join("libc10_cuda.so").exists() {
        assert!(!cuda.is_empty(), "CUDA PyTorch requires CUDA_HOME for the stream headers");
        compiler.arg("-DALPHA_CUDA").arg(format!("-I{cuda}/include"));
        compiler.args(["-lc10_cuda", "-ltorch_cuda"]);
    }
    let status = compiler
        .args(["-std=c++17", "-O3", "-shared", "-fPIC", "native/inference.cpp", "-o"])
        .arg(out.join("libalpha_inference.so"))
        .arg(format!("-D_GLIBCXX_USE_CXX11_ABI={abi}"))
        .arg(format!("-I{}", torch.join("include").display()))
        .arg(format!("-I{}", torch.join("include/torch/csrc/api/include").display()))
        .arg(format!("-L{}", lib.display()))
        .arg(format!("-Wl,-rpath,{}", lib.display()))
        // --no-as-needed: the shim references CUDA symbols only indirectly through the
        // AOTI loader, so a default --as-needed link would drop the CUDA libs from
        // DT_NEEDED and the .so would fail at load time with an undefined
        // c10::cuda::CUDACachingAllocator symbol.
        .args(["-Wl,--no-as-needed"])
        .args(["-ltorch", "-ltorch_cpu", "-lc10", "-ltorch_cuda", "-lc10_cuda"])
        .args(["-Wl,--as-needed"])
        .status().expect("cannot run C++ compiler");
    assert!(status.success(), "failed to build AOTInductor runtime");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=alpha_inference");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
}
