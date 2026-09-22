use std::{env, path::PathBuf};

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
}
