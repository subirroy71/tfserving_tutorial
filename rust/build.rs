// Generates the prost/tonic client code from ../proto at build time.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the system protoc when PROTOC is set, otherwise the vendored binary,
    // so `cargo build` works without installing the protobuf compiler.
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }
    println!("cargo:rerun-if-changed=../proto");
    tonic_prost_build::configure()
        .build_server(false)
        .compile_protos(
            &["../proto/tensorflow_serving/apis/prediction_service.proto"],
            &["../proto"],
        )?;
    Ok(())
}
