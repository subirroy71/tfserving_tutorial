//! TensorFlow Serving gRPC client that builds its request tensors from JSON.
//!
//!     cargo run -- --model demo --input ../examples/applicants.json
//!     echo '{"inputs": {"x": [1.0, 2.0]}}' | cargo run -- --signature half_plus_two

use std::collections::BTreeMap;
use std::io::Read;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use serde_json::{json, Value};

mod json2tensor;

/// Generated from ../proto by build.rs. The modules are nested like the
/// protobuf packages because `tensorflow.serving` refers to `tensorflow`.
pub mod pb {
    pub mod tensorflow {
        tonic::include_proto!("tensorflow");
        pub mod serving {
            tonic::include_proto!("tensorflow.serving");
        }
    }
}

use pb::tensorflow::serving::{
    model_spec::VersionChoice, prediction_service_client::PredictionServiceClient, ModelSpec,
    PredictRequest,
};

#[derive(Parser)]
#[command(about = "TensorFlow Serving gRPC client that builds its request tensors from JSON")]
struct Args {
    /// host:port of the gRPC endpoint
    #[arg(long, default_value = "localhost:8500")]
    addr: String,
    /// Model name
    #[arg(long, default_value = "demo")]
    model: String,
    /// Signature name
    #[arg(long, default_value = "serving_default")]
    signature: String,
    /// Model version (default: latest)
    #[arg(long)]
    version: Option<i64>,
    /// JSON request file, or - for stdin
    #[arg(long, default_value = "-")]
    input: String,
    /// RPC deadline in seconds
    #[arg(long, default_value_t = 10.0)]
    timeout: f64,
}

/// Turns tonic's "InvalidArgument" into gRPC's canonical INVALID_ARGUMENT.
fn code_name(code: tonic::Code) -> String {
    let mut out = String::new();
    for (i, c) in format!("{code:?}").chars().enumerate() {
        if i > 0 && c.is_ascii_uppercase() {
            out.push('_');
        }
        out.push(c.to_ascii_uppercase());
    }
    out
}

fn read_request(path: &str) -> Result<Value, Box<dyn std::error::Error>> {
    let mut text = String::new();
    if path == "-" {
        std::io::stdin().read_to_string(&mut text)?;
    } else {
        text = std::fs::read_to_string(path)?;
    }
    Ok(serde_json::from_str(&text)?)
}

#[tokio::main]
async fn main() -> ExitCode {
    let args = Args::parse();

    let inputs = match read_request(&args.input)
        .and_then(|doc| json2tensor::inputs_from_json(&doc).map_err(Into::into))
    {
        Ok(inputs) => inputs,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    for (name, tensor) in inputs.iter().collect::<BTreeMap<_, _>>() {
        eprintln!("input  {name}: {}", json2tensor::describe(tensor));
    }

    let mut request = tonic::Request::new(PredictRequest {
        model_spec: Some(ModelSpec {
            name: args.model,
            signature_name: args.signature,
            version_choice: args.version.map(VersionChoice::Version),
        }),
        inputs,
        ..Default::default()
    });
    request.set_timeout(Duration::from_secs_f64(args.timeout));

    let mut client = match PredictionServiceClient::connect(format!("http://{}", args.addr)).await {
        Ok(client) => client,
        Err(e) => {
            eprintln!("error: cannot connect to {}: {e}", args.addr);
            return ExitCode::from(1);
        }
    };
    let response = match client.predict(request).await {
        Ok(response) => response.into_inner(),
        Err(status) => {
            eprintln!(
                "error: Predict failed: {}: {}",
                code_name(status.code()),
                status.message()
            );
            return ExitCode::from(1);
        }
    };

    let mut outputs = BTreeMap::new();
    for (name, tensor) in response.outputs.iter().collect::<BTreeMap<_, _>>() {
        eprintln!("output {name}: {}", json2tensor::describe(tensor));
        match json2tensor::tensor_to_json(tensor) {
            Ok(value) => outputs.insert(name, value),
            Err(e) => {
                eprintln!("error: output \"{name}\": {e}");
                return ExitCode::from(1);
            }
        };
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({ "outputs": outputs }))
            .expect("JSON values serialize")
    );
    ExitCode::SUCCESS
}
