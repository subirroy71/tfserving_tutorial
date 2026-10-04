//! TensorFlow Serving gRPC client. Requests are built from JSON at runtime.
//!
//!     cargo run -- predict  --model demo --input ../examples/applicants.json
//!     cargo run -- predict  --signature half_plus_two --label canary --input -
//!     cargo run -- classify --model iris --input ../examples/iris_classify.json
//!     cargo run -- regress  --model iris --signature regress --input ../examples/iris_regress.json
//!     cargo run -- metadata --model iris
//!     cargo run -- status   --model demo
//!
//! Tensors and examples sent are logged to stderr, along with the servable
//! that answered; the result is printed as JSON on stdout. Exit status: 0 on
//! success, 1 if the RPC failed, 2 for bad arguments or input.

use std::collections::BTreeMap;
use std::io::Read;
use std::process::ExitCode;
use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::{json, Value};
use tonic::transport::Channel;

mod json2example;
mod json2tensor;
mod responses;

/// Generated from ../proto by build.rs. The modules are nested like the
/// protobuf packages because `tensorflow.serving` refers to `tensorflow`
/// and `tensorflow.error`.
#[allow(clippy::large_enum_variant)]
pub mod pb {
    pub mod tensorflow {
        tonic::include_proto!("tensorflow");
        pub mod error {
            tonic::include_proto!("tensorflow.error");
        }
        pub mod serving {
            tonic::include_proto!("tensorflow.serving");
        }
    }
}

use pb::tensorflow::serving::{
    model_service_client::ModelServiceClient, model_spec::VersionChoice,
    prediction_service_client::PredictionServiceClient, ClassificationRequest,
    GetModelMetadataRequest, GetModelStatusRequest, Input, ModelSpec, PredictRequest,
    RegressionRequest,
};

#[derive(Parser)]
#[command(about = "TensorFlow Serving gRPC client that builds its requests from JSON")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Call Predict
    Predict(InputArgs),
    /// Call Classify
    Classify(InputArgs),
    /// Call Regress
    Regress(InputArgs),
    /// Call GetModelMetadata
    Metadata(CommonArgs),
    /// Call GetModelStatus
    Status(CommonArgs),
}

#[derive(Args)]
struct CommonArgs {
    /// host:port of the gRPC endpoint
    #[arg(long, default_value = "localhost:8500")]
    addr: String,
    /// Model name
    #[arg(long, default_value = "demo")]
    model: String,
    /// Model version (default: latest)
    #[arg(long)]
    version: Option<i64>,
    /// Version label, e.g. stable or canary
    #[arg(long)]
    label: Option<String>,
    /// RPC deadline in seconds
    #[arg(long, default_value_t = 10.0)]
    timeout: f64,
}

#[derive(Args)]
struct InputArgs {
    #[command(flatten)]
    common: CommonArgs,
    /// Signature name
    #[arg(long, default_value = "serving_default")]
    signature: String,
    /// JSON request file, or - for stdin
    #[arg(long, default_value = "-")]
    input: String,
}

/// Why a command failed; decides the message and the exit status.
enum Failure {
    /// Bad arguments or input (exit 2).
    Usage(String),
    /// The RPC, or the connection for it, failed (exit 1).
    Rpc(String),
}

type Outcome = Result<Value, Failure>;

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

fn rpc_failed(rpc: &str) -> impl Fn(tonic::Status) -> Failure + '_ {
    move |status| {
        Failure::Rpc(format!(
            "{rpc} failed: {}: {}",
            code_name(status.code()),
            status.message()
        ))
    }
}

fn read_json(path: &str) -> Result<Value, Failure> {
    let read = || -> Result<Value, Box<dyn std::error::Error>> {
        let mut text = String::new();
        if path == "-" {
            std::io::stdin().read_to_string(&mut text)?;
        } else {
            text = std::fs::read_to_string(path)?;
        }
        Ok(serde_json::from_str(&text)?)
    };
    read().map_err(|e| Failure::Usage(e.to_string()))
}

fn model_spec(args: &CommonArgs, signature: Option<&str>) -> Result<ModelSpec, Failure> {
    let version_choice = match (args.version, &args.label) {
        (Some(_), Some(_)) => {
            return Err(Failure::Usage(
                "--version and --label are mutually exclusive".into(),
            ))
        }
        (Some(v), None) => Some(VersionChoice::Version(v)),
        (None, Some(label)) => Some(VersionChoice::VersionLabel(label.clone())),
        (None, None) => None,
    };
    Ok(ModelSpec {
        name: args.model.clone(),
        signature_name: signature.unwrap_or_default().to_string(),
        version_choice,
    })
}

/// Connects on first use, like grpc's channels, so an unreachable server
/// shows up as an UNAVAILABLE error from the RPC itself.
fn connect(args: &CommonArgs) -> Result<Channel, Failure> {
    let endpoint = Channel::from_shared(format!("http://{}", args.addr))
        .map_err(|e| Failure::Usage(format!("bad --addr {:?}: {e}", args.addr)))?;
    Ok(endpoint.connect_lazy())
}

fn request<T>(message: T, args: &CommonArgs) -> tonic::Request<T> {
    let mut request = tonic::Request::new(message);
    request.set_timeout(Duration::from_secs_f64(args.timeout));
    request
}

fn log_model_spec(spec: Option<ModelSpec>) {
    eprintln!("{}", responses::model_spec_line(&spec.unwrap_or_default()));
}

async fn predict(args: InputArgs) -> Outcome {
    let inputs = json2tensor::inputs_from_json(&read_json(&args.input)?)
        .map_err(|e| Failure::Usage(e.to_string()))?;
    let model_spec = model_spec(&args.common, Some(&args.signature))?;
    for (name, tensor) in inputs.iter().collect::<BTreeMap<_, _>>() {
        eprintln!("input  {name}: {}", json2tensor::describe(tensor));
    }
    let message = PredictRequest {
        model_spec: Some(model_spec),
        inputs,
        ..Default::default()
    };

    let mut client = PredictionServiceClient::new(connect(&args.common)?);
    let response = client
        .predict(request(message, &args.common))
        .await
        .map_err(rpc_failed("Predict"))?
        .into_inner();
    log_model_spec(response.model_spec);
    let mut outputs = BTreeMap::new();
    for (name, tensor) in response.outputs.iter().collect::<BTreeMap<_, _>>() {
        eprintln!("output {name}: {}", json2tensor::describe(tensor));
        let value = json2tensor::tensor_to_json(tensor)
            .map_err(|e| Failure::Rpc(format!("output \"{name}\": {e}")))?;
        outputs.insert(name, value);
    }
    Ok(json!({ "outputs": outputs }))
}

/// The model spec and tf.Example input shared by Classify and Regress.
fn examples_request(args: &InputArgs) -> Result<(ModelSpec, Input), Failure> {
    let model_spec = model_spec(&args.common, Some(&args.signature))?;
    let input = json2example::input_from_json(&read_json(&args.input)?)
        .map_err(|e| Failure::Usage(e.to_string()))?;
    eprintln!("input  {}", json2example::describe(&input));
    Ok((model_spec, input))
}

async fn classify(args: InputArgs) -> Outcome {
    let (model_spec, input) = examples_request(&args)?;
    let message = ClassificationRequest {
        model_spec: Some(model_spec),
        input: Some(input),
    };
    let mut client = PredictionServiceClient::new(connect(&args.common)?);
    let response = client
        .classify(request(message, &args.common))
        .await
        .map_err(rpc_failed("Classify"))?
        .into_inner();
    log_model_spec(response.model_spec);
    Ok(responses::classification_to_json(
        &response.result.unwrap_or_default(),
    ))
}

async fn regress(args: InputArgs) -> Outcome {
    let (model_spec, input) = examples_request(&args)?;
    let message = RegressionRequest {
        model_spec: Some(model_spec),
        input: Some(input),
    };
    let mut client = PredictionServiceClient::new(connect(&args.common)?);
    let response = client
        .regress(request(message, &args.common))
        .await
        .map_err(rpc_failed("Regress"))?
        .into_inner();
    log_model_spec(response.model_spec);
    Ok(responses::regression_to_json(
        &response.result.unwrap_or_default(),
    ))
}

async fn metadata(args: CommonArgs) -> Outcome {
    let message = GetModelMetadataRequest {
        model_spec: Some(model_spec(&args, None)?),
        // The only field the server supports.
        metadata_field: vec!["signature_def".into()],
    };
    let mut client = PredictionServiceClient::new(connect(&args)?);
    let response = client
        .get_model_metadata(request(message, &args))
        .await
        .map_err(rpc_failed("GetModelMetadata"))?
        .into_inner();
    log_model_spec(response.model_spec.clone());
    responses::metadata_to_json(&response).map_err(Failure::Rpc)
}

async fn status(args: CommonArgs) -> Outcome {
    if args.label.is_some() {
        // The server would ignore it and report every version.
        return Err(Failure::Usage(
            "status does not support --label; use --version or neither".into(),
        ));
    }
    let message = GetModelStatusRequest {
        model_spec: Some(model_spec(&args, None)?),
    };
    let mut client = ModelServiceClient::new(connect(&args)?);
    let response = client
        .get_model_status(request(message, &args))
        .await
        .map_err(rpc_failed("GetModelStatus"))?
        .into_inner();
    Ok(responses::status_to_json(&response))
}

#[tokio::main]
async fn main() -> ExitCode {
    let outcome = match Cli::parse().command {
        Command::Predict(args) => predict(args).await,
        Command::Classify(args) => classify(args).await,
        Command::Regress(args) => regress(args).await,
        Command::Metadata(args) => metadata(args).await,
        Command::Status(args) => status(args).await,
    };
    match outcome {
        Ok(result) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&result).expect("JSON values serialize")
            );
            ExitCode::SUCCESS
        }
        Err(Failure::Usage(message)) => {
            eprintln!("error: {message}");
            ExitCode::from(2)
        }
        Err(Failure::Rpc(message)) => {
            eprintln!("error: {message}");
            ExitCode::from(1)
        }
    }
}
