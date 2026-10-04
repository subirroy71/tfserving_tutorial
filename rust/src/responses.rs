//! Turn TF Serving responses (other than Predict) into plain JSON.
//!
//! Predict outputs are tensors and go through `json2tensor::tensor_to_json`.
//! The other APIs answer with their own messages; these functions give each
//! one a small, stable JSON shape that all three clients print identically.

use std::collections::BTreeMap;

use prost::Message;
use serde_json::{json, Value};

use crate::json2tensor::json_f32;
use crate::pb::tensorflow::error::Code;
use crate::pb::tensorflow::serving::{
    model_spec::VersionChoice, model_version_status::State, ClassificationResult,
    GetModelMetadataResponse, GetModelStatusResponse, ModelSpec, RegressionResult, SignatureDefMap,
};
use crate::pb::tensorflow::{DataType, TensorInfo};

/// Which servable answered, e.g. "model demo version 2 signature half_plus_two".
pub fn model_spec_line(spec: &ModelSpec) -> String {
    let mut line = format!("model {}", spec.name);
    if let Some(VersionChoice::Version(v)) = spec.version_choice {
        line += &format!(" version {v}");
    }
    if !spec.signature_name.is_empty() {
        line += &format!(" signature {}", spec.signature_name);
    }
    line
}

/// `{"classifications": [[{"label": ..., "score": ...}, ...] per example]}`
pub fn classification_to_json(result: &ClassificationResult) -> Value {
    let classifications: Vec<Vec<Value>> = result
        .classifications
        .iter()
        .map(|c| {
            c.classes
                .iter()
                .map(|class| json!({ "label": class.label, "score": json_f32(class.score) }))
                .collect()
        })
        .collect();
    json!({ "classifications": classifications })
}

/// `{"regressions": [value per example]}`
pub fn regression_to_json(result: &RegressionResult) -> Value {
    let regressions: Vec<Value> = result
        .regressions
        .iter()
        .map(|r| json_f32(r.value))
        .collect();
    json!({ "regressions": regressions })
}

/// The enum's protobuf name, or its number when this build doesn't know it.
fn enum_name<E: TryFrom<i32>>(value: i32, name: fn(&E) -> &'static str) -> Value {
    E::try_from(value).map_or_else(|_| value.into(), |e| name(&e).into())
}

/// `{"versions": [{"version", "state", "error_code", "error_message"}, ...]}`
pub fn status_to_json(response: &GetModelStatusResponse) -> Value {
    let versions: Vec<Value> = response
        .model_version_status
        .iter()
        .map(|v| {
            let status = v.status.clone().unwrap_or_default();
            json!({
                "version": v.version,
                "state": enum_name(v.state, State::as_str_name),
                "error_code": enum_name(status.error_code, Code::as_str_name),
                "error_message": status.error_message,
            })
        })
        .collect();
    json!({ "versions": versions })
}

fn tensor_info(info: &TensorInfo) -> Value {
    let shape = info.tensor_shape.clone().unwrap_or_default();
    json!({
        "dtype": enum_name(info.dtype, DataType::as_str_name),
        // null for unknown rank; -1 for an unknown dimension (usually the batch).
        "shape": if shape.unknown_rank {
            Value::Null
        } else {
            shape.dim.iter().map(|d| d.size).collect::<Vec<_>>().into()
        },
    })
}

fn tensor_infos(infos: &std::collections::HashMap<String, TensorInfo>) -> BTreeMap<&String, Value> {
    infos.iter().map(|(k, v)| (k, tensor_info(v))).collect()
}

/// `{"signatures": {name: {"method", "inputs": {name: {dtype, shape}}, "outputs": ...}}}`
pub fn metadata_to_json(response: &GetModelMetadataResponse) -> Result<Value, String> {
    let Some(any) = response.metadata.get("signature_def") else {
        return Err("metadata has no signature_def".into());
    };
    if any.type_url.rsplit('/').next() != Some("tensorflow.serving.SignatureDefMap") {
        return Err(format!("metadata has unexpected type {}", any.type_url));
    }
    let sigs = SignatureDefMap::decode(any.value.as_slice())
        .map_err(|e| format!("cannot decode signature_def: {e}"))?;
    let signatures: BTreeMap<&String, Value> = sigs
        .signature_def
        .iter()
        .map(|(name, sig)| {
            let value = json!({
                "method": sig.method_name,
                "inputs": tensor_infos(&sig.inputs),
                "outputs": tensor_infos(&sig.outputs),
            });
            (name, value)
        })
        .collect();
    Ok(json!({ "signatures": signatures }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::tensorflow::serving::{
        Class, Classifications, ModelVersionStatus, Regression, StatusProto,
    };
    use crate::pb::tensorflow::{tensor_shape_proto::Dim, SignatureDef, TensorShapeProto};

    #[test]
    fn model_spec_lines() {
        let spec = ModelSpec {
            name: "demo".into(),
            signature_name: "half_plus_two".into(),
            version_choice: Some(VersionChoice::Version(2)),
        };
        assert_eq!(
            model_spec_line(&spec),
            "model demo version 2 signature half_plus_two"
        );
        let spec = ModelSpec {
            name: "demo".into(),
            ..Default::default()
        };
        assert_eq!(model_spec_line(&spec), "model demo");
    }

    #[test]
    fn classification() {
        let class = |label: &str, score| Class {
            label: label.into(),
            score,
        };
        let result = ClassificationResult {
            classifications: vec![Classifications {
                classes: vec![class("a", 0.1), class("b", 0.9)],
            }],
        };
        assert_eq!(
            classification_to_json(&result),
            json!({"classifications": [[{"label": "a", "score": 0.1}, {"label": "b", "score": 0.9}]]})
        );
    }

    #[test]
    fn regression() {
        let result = RegressionResult {
            regressions: vec![Regression { value: 0.1 }],
        };
        assert_eq!(regression_to_json(&result), json!({"regressions": [0.1]}));
    }

    #[test]
    fn status() {
        let response = GetModelStatusResponse {
            model_version_status: vec![ModelVersionStatus {
                version: 3,
                state: State::Loading as i32,
                status: Some(StatusProto::default()),
            }],
        };
        assert_eq!(
            status_to_json(&response),
            json!({"versions": [{"version": 3, "state": "LOADING", "error_code": "OK", "error_message": ""}]})
        );
    }

    #[test]
    fn metadata() {
        let dim = |size| Dim {
            size,
            name: String::new(),
        };
        let mut sig = SignatureDef {
            method_name: "tensorflow/serving/predict".into(),
            ..Default::default()
        };
        sig.inputs.insert(
            "x".into(),
            TensorInfo {
                dtype: DataType::DtFloat as i32,
                tensor_shape: Some(TensorShapeProto {
                    dim: vec![dim(-1), dim(3)],
                    unknown_rank: false,
                }),
                ..Default::default()
            },
        );
        sig.outputs.insert(
            "y".into(),
            TensorInfo {
                dtype: DataType::DtString as i32,
                tensor_shape: Some(TensorShapeProto {
                    dim: vec![],
                    unknown_rank: true,
                }),
                ..Default::default()
            },
        );
        let mut sigs = SignatureDefMap::default();
        sigs.signature_def.insert("serving_default".into(), sig);
        let mut response = GetModelMetadataResponse::default();
        response.metadata.insert(
            "signature_def".into(),
            prost_types::Any {
                type_url: "type.googleapis.com/tensorflow.serving.SignatureDefMap".into(),
                value: sigs.encode_to_vec(),
            },
        );
        assert_eq!(
            metadata_to_json(&response).unwrap(),
            json!({"signatures": {"serving_default": {
                "method": "tensorflow/serving/predict",
                "inputs": {"x": {"dtype": "DT_FLOAT", "shape": [-1, 3]}},
                "outputs": {"y": {"dtype": "DT_STRING", "shape": null}},
            }}})
        );

        response.metadata.get_mut("signature_def").unwrap().type_url =
            "type.googleapis.com/tensorflow.SignatureDef".into();
        assert!(metadata_to_json(&response).is_err());
    }
}
