//! Build the tf.Example input of Classify/Regress requests from plain JSON.
//!
//! A request document lists one object per example, mapping feature names to
//! values; an optional "context" object holds features shared by every
//! example:
//!
//! ```text
//! {"examples": [{"petal_length": 1.4, "color": "red"}, ...],
//!  "context":  {"site": "lab-3"}}
//! ```
//!
//! Each value becomes one `Feature`. The list type is inferred from the JSON
//! literals, or given explicitly:
//!
//! ```text
//! 5.1, [5.1, 3.0], [1, 2.5]   float_list   (any decimal point or exponent)
//! 3, [3, 4]                   int64_list   (integers only)
//! "a", ["a", {"b64": ".."}]   bytes_list
//! {"float_list": [5, 3]}      explicit; the only way to send an empty list
//! ```

use std::collections::BTreeMap;
use std::fmt;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{json, Map, Value};

use crate::json2tensor::{json_bytes, json_f32};
use crate::pb::tensorflow::serving::{input, ExampleList, ExampleListWithContext, Input};
use crate::pb::tensorflow::{
    feature::Kind, BytesList, Example, Feature, Features, FloatList, Int64List,
};

const LIST_TYPES: [&str; 3] = ["float_list", "int64_list", "bytes_list"];

/// The JSON value cannot be turned into a tf.Example.
#[derive(Debug)]
pub struct ExampleJsonError(String);

impl fmt::Display for ExampleJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ExampleJsonError {}

type Result<T> = std::result::Result<T, ExampleJsonError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(ExampleJsonError(message.into()))
}

/// Adds `prefix` to the message of an error from a nested value.
fn context<T>(result: Result<T>, prefix: impl fmt::Display) -> Result<T> {
    result.map_err(|e| ExampleJsonError(format!("{prefix}: {}", e.0)))
}

// ---------------------------------------------------------------------------
// JSON -> Input
// ---------------------------------------------------------------------------

fn is_b64(node: &Value) -> bool {
    matches!(node, Value::Object(map) if map.contains_key("b64"))
}

fn to_bytes(leaf: &Value) -> Result<Vec<u8>> {
    match leaf {
        Value::String(s) => Ok(s.as_bytes().to_vec()),
        Value::Object(map) => match (map.len(), &map["b64"]) {
            (1, Value::String(s)) => BASE64
                .decode(s)
                .map_err(|e| ExampleJsonError(format!("invalid base64: {e}"))),
            _ => err(r#"a bytes value must be exactly {"b64": "<base64>"}"#),
        },
        _ => unreachable!("callers pass strings and b64 objects only"),
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum LeafKind {
    Int,
    Float,
    Bytes,
}

fn leaf_kind(leaf: &Value) -> Result<LeafKind> {
    Ok(match leaf {
        Value::Bool(_) => return err("booleans are not a feature type; use 0/1"),
        // serde_json keeps 1 and 1.0 apart: only the former is i64/u64.
        Value::Number(n) if n.is_i64() || n.is_u64() => LeafKind::Int,
        Value::Number(_) => LeafKind::Float,
        Value::String(_) => LeafKind::Bytes,
        Value::Object(_) if is_b64(leaf) => LeafKind::Bytes,
        Value::Array(_) => return err("nested arrays are not allowed; a feature is a flat list"),
        Value::Null => return err("null is not a feature value"),
        Value::Object(_) => return err(format!("unsupported value {leaf}")),
    })
}

fn fill(list_type: &str, values: &[Value]) -> Result<Kind> {
    Ok(match list_type {
        "float_list" => Kind::FloatList(FloatList {
            value: values
                .iter()
                .map(|v| match v.as_f64() {
                    Some(f) => Ok(f as f32),
                    None => err(format!("float_list needs numbers, got {v}")),
                })
                .collect::<Result<_>>()?,
        }),
        "int64_list" => Kind::Int64List(Int64List {
            value: values
                .iter()
                .map(|v| match v {
                    Value::Number(n) if n.is_i64() => Ok(n.as_i64().expect("is_i64")),
                    Value::Number(n) if n.is_u64() => err(format!("{n} is out of range for int64")),
                    _ => err(format!("int64_list needs integers, got {v}")),
                })
                .collect::<Result<_>>()?,
        }),
        _ => Kind::BytesList(BytesList {
            value: values
                .iter()
                .map(|v| match v {
                    Value::String(_) => to_bytes(v),
                    Value::Object(_) if is_b64(v) => to_bytes(v),
                    _ => err(format!("bytes_list needs strings, got {v}")),
                })
                .collect::<Result<_>>()?,
        }),
    })
}

/// A scalar stands for a list of one value.
fn as_list(value: &Value) -> &[Value] {
    match value {
        Value::Array(items) => items,
        other => std::slice::from_ref(other),
    }
}

/// Converts one JSON value (scalar, flat list, or `{"<type>_list": [...]}`).
pub fn feature_from_json(value: &Value) -> Result<Feature> {
    if let Value::Object(map) = value {
        if !is_b64(value) {
            return match map.iter().next() {
                Some((list_type, values))
                    if map.len() == 1 && LIST_TYPES.contains(&list_type.as_str()) =>
                {
                    Ok(Feature {
                        kind: Some(fill(list_type, as_list(values))?),
                    })
                }
                _ => err(format!(
                    "an explicit feature is one of {{{}: [...]}}",
                    LIST_TYPES.join(", ")
                )),
            };
        }
    }

    let values = as_list(value);
    if values.is_empty() {
        return err(
            r#"an empty list has no type; write {"float_list": []} (or int64_list, bytes_list)"#,
        );
    }
    let mut kinds = values.iter().map(leaf_kind).collect::<Result<Vec<_>>>()?;
    kinds.sort_unstable();
    kinds.dedup();
    let list_type = match kinds.as_slice() {
        [LeafKind::Int] => "int64_list",
        [LeafKind::Float] | [LeafKind::Int, LeafKind::Float] => "float_list",
        [LeafKind::Bytes] => "bytes_list",
        _ => return err("mixes strings and numbers"),
    };
    Ok(Feature {
        kind: Some(fill(list_type, values)?),
    })
}

/// Converts an object of features into an `Example`.
pub fn example_from_json(obj: &Value) -> Result<Example> {
    let Some(map) = obj.as_object() else {
        return err("must be an object of features");
    };
    let feature = map
        .iter()
        .map(|(name, value)| {
            let feature = context(feature_from_json(value), format!("feature \"{name}\""))?;
            Ok((name.clone(), feature))
        })
        .collect::<Result<_>>()?;
    Ok(Example {
        features: Some(Features { feature }),
    })
}

/// Converts `{"examples": [...], "context": {...}}` into an `Input` message.
pub fn input_from_json(doc: &Value) -> Result<Input> {
    let examples = doc
        .get("examples")
        .and_then(Value::as_array)
        .filter(|examples| !examples.is_empty());
    let (Some(map), Some(examples)) = (doc.as_object(), examples) else {
        return err(r#"request must be an object with a non-empty "examples" array"#);
    };
    if let Some(key) = map
        .keys()
        .find(|k| !matches!(k.as_str(), "examples" | "context"))
    {
        return err(format!(
            r#"unknown key {key:?}; expected "examples" and optional "context""#
        ));
    }
    let examples = examples
        .iter()
        .enumerate()
        .map(|(i, obj)| context(example_from_json(obj), format!("example {i}")))
        .collect::<Result<Vec<_>>>()?;

    let kind = match map.get("context") {
        Some(obj) => input::Kind::ExampleListWithContext(ExampleListWithContext {
            examples,
            context: Some(context(example_from_json(obj), "context")?),
        }),
        None => input::Kind::ExampleList(ExampleList { examples }),
    };
    Ok(Input { kind: Some(kind) })
}

// ---------------------------------------------------------------------------
// Input -> JSON, for tests and logging
// ---------------------------------------------------------------------------

fn features(example: &Example) -> BTreeMap<&String, &Feature> {
    example
        .features
        .iter()
        .flat_map(|f| f.feature.iter())
        .collect()
}

/// The list type of a feature without its "_list" suffix, or "empty".
fn type_name(feature: &Feature) -> &'static str {
    match feature.kind {
        Some(Kind::FloatList(_)) => "float",
        Some(Kind::Int64List(_)) => "int64",
        Some(Kind::BytesList(_)) => "bytes",
        None => "empty",
    }
}

fn example_to_json(example: &Example) -> Value {
    let out: Map<String, Value> = features(example)
        .into_iter()
        .map(|(name, feature)| {
            let value = match &feature.kind {
                Some(Kind::FloatList(l)) => {
                    json!({ "float_list": l.value.iter().copied().map(json_f32).collect::<Vec<_>>() })
                }
                Some(Kind::Int64List(l)) => json!({ "int64_list": l.value }),
                Some(Kind::BytesList(l)) => {
                    json!({ "bytes_list": l.value.iter().cloned().map(json_bytes).collect::<Vec<_>>() })
                }
                None => json!({}),
            };
            (name.clone(), value)
        })
        .collect();
    Value::Object(out)
}

/// The examples and the context (if any) of an `Input`.
fn parts(inp: &Input) -> (&[Example], Option<&Example>) {
    match &inp.kind {
        Some(input::Kind::ExampleListWithContext(list)) => {
            (&list.examples, Some(list.context.as_ref().unwrap_or(EMPTY)))
        }
        Some(input::Kind::ExampleList(list)) => (&list.examples, None),
        None => (&[], None),
    }
}

const EMPTY: &Example = &Example { features: None };

/// The normalized form the shared test cases are written in.
#[cfg_attr(not(test), allow(dead_code))]
pub fn input_to_json(inp: &Input) -> Value {
    let (examples, ctx) = parts(inp);
    let mut out = json!({ "examples": examples.iter().map(example_to_json).collect::<Vec<_>>() });
    if let Some(ctx) = ctx {
        out["context"] = example_to_json(ctx);
    }
    out
}

/// One line for stderr: example count and each feature's list type.
pub fn describe(inp: &Input) -> String {
    let (examples, ctx) = parts(inp);
    let mut types = BTreeMap::new();
    for example in examples.iter().chain(ctx) {
        for (name, feature) in features(example) {
            types.entry(name).or_insert(type_name(feature));
        }
    }
    let feats: Vec<String> = types.iter().map(|(n, t)| format!("{n}:{t}")).collect();
    let ctx = if ctx.is_some() { " + context" } else { "" };
    format!("{} examples{ctx} ({})", examples.len(), feats.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_cases() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../testdata/example_cases.json"
        );
        let file: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let cases = file["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let result = input_from_json(&case["input"]);
            if case["error"] == json!(true) {
                assert!(result.is_err(), "{name}: expected an error");
                continue;
            }
            let got = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(input_to_json(&got), case["expect"], "{name}");
        }
    }

    #[test]
    fn errors_name_the_example_and_feature() {
        let e = input_from_json(&json!({"examples": [{"a": 1}, {"b": []}]})).unwrap_err();
        assert!(
            e.to_string().starts_with("example 1: feature \"b\": "),
            "{e}"
        );
        let e =
            input_from_json(&json!({"examples": [{"a": 1}], "context": {"c": null}})).unwrap_err();
        assert!(e.to_string().starts_with("context: feature \"c\": "), "{e}");
    }

    #[test]
    fn describes_the_input() {
        let inp = input_from_json(&json!({
            "examples": [{"x": 1.5, "n": 2}, {"x": 2.5}],
            "context": {"s": "lab", "e": {"bytes_list": []}}
        }))
        .unwrap();
        assert_eq!(
            describe(&inp),
            "2 examples + context (e:bytes, n:int64, s:bytes, x:float)"
        );
        let inp = input_from_json(&json!({"examples": [{}]})).unwrap();
        assert_eq!(describe(&inp), "1 examples ()");
    }
}
