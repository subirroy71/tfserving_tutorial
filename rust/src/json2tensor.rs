//! Build `TensorProto` messages from plain JSON at runtime, and back.
//!
//! Nothing here knows the model's signature in advance: the tensor's shape is
//! discovered by walking the nested JSON arrays and its dtype is inferred
//! from the JSON literals (or taken from an explicit `{"dtype": ...}` wrapper).

use std::collections::HashMap;
use std::fmt;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use serde_json::{Map, Value};

use crate::pb::tensorflow::{tensor_shape_proto::Dim, DataType, TensorProto, TensorShapeProto};

/// The JSON value cannot be turned into a tensor (or the tensor into JSON).
#[derive(Debug)]
pub struct TensorJsonError(String);

impl fmt::Display for TensorJsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TensorJsonError {}

type Result<T> = std::result::Result<T, TensorJsonError>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(TensorJsonError(message.into()))
}

/// Resolves names such as "float32", "DT_FLOAT" or "Int64".
pub fn parse_dtype(name: &str) -> Result<DataType> {
    let lower = name.trim().to_lowercase();
    Ok(match lower.strip_prefix("dt_").unwrap_or(&lower) {
        "float" | "float32" => DataType::DtFloat,
        "double" | "float64" => DataType::DtDouble,
        "int8" => DataType::DtInt8,
        "int16" => DataType::DtInt16,
        "int32" => DataType::DtInt32,
        "int64" => DataType::DtInt64,
        "uint8" => DataType::DtUint8,
        "uint32" => DataType::DtUint32,
        "uint64" => DataType::DtUint64,
        "bool" => DataType::DtBool,
        "string" | "bytes" => DataType::DtString,
        _ => return err(format!("unsupported dtype {name:?}")),
    })
}

// ---------------------------------------------------------------------------
// JSON -> TensorProto
// ---------------------------------------------------------------------------

enum Leaf<'a> {
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(&'a str),
    B64(&'a str),
}

impl Leaf<'_> {
    fn kind(&self) -> &'static str {
        match self {
            Leaf::Bool(_) => "bool",
            Leaf::Int(_) => "int",
            Leaf::Float(_) => "float",
            Leaf::Str(_) | Leaf::B64(_) => "string",
        }
    }
}

fn is_b64(map: &Map<String, Value>) -> bool {
    map.contains_key("b64")
}

fn leaf(node: &Value) -> Result<Leaf<'_>> {
    Ok(match node {
        Value::Bool(b) => Leaf::Bool(*b),
        Value::Number(n) => {
            // serde_json keeps 1 and 1.0 apart: only the former is i64/u64.
            if let Some(i) = n.as_i64() {
                Leaf::Int(i.into())
            } else if let Some(u) = n.as_u64() {
                Leaf::Int(u.into())
            } else {
                Leaf::Float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => Leaf::Str(s),
        Value::Object(map) if is_b64(map) => match (map.len(), &map["b64"]) {
            (1, Value::String(s)) => Leaf::B64(s),
            _ => return err(r#"a bytes value must be exactly {"b64": "<base64>"}"#),
        },
        Value::Object(_) => return err("objects are not valid tensor elements"),
        Value::Null => return err("null is not a valid tensor element"),
        Value::Array(_) => unreachable!("arrays are handled by the flattener"),
    })
}

/// Records the shape and collects leaves in row-major order.
///
/// `shape` grows by one entry the first time each depth is entered; every
/// later array at that depth must have the same length. `rank` is fixed by
/// the first leaf, after which arrays may not appear at leaf depth.
#[derive(Default)]
struct Flattener<'a> {
    shape: Vec<i64>,
    leaves: Vec<Leaf<'a>>,
    rank: Option<usize>,
}

impl<'a> Flattener<'a> {
    fn walk(&mut self, node: &'a Value, depth: usize) -> Result<()> {
        if let Value::Array(items) = node {
            if self.rank.is_some_and(|rank| depth >= rank) {
                return err("ragged tensor: array found where a value was expected");
            }
            let len = items.len() as i64;
            if depth == self.shape.len() {
                self.shape.push(len);
            } else if self.shape[depth] != len {
                return err(format!(
                    "ragged tensor: dimension {depth} has length {len}, expected {}",
                    self.shape[depth]
                ));
            }
            return items
                .iter()
                .try_for_each(|child| self.walk(child, depth + 1));
        }
        if depth != self.shape.len() {
            return err("ragged tensor: value found where an array was expected");
        }
        self.rank = Some(depth);
        self.leaves.push(leaf(node)?);
        Ok(())
    }
}

fn infer_dtype(leaves: &[Leaf]) -> Result<DataType> {
    if leaves.is_empty() {
        return err(
            r#"cannot infer the dtype of an empty tensor; use {"dtype": ..., "values": ...}"#,
        );
    }
    let mut kinds: Vec<&str> = leaves.iter().map(Leaf::kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    Ok(match kinds.as_slice() {
        ["bool"] => DataType::DtBool,
        ["string"] => DataType::DtString,
        ["int"] => DataType::DtInt64,
        ["float"] | ["float", "int"] => DataType::DtFloat,
        _ => return err(format!("mixed element types: {}", kinds.join(", "))),
    })
}

fn integers(leaves: &[Leaf], name: &str, min: i128, max: i128) -> Result<Vec<i128>> {
    leaves
        .iter()
        .map(|leaf| match leaf {
            Leaf::Int(v) if (min..=max).contains(v) => Ok(*v),
            Leaf::Int(v) => err(format!("{v} is out of range for {name}")),
            _ => err(format!("{name} needs integers")),
        })
        .collect()
}

fn fill(tensor: &mut TensorProto, dtype: DataType, leaves: &[Leaf]) -> Result<()> {
    let name = dtype.as_str_name();
    let ints = |min: i128, max: i128| integers(leaves, name, min, max);
    match dtype {
        DataType::DtFloat | DataType::DtDouble => {
            for leaf in leaves {
                let v = match leaf {
                    Leaf::Int(i) => *i as f64,
                    Leaf::Float(f) => *f,
                    _ => return err(format!("{name} needs numbers")),
                };
                if dtype == DataType::DtFloat {
                    tensor.float_val.push(v as f32);
                } else {
                    tensor.double_val.push(v);
                }
            }
        }
        DataType::DtInt8 => tensor.int_val = to(ints(i8::MIN.into(), i8::MAX.into())?),
        DataType::DtInt16 => tensor.int_val = to(ints(i16::MIN.into(), i16::MAX.into())?),
        DataType::DtInt32 => tensor.int_val = to(ints(i32::MIN.into(), i32::MAX.into())?),
        DataType::DtUint8 => tensor.int_val = to(ints(0, u8::MAX.into())?),
        DataType::DtInt64 => tensor.int64_val = to(ints(i64::MIN.into(), i64::MAX.into())?),
        DataType::DtUint32 => tensor.uint32_val = to(ints(0, u32::MAX.into())?),
        DataType::DtUint64 => tensor.uint64_val = to(ints(0, u64::MAX.into())?),
        DataType::DtBool => {
            for leaf in leaves {
                match leaf {
                    Leaf::Bool(b) => tensor.bool_val.push(*b),
                    _ => return err("DT_BOOL needs true/false"),
                }
            }
        }
        DataType::DtString => {
            for leaf in leaves {
                tensor.string_val.push(match leaf {
                    Leaf::Str(s) => s.as_bytes().to_vec(),
                    Leaf::B64(s) => BASE64
                        .decode(s)
                        .map_err(|e| TensorJsonError(format!("invalid base64: {e}")))?,
                    _ => return err(r#"DT_STRING needs strings or {"b64": ...}"#),
                });
            }
        }
        _ => return err(format!("unsupported dtype {name}")),
    }
    Ok(())
}

/// Narrows range-checked integers to the width of the TensorProto field.
fn to<T: TryFrom<i128>>(values: Vec<i128>) -> Vec<T> {
    values
        .into_iter()
        .map(|v| T::try_from(v).unwrap_or_else(|_| unreachable!("range was checked")))
        .collect()
}

/// Converts one JSON value into a `TensorProto`.
///
/// `value` is either a bare literal (scalar or nested arrays), or a spec
/// object `{"dtype": <name>, "shape": [..], "values": <literal>}` in which
/// "dtype" and "shape" are both optional.
pub fn tensor_from_json(value: &Value) -> Result<TensorProto> {
    let mut dtype = None;
    let mut explicit_shape: Option<Vec<i64>> = None;
    let mut literal = value;

    if let Value::Object(spec) = value {
        if !is_b64(spec) {
            let mut unknown: Vec<&str> = spec
                .keys()
                .map(String::as_str)
                .filter(|k| !matches!(*k, "dtype" | "shape" | "values"))
                .collect();
            if !unknown.is_empty() {
                unknown.sort_unstable();
                return err(format!(
                    "unknown key(s) in tensor spec: {}",
                    unknown.join(", ")
                ));
            }
            literal = match spec.get("values") {
                Some(values) => values,
                None => return err(r#"tensor spec needs "values""#),
            };
            if let Some(name) = spec.get("dtype") {
                match name.as_str() {
                    Some(name) => dtype = Some(parse_dtype(name)?),
                    None => return err(r#""dtype" must be a string"#),
                }
            }
            if let Some(shape) = spec.get("shape") {
                let dims = shape.as_array().and_then(|dims| {
                    dims.iter()
                        .map(|d| d.as_i64().filter(|size| *size >= 0))
                        .collect::<Option<Vec<i64>>>()
                });
                match dims {
                    Some(dims) => explicit_shape = Some(dims),
                    None => return err(r#""shape" must be an array of non-negative integers"#),
                }
            }
        }
    }

    let mut flat = Flattener::default();
    flat.walk(literal, 0)?;

    let shape = match explicit_shape {
        Some(shape) => {
            let count: i64 = shape.iter().product();
            if count != flat.leaves.len() as i64 {
                return err(format!(
                    "shape {shape:?} needs {count} values, got {}",
                    flat.leaves.len()
                ));
            }
            shape
        }
        None => flat.shape,
    };
    let dtype = match dtype {
        Some(dtype) => dtype,
        None => infer_dtype(&flat.leaves)?,
    };

    let mut tensor = TensorProto {
        dtype: dtype as i32,
        tensor_shape: Some(TensorShapeProto {
            dim: shape
                .into_iter()
                .map(|size| Dim {
                    size,
                    name: String::new(),
                })
                .collect(),
            unknown_rank: false,
        }),
        ..Default::default()
    };
    fill(&mut tensor, dtype, &flat.leaves)?;
    Ok(tensor)
}

/// Converts a request document `{"inputs": {name: value, ...}}` to tensors.
pub fn inputs_from_json(doc: &Value) -> Result<HashMap<String, TensorProto>> {
    let Some(inputs) = doc.get("inputs").and_then(Value::as_object) else {
        return err(r#"request must be an object with an "inputs" object"#);
    };
    inputs
        .iter()
        .map(|(name, value)| match tensor_from_json(value) {
            Ok(tensor) => Ok((name.clone(), tensor)),
            Err(e) => err(format!("input \"{name}\": {e}")),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// TensorProto -> JSON
// ---------------------------------------------------------------------------

/// Returns the dimensions of a tensor.
pub fn tensor_shape(tensor: &TensorProto) -> Result<Vec<i64>> {
    match &tensor.tensor_shape {
        Some(shape) if shape.unknown_rank => err("tensor has unknown rank"),
        Some(shape) => Ok(shape.dim.iter().map(|d| d.size).collect()),
        None => Ok(Vec::new()),
    }
}

fn dtype_of(tensor: &TensorProto) -> Result<DataType> {
    DataType::try_from(tensor.dtype)
        .map_err(|_| TensorJsonError(format!("unknown dtype {}", tensor.dtype)))
}

/// Renders "DT_FLOAT [2, 3]"-style summaries for logs.
pub fn describe(tensor: &TensorProto) -> String {
    let dtype = dtype_of(tensor).map_or("DT_?", |d| d.as_str_name());
    match tensor_shape(tensor) {
        Ok(shape) => format!("{dtype} {shape:?}"),
        Err(_) => format!("{dtype} <unknown rank>"),
    }
}

/// Returns exactly `count` elements, taken from `tensor_content` when the
/// server packed the tensor into raw little-endian bytes and from the typed
/// field otherwise. TF may send fewer typed values than elements: the last
/// one is repeated (and none at all means all-default).
fn fixed<T: Copy + Default, const N: usize>(
    tensor: &TensorProto,
    count: usize,
    field: Vec<T>,
    from_le_bytes: fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    let content = &tensor.tensor_content;
    if !content.is_empty() {
        if content.len() != count * N {
            return err(format!(
                "tensor_content has {} bytes, expected {}",
                content.len(),
                count * N
            ));
        }
        return Ok(content
            .chunks_exact(N)
            .map(|chunk| from_le_bytes(chunk.try_into().expect("chunk has N bytes")))
            .collect());
    }
    pad(field, count)
}

fn pad<T: Clone + Default>(mut values: Vec<T>, count: usize) -> Result<Vec<T>> {
    if values.len() > count {
        return err(format!(
            "tensor has {} values for {count} elements",
            values.len()
        ));
    }
    let last = values.last().cloned().unwrap_or_default();
    values.resize(count, last);
    Ok(values)
}

fn json_float(v: f64, shortest: String) -> Value {
    if v.is_nan() {
        "NaN".into()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        // Going through the shortest decimal form keeps a float32 printing as
        // 0.1 rather than 0.10000000149011612.
        shortest.parse::<f64>().map_or(Value::Null, Value::from)
    }
}

fn json_bytes(bytes: Vec<u8>) -> Value {
    match String::from_utf8(bytes) {
        Ok(text) => Value::String(text),
        Err(e) => serde_json::json!({ "b64": BASE64.encode(e.as_bytes()) }),
    }
}

fn flat_values(tensor: &TensorProto, n: usize) -> Result<Vec<Value>> {
    fn all<T>(values: Vec<T>, f: impl Fn(T) -> Value) -> Vec<Value> {
        values.into_iter().map(f).collect()
    }
    let ints = |narrow: fn(i32) -> i32| {
        tensor
            .int_val
            .iter()
            .map(|v| narrow(*v))
            .collect::<Vec<i32>>()
    };
    let t = tensor;
    Ok(match dtype_of(t)? {
        DataType::DtFloat => all(fixed(t, n, t.float_val.clone(), f32::from_le_bytes)?, |v| {
            json_float(v.into(), v.to_string())
        }),
        DataType::DtDouble => all(
            fixed(t, n, t.double_val.clone(), f64::from_le_bytes)?,
            |v| json_float(v, v.to_string()),
        ),
        DataType::DtInt8 => all(
            fixed(t, n, ints(|v| v as i8 as i32), |b: [u8; 1]| {
                i8::from_le_bytes(b).into()
            })?,
            Value::from,
        ),
        DataType::DtInt16 => all(
            fixed(t, n, ints(|v| v as i16 as i32), |b: [u8; 2]| {
                i16::from_le_bytes(b).into()
            })?,
            Value::from,
        ),
        DataType::DtUint8 => all(
            fixed(t, n, ints(|v| v as u8 as i32), |b: [u8; 1]| b[0].into())?,
            Value::from,
        ),
        DataType::DtInt32 => all(
            fixed(t, n, t.int_val.clone(), i32::from_le_bytes)?,
            Value::from,
        ),
        DataType::DtInt64 => all(
            fixed(t, n, t.int64_val.clone(), i64::from_le_bytes)?,
            Value::from,
        ),
        DataType::DtUint32 => all(
            fixed(t, n, t.uint32_val.clone(), u32::from_le_bytes)?,
            Value::from,
        ),
        DataType::DtUint64 => all(
            fixed(t, n, t.uint64_val.clone(), u64::from_le_bytes)?,
            Value::from,
        ),
        DataType::DtBool => all(
            fixed(t, n, t.bool_val.clone(), |b: [u8; 1]| b[0] != 0)?,
            Value::Bool,
        ),
        DataType::DtString => all(pad(t.string_val.clone(), n)?, json_bytes),
        other => return err(format!("unsupported dtype {}", other.as_str_name())),
    })
}

/// Converts a `TensorProto` into nested JSON arrays (a bare value for rank 0).
pub fn tensor_to_json(tensor: &TensorProto) -> Result<Value> {
    let shape = tensor_shape(tensor)?;
    if shape.iter().any(|d| *d < 0) {
        return err(format!("tensor has an unknown dimension: {shape:?}"));
    }
    let count = shape.iter().product::<i64>() as usize;
    let mut flat = flat_values(tensor, count)?.into_iter();

    fn nest(dims: &[i64], flat: &mut std::vec::IntoIter<Value>) -> Value {
        match dims.split_first() {
            None => flat.next().expect("element count matches the shape"),
            Some((size, rest)) => Value::Array((0..*size).map(|_| nest(rest, flat)).collect()),
        }
    }
    Ok(nest(&shape, &mut flat))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shaped(dtype: DataType, dims: &[i64]) -> TensorProto {
        TensorProto {
            dtype: dtype as i32,
            tensor_shape: Some(TensorShapeProto {
                dim: dims
                    .iter()
                    .map(|&size| Dim {
                        size,
                        name: String::new(),
                    })
                    .collect(),
                unknown_rank: false,
            }),
            ..Default::default()
        }
    }

    /// Flat values of the typed field, as JSON, for comparison with cases.json.
    fn field_values(t: &TensorProto) -> Vec<Value> {
        let mut out: Vec<Value> = Vec::new();
        out.extend(t.float_val.iter().map(|v| json!(*v as f64)));
        out.extend(t.double_val.iter().map(|v| json!(v)));
        out.extend(t.int_val.iter().map(|v| json!(v)));
        out.extend(t.int64_val.iter().map(|v| json!(v)));
        out.extend(t.uint32_val.iter().map(|v| json!(v)));
        out.extend(t.uint64_val.iter().map(|v| json!(v)));
        out.extend(t.bool_val.iter().map(|v| json!(v)));
        out
    }

    #[test]
    fn shared_cases() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../testdata/cases.json");
        let file: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let cases = file["cases"].as_array().unwrap();
        assert!(!cases.is_empty());
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let result = tensor_from_json(&case["input"]);
            if case["error"] == json!(true) {
                assert!(result.is_err(), "{name}: expected an error");
                continue;
            }
            let got = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            let want = &case["expect"];
            assert_eq!(
                dtype_of(&got).unwrap().as_str_name(),
                want["dtype"],
                "{name}: dtype"
            );
            assert_eq!(
                json!(tensor_shape(&got).unwrap()),
                want["shape"],
                "{name}: shape"
            );
            if let Some(b64) = want.get("b64_values") {
                let bytes: Vec<Value> = got
                    .string_val
                    .iter()
                    .map(|b| json!(BASE64.encode(b)))
                    .collect();
                assert_eq!(&json!(bytes), b64, "{name}: bytes");
            } else if want["dtype"] == "DT_STRING" {
                let text: Vec<Value> = got
                    .string_val
                    .iter()
                    .map(|b| json!(String::from_utf8(b.clone()).unwrap()))
                    .collect();
                assert_eq!(json!(text), want["values"], "{name}: strings");
            } else {
                let values = field_values(&got);
                let expected = want["values"].as_array().unwrap();
                assert_eq!(values.len(), expected.len(), "{name}: value count");
                for (g, w) in values.iter().zip(expected) {
                    if want["dtype"] == "DT_FLOAT" || want["dtype"] == "DT_DOUBLE" {
                        let (g, w) = (g.as_f64().unwrap(), w.as_f64().unwrap());
                        assert!((g - w).abs() <= 1e-6, "{name}: {g} != {w}");
                    } else {
                        assert_eq!(g, w, "{name}: value");
                    }
                }
            }
        }
    }

    #[test]
    fn decodes_tensor_content() {
        let mut t = shaped(DataType::DtFloat, &[2, 2]);
        t.tensor_content = [1f32, 2.0, 3.0, 4.5]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(tensor_to_json(&t).unwrap(), json!([[1.0, 2.0], [3.0, 4.5]]));

        let mut t = shaped(DataType::DtInt64, &[2]);
        t.tensor_content = [-1i64, 1 << 40]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        assert_eq!(tensor_to_json(&t).unwrap(), json!([-1, 1099511627776i64]));

        let mut bad = shaped(DataType::DtFloat, &[2]);
        bad.tensor_content = vec![0; 7];
        assert!(tensor_to_json(&bad).is_err());
    }

    #[test]
    fn decodes_typed_fields() {
        let mut scalar = shaped(DataType::DtDouble, &[]);
        scalar.double_val = vec![0.1];
        assert_eq!(tensor_to_json(&scalar).unwrap(), json!(0.1));

        let mut bools = shaped(DataType::DtBool, &[2]);
        bools.bool_val = vec![true, false];
        assert_eq!(tensor_to_json(&bools).unwrap(), json!([true, false]));

        let mut repeated = shaped(DataType::DtInt32, &[3]);
        repeated.int_val = vec![7];
        assert_eq!(tensor_to_json(&repeated).unwrap(), json!([7, 7, 7]));

        assert_eq!(
            tensor_to_json(&shaped(DataType::DtFloat, &[2])).unwrap(),
            json!([0.0, 0.0])
        );
        assert_eq!(
            tensor_to_json(&shaped(DataType::DtFloat, &[2, 0])).unwrap(),
            json!([[], []])
        );

        let mut short = shaped(DataType::DtFloat, &[1]);
        short.float_val = vec![0.1];
        assert_eq!(tensor_to_json(&short).unwrap().to_string(), "[0.1]");
    }

    #[test]
    fn decodes_strings_and_non_finite() {
        let mut strings = shaped(DataType::DtString, &[2]);
        strings.string_val = vec![b"ok".to_vec(), vec![0xff, 0x00]];
        assert_eq!(
            tensor_to_json(&strings).unwrap(),
            json!(["ok", {"b64": "/wA="}])
        );

        let mut t = shaped(DataType::DtFloat, &[3]);
        t.float_val = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(
            tensor_to_json(&t).unwrap(),
            json!(["NaN", "Infinity", "-Infinity"])
        );
    }

    #[test]
    fn round_trips() {
        for value in [
            json!([[1, 2], [3, 4]]),
            json!(["a", "b"]),
            json!([[true], [false]]),
            json!(2.5),
            json!([[[0.5, 1.5]]]),
        ] {
            assert_eq!(
                tensor_to_json(&tensor_from_json(&value).unwrap()).unwrap(),
                value
            );
        }
    }

    #[test]
    fn input_errors_name_the_input() {
        let e = inputs_from_json(&json!({"inputs": {"x": [[1], [2, 3]]}})).unwrap_err();
        assert!(e.to_string().contains("input \"x\""), "{e}");
        assert!(inputs_from_json(&json!({"x": [1]})).is_err());
    }
}
