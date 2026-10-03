// Package tensorjson builds TensorProto messages from plain JSON at runtime,
// and turns them back into JSON.
//
// Nothing here knows the model's signature in advance: the tensor's shape is
// discovered by walking the nested JSON arrays and its dtype is inferred from
// the JSON literals (or taken from an explicit {"dtype": ...} wrapper).
package tensorjson

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	fw "github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/framework"
)

// dtypes maps the accepted names (lower-cased, "dt_" prefix removed).
var dtypes = map[string]fw.DataType{
	"float": fw.DataType_DT_FLOAT, "float32": fw.DataType_DT_FLOAT,
	"double": fw.DataType_DT_DOUBLE, "float64": fw.DataType_DT_DOUBLE,
	"int8": fw.DataType_DT_INT8, "int16": fw.DataType_DT_INT16,
	"int32": fw.DataType_DT_INT32, "int64": fw.DataType_DT_INT64,
	"uint8": fw.DataType_DT_UINT8, "uint32": fw.DataType_DT_UINT32, "uint64": fw.DataType_DT_UINT64,
	"bool":   fw.DataType_DT_BOOL,
	"string": fw.DataType_DT_STRING, "bytes": fw.DataType_DT_STRING,
}

// ParseDType resolves names such as "float32", "DT_FLOAT" or "Int64".
func ParseDType(name string) (fw.DataType, error) {
	key := strings.TrimPrefix(strings.ToLower(strings.TrimSpace(name)), "dt_")
	if dt, ok := dtypes[key]; ok {
		return dt, nil
	}
	return 0, fmt.Errorf("unsupported dtype %q", name)
}

// Decode parses JSON keeping numbers as json.Number, so that 1 and 1.0 stay
// distinguishable and 64-bit integers keep full precision.
func Decode(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	if _, err := dec.Token(); err != io.EOF {
		return nil, errors.New("unexpected data after the JSON value")
	}
	return v, nil
}

type kind int

const (
	kindBool kind = iota
	kindInt
	kindFloat
	kindString
)

func (k kind) String() string { return [...]string{"bool", "int", "float", "string"}[k] }

func isB64(node any) bool {
	m, ok := node.(map[string]any)
	if !ok {
		return false
	}
	_, ok = m["b64"]
	return ok
}

func kindOf(leaf any) (kind, error) {
	switch v := leaf.(type) {
	case bool:
		return kindBool, nil
	case json.Number:
		if strings.ContainsAny(v.String(), ".eE") {
			return kindFloat, nil
		}
		return kindInt, nil
	case string:
		return kindString, nil
	case map[string]any:
		if !isB64(v) {
			return 0, errors.New("objects are not valid tensor elements")
		}
		if _, ok := v["b64"].(string); !ok || len(v) != 1 {
			return 0, errors.New(`a bytes value must be exactly {"b64": "<base64>"}`)
		}
		return kindString, nil
	case nil:
		return 0, errors.New("null is not a valid tensor element")
	}
	return 0, fmt.Errorf("unsupported JSON value %T (decode with tensorjson.Decode)", leaf)
}

// flattener records the shape and collects leaves in row-major order.
//
// shape grows by one entry the first time each depth is entered; every later
// array at that depth must have the same length. rank is fixed by the first
// leaf, after which arrays may not appear at leaf depth.
type flattener struct {
	shape  []int64
	leaves []any
	rank   int // -1 until the first leaf
}

func (f *flattener) walk(node any, depth int) error {
	if arr, ok := node.([]any); ok {
		if f.rank >= 0 && depth >= f.rank {
			return errors.New("ragged tensor: array found where a value was expected")
		}
		if depth == len(f.shape) {
			f.shape = append(f.shape, int64(len(arr)))
		} else if f.shape[depth] != int64(len(arr)) {
			return fmt.Errorf("ragged tensor: dimension %d has length %d, expected %d", depth, len(arr), f.shape[depth])
		}
		for _, child := range arr {
			if err := f.walk(child, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if depth != len(f.shape) {
		return errors.New("ragged tensor: value found where an array was expected")
	}
	f.rank = depth
	f.leaves = append(f.leaves, node)
	return nil
}

func inferDType(kinds []kind) (fw.DataType, error) {
	if len(kinds) == 0 {
		return 0, errors.New(`cannot infer the dtype of an empty tensor; use {"dtype": ..., "values": ...}`)
	}
	var seen [4]bool
	for _, k := range kinds {
		seen[k] = true
	}
	switch {
	case seen[kindBool] && !seen[kindInt] && !seen[kindFloat] && !seen[kindString]:
		return fw.DataType_DT_BOOL, nil
	case seen[kindString] && !seen[kindInt] && !seen[kindFloat] && !seen[kindBool]:
		return fw.DataType_DT_STRING, nil
	case seen[kindInt] && !seen[kindFloat] && !seen[kindBool] && !seen[kindString]:
		return fw.DataType_DT_INT64, nil
	case !seen[kindBool] && !seen[kindString]:
		return fw.DataType_DT_FLOAT, nil
	}
	var names []string
	for k, s := range seen {
		if s {
			names = append(names, kind(k).String())
		}
	}
	return 0, fmt.Errorf("mixed element types: %s", strings.Join(names, ", "))
}

// intBits gives the width of the integer dtypes; negative means signed.
var intBits = map[fw.DataType]int{
	fw.DataType_DT_INT8: -8, fw.DataType_DT_INT16: -16, fw.DataType_DT_INT32: -32, fw.DataType_DT_INT64: -64,
	fw.DataType_DT_UINT8: 8, fw.DataType_DT_UINT32: 32, fw.DataType_DT_UINT64: 64,
}

func fill(t *fw.TensorProto, leaves []any, kinds []kind) error {
	name := t.Dtype.String()
	switch dt := t.Dtype; dt {
	case fw.DataType_DT_FLOAT, fw.DataType_DT_DOUBLE:
		for i, leaf := range leaves {
			if kinds[i] != kindInt && kinds[i] != kindFloat {
				return fmt.Errorf("%s needs numbers", name)
			}
			bits := 64
			if dt == fw.DataType_DT_FLOAT {
				bits = 32
			}
			v, err := strconv.ParseFloat(leaf.(json.Number).String(), bits)
			if err != nil {
				return fmt.Errorf("%v is out of range for %s", leaf, name)
			}
			if dt == fw.DataType_DT_FLOAT {
				t.FloatVal = append(t.FloatVal, float32(v))
			} else {
				t.DoubleVal = append(t.DoubleVal, v)
			}
		}
	case fw.DataType_DT_BOOL:
		for i, leaf := range leaves {
			if kinds[i] != kindBool {
				return errors.New("DT_BOOL needs true/false")
			}
			t.BoolVal = append(t.BoolVal, leaf.(bool))
		}
	case fw.DataType_DT_STRING:
		for i, leaf := range leaves {
			if kinds[i] != kindString {
				return errors.New(`DT_STRING needs strings or {"b64": ...}`)
			}
			if s, ok := leaf.(string); ok {
				t.StringVal = append(t.StringVal, []byte(s))
				continue
			}
			raw, err := base64.StdEncoding.Strict().DecodeString(leaf.(map[string]any)["b64"].(string))
			if err != nil {
				return fmt.Errorf("invalid base64: %v", err)
			}
			t.StringVal = append(t.StringVal, raw)
		}
	default:
		bits, ok := intBits[dt]
		if !ok {
			return fmt.Errorf("unsupported dtype %s", name)
		}
		for i, leaf := range leaves {
			if kinds[i] != kindInt {
				return fmt.Errorf("%s needs integers, got %v", name, leaf)
			}
			s := leaf.(json.Number).String()
			if bits > 0 { // unsigned
				v, err := strconv.ParseUint(s, 10, bits)
				if err != nil {
					return fmt.Errorf("%s is out of range for %s", s, name)
				}
				switch dt {
				case fw.DataType_DT_UINT8:
					t.IntVal = append(t.IntVal, int32(v))
				case fw.DataType_DT_UINT32:
					t.Uint32Val = append(t.Uint32Val, uint32(v))
				default:
					t.Uint64Val = append(t.Uint64Val, v)
				}
				continue
			}
			v, err := strconv.ParseInt(s, 10, -bits)
			if err != nil {
				return fmt.Errorf("%s is out of range for %s", s, name)
			}
			if dt == fw.DataType_DT_INT64 {
				t.Int64Val = append(t.Int64Val, v)
			} else {
				t.IntVal = append(t.IntVal, int32(v))
			}
		}
	}
	return nil
}

// FromJSON converts one decoded JSON value (see Decode) into a TensorProto.
//
// value is either a bare literal (scalar or nested arrays), or a spec object
// {"dtype": <name>, "shape": [..], "values": <literal>} in which "dtype" and
// "shape" are both optional.
func FromJSON(value any) (*fw.TensorProto, error) {
	var (
		dtype    fw.DataType
		hasDType bool
		shape    []int64
		hasShape bool
	)
	specError := func(format string, a ...any) (*fw.TensorProto, error) { return nil, fmt.Errorf(format, a...) }
	if spec, ok := value.(map[string]any); ok && !isB64(spec) {
		var unknown []string
		for k := range spec {
			if k != "dtype" && k != "shape" && k != "values" {
				unknown = append(unknown, k)
			}
		}
		if len(unknown) > 0 {
			sort.Strings(unknown)
			return specError("unknown key(s) in tensor spec: %s", strings.Join(unknown, ", "))
		}
		values, ok := spec["values"]
		if !ok {
			return specError(`tensor spec needs "values"`)
		}
		if raw, ok := spec["dtype"]; ok {
			name, isString := raw.(string)
			if !isString {
				return specError(`"dtype" must be a string`)
			}
			dt, err := ParseDType(name)
			if err != nil {
				return nil, err
			}
			dtype, hasDType = dt, true
		}
		if raw, ok := spec["shape"]; ok {
			dims, isArray := raw.([]any)
			if !isArray {
				return specError(`"shape" must be an array of non-negative integers`)
			}
			shape, hasShape = make([]int64, 0, len(dims)), true
			for _, d := range dims {
				n, isNumber := d.(json.Number)
				size, err := strconv.ParseInt(n.String(), 10, 64)
				if !isNumber || err != nil || size < 0 {
					return specError(`"shape" must be an array of non-negative integers`)
				}
				shape = append(shape, size)
			}
		}
		value = values
	}

	f := flattener{rank: -1, shape: []int64{}}
	if err := f.walk(value, 0); err != nil {
		return nil, err
	}
	kinds := make([]kind, len(f.leaves))
	for i, leaf := range f.leaves {
		k, err := kindOf(leaf)
		if err != nil {
			return nil, err
		}
		kinds[i] = k
	}

	if hasShape {
		count := int64(1)
		for _, d := range shape {
			count *= d
		}
		if count != int64(len(f.leaves)) {
			return nil, fmt.Errorf("shape %v needs %d values, got %d", shape, count, len(f.leaves))
		}
	} else {
		shape = f.shape
	}
	if !hasDType {
		dt, err := inferDType(kinds)
		if err != nil {
			return nil, err
		}
		dtype = dt
	}

	t := &fw.TensorProto{Dtype: dtype, TensorShape: &fw.TensorShapeProto{}}
	for _, d := range shape {
		t.TensorShape.Dim = append(t.TensorShape.Dim, &fw.TensorShapeProto_Dim{Size: d})
	}
	if err := fill(t, f.leaves, kinds); err != nil {
		return nil, err
	}
	return t, nil
}

// ParseInputs converts a request document {"inputs": {name: value, ...}}.
func ParseInputs(data []byte) (map[string]*fw.TensorProto, error) {
	doc, err := Decode(data)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	obj, _ := doc.(map[string]any)
	inputs, ok := obj["inputs"].(map[string]any)
	if !ok {
		return nil, errors.New(`request must be an object with an "inputs" object`)
	}
	tensors := make(map[string]*fw.TensorProto, len(inputs))
	for name, value := range inputs {
		t, err := FromJSON(value)
		if err != nil {
			return nil, fmt.Errorf("input %q: %w", name, err)
		}
		tensors[name] = t
	}
	return tensors, nil
}

// Shape returns the dimensions of a tensor.
func Shape(t *fw.TensorProto) ([]int64, error) {
	if t.GetTensorShape().GetUnknownRank() {
		return nil, errors.New("tensor has unknown rank")
	}
	shape := make([]int64, 0, len(t.GetTensorShape().GetDim()))
	for _, d := range t.GetTensorShape().GetDim() {
		shape = append(shape, d.GetSize())
	}
	return shape, nil
}

// Describe renders "DT_FLOAT [2 3]"-style summaries for logs.
func Describe(t *fw.TensorProto) string {
	shape, err := Shape(t)
	if err != nil {
		return fmt.Sprintf("%s <unknown rank>", t.GetDtype())
	}
	parts := make([]string, len(shape))
	for i, d := range shape {
		parts[i] = strconv.FormatInt(d, 10)
	}
	return fmt.Sprintf("%s [%s]", t.GetDtype(), strings.Join(parts, ", "))
}

// fixed is any element type with a fixed-size little-endian encoding.
type fixed interface {
	~float32 | ~float64 | ~int8 | ~int16 | ~int32 | ~int64 | ~uint8 | ~uint32 | ~uint64 | ~bool
}

// values returns exactly count elements, taken from tensor_content when the
// server packed the tensor into raw bytes and from the typed field otherwise.
// TF may send fewer typed values than elements: the last one is repeated (and
// none at all means all-default).
func values[W fixed, F any](t *fw.TensorProto, count int, field []F, conv func(F) W) ([]W, error) {
	out := make([]W, count)
	if content := t.GetTensorContent(); len(content) > 0 {
		if size := binary.Size(out); len(content) != size {
			return nil, fmt.Errorf("tensor_content has %d bytes, expected %d", len(content), size)
		}
		if _, err := binary.Decode(content, binary.LittleEndian, out); err != nil {
			return nil, err
		}
		return out, nil
	}
	if len(field) > count {
		return nil, fmt.Errorf("tensor has %d values for %d elements", len(field), count)
	}
	for i := range out {
		switch {
		case i < len(field):
			out[i] = conv(field[i])
		case len(field) > 0:
			out[i] = conv(field[len(field)-1])
		}
	}
	return out, nil
}

func same[T any](v T) T { return v }

func boxed[W any](vals []W, err error, box func(W) any) ([]any, error) {
	if err != nil {
		return nil, err
	}
	out := make([]any, len(vals))
	for i, v := range vals {
		out[i] = box(v)
	}
	return out, nil
}

func jsonFloat[T float32 | float64](v T) any {
	switch f := float64(v); {
	case math.IsNaN(f):
		return "NaN"
	case math.IsInf(f, 1):
		return "Infinity"
	case math.IsInf(f, -1):
		return "-Infinity"
	}
	return v // json.Marshal prints float32 with the shortest float32 form
}

func jsonInt[T int8 | int16 | int32 | int64 | uint8 | uint32](v T) any { return int64(v) }

func flatValues(t *fw.TensorProto, n int) ([]any, error) {
	switch t.GetDtype() {
	case fw.DataType_DT_FLOAT:
		v, err := values(t, n, t.GetFloatVal(), same[float32])
		return boxed(v, err, jsonFloat[float32])
	case fw.DataType_DT_DOUBLE:
		v, err := values(t, n, t.GetDoubleVal(), same[float64])
		return boxed(v, err, jsonFloat[float64])
	case fw.DataType_DT_INT8:
		v, err := values(t, n, t.GetIntVal(), func(x int32) int8 { return int8(x) })
		return boxed(v, err, jsonInt[int8])
	case fw.DataType_DT_INT16:
		v, err := values(t, n, t.GetIntVal(), func(x int32) int16 { return int16(x) })
		return boxed(v, err, jsonInt[int16])
	case fw.DataType_DT_INT32:
		v, err := values(t, n, t.GetIntVal(), same[int32])
		return boxed(v, err, jsonInt[int32])
	case fw.DataType_DT_INT64:
		v, err := values(t, n, t.GetInt64Val(), same[int64])
		return boxed(v, err, jsonInt[int64])
	case fw.DataType_DT_UINT8:
		v, err := values(t, n, t.GetIntVal(), func(x int32) uint8 { return uint8(x) })
		return boxed(v, err, jsonInt[uint8])
	case fw.DataType_DT_UINT32:
		v, err := values(t, n, t.GetUint32Val(), same[uint32])
		return boxed(v, err, jsonInt[uint32])
	case fw.DataType_DT_UINT64:
		v, err := values(t, n, t.GetUint64Val(), same[uint64])
		return boxed(v, err, func(x uint64) any { return x })
	case fw.DataType_DT_BOOL:
		v, err := values(t, n, t.GetBoolVal(), same[bool])
		return boxed(v, err, func(x bool) any { return x })
	case fw.DataType_DT_STRING:
		field := t.GetStringVal()
		if len(field) > n {
			return nil, fmt.Errorf("tensor has %d values for %d elements", len(field), n)
		}
		out := make([]any, n)
		for i := range out {
			var b []byte
			if i < len(field) {
				b = field[i]
			} else if len(field) > 0 {
				b = field[len(field)-1]
			}
			if utf8.Valid(b) {
				out[i] = string(b)
			} else {
				out[i] = map[string]string{"b64": base64.StdEncoding.EncodeToString(b)}
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("unsupported dtype %s", t.GetDtype())
}

// ToJSON converts a TensorProto into nested []any slices (a bare value for
// rank 0) ready for json.Marshal.
func ToJSON(t *fw.TensorProto) (any, error) {
	shape, err := Shape(t)
	if err != nil {
		return nil, err
	}
	count := int64(1)
	for _, d := range shape {
		if d < 0 {
			return nil, fmt.Errorf("tensor has an unknown dimension: %v", shape)
		}
		count *= d
	}
	flat, err := flatValues(t, int(count))
	if err != nil {
		return nil, err
	}
	var nest func(dims []int64, flat []any) any
	nest = func(dims []int64, flat []any) any {
		if len(dims) == 0 {
			return flat[0]
		}
		out := make([]any, dims[0])
		stride := 0
		if dims[0] > 0 {
			stride = len(flat) / int(dims[0])
		}
		for i := range out {
			out[i] = nest(dims[1:], flat[i*stride:(i+1)*stride])
		}
		return out
	}
	return nest(shape, flat), nil
}
