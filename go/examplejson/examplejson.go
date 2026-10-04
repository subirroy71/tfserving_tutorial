// Package examplejson builds the tf.Example input of Classify/Regress requests
// from plain JSON.
//
// A request document lists one object per example, mapping feature names to
// values; an optional "context" object holds features shared by every example:
//
//	{"examples": [{"petal_length": 1.4, "color": "red"}, ...],
//	 "context":  {"site": "lab-3"}}
//
// Each value becomes one tf.train.Feature. The list type is inferred from the
// JSON literals, or given explicitly:
//
//	5.1, [5.1, 3.0], [1, 2.5]   float_list   (any decimal point or exponent)
//	3, [3, 4]                   int64_list   (integers only)
//	"a", ["a", {"b64": ".."}]   bytes_list
//	{"float_list": [5, 3]}      explicit; the only way to send an empty list
package examplejson

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/example"
	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow_serving/apis"
	"github.com/subirroy71/tfserving_tutorial/go/tensorjson"
)

const (
	floatList = "float_list"
	int64List = "int64_list"
	bytesList = "bytes_list"
)

func isB64(node any) bool {
	m, ok := node.(map[string]any)
	if !ok {
		return false
	}
	_, ok = m["b64"]
	return ok
}

func isInt(n json.Number) bool { return !strings.ContainsAny(n.String(), ".eE") }

func toBytes(leaf any) ([]byte, error) {
	if s, ok := leaf.(string); ok {
		return []byte(s), nil
	}
	m := leaf.(map[string]any)
	s, ok := m["b64"].(string)
	if !ok || len(m) != 1 {
		return nil, errors.New(`a bytes value must be exactly {"b64": "<base64>"}`)
	}
	raw, err := base64.StdEncoding.Strict().DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("invalid base64: %v", err)
	}
	return raw, nil
}

// leafKind classifies one element of an inferred list as int, float or bytes.
func leafKind(leaf any) (string, error) {
	switch v := leaf.(type) {
	case bool:
		return "", errors.New("booleans are not a feature type; use 0/1")
	case json.Number:
		if isInt(v) {
			return "int", nil
		}
		return "float", nil
	case string:
		return "bytes", nil
	case []any:
		return "", errors.New("nested arrays are not allowed; a feature is a flat list")
	case nil:
		return "", errors.New("null is not a feature value")
	}
	if isB64(leaf) {
		return "bytes", nil
	}
	return "", fmt.Errorf("unsupported value %v", leaf)
}

func fill(listType string, values []any) (*example.Feature, error) {
	switch listType {
	case floatList:
		out := make([]float32, 0, len(values))
		for _, v := range values {
			n, ok := v.(json.Number)
			if !ok {
				return nil, fmt.Errorf("float_list needs numbers, got %v", v)
			}
			// Parse as a double first, then round, like the Python client.
			f, err := strconv.ParseFloat(n.String(), 64)
			if err != nil && !errors.Is(err, strconv.ErrRange) {
				return nil, fmt.Errorf("float_list needs numbers, got %v", v)
			}
			out = append(out, float32(f))
		}
		return &example.Feature{Kind: &example.Feature_FloatList{FloatList: &example.FloatList{Value: out}}}, nil
	case int64List:
		out := make([]int64, 0, len(values))
		for _, v := range values {
			n, ok := v.(json.Number)
			if !ok || !isInt(n) {
				return nil, fmt.Errorf("int64_list needs integers, got %v", v)
			}
			i, err := strconv.ParseInt(n.String(), 10, 64)
			if err != nil {
				return nil, fmt.Errorf("%s is out of range for int64", n)
			}
			out = append(out, i)
		}
		return &example.Feature{Kind: &example.Feature_Int64List{Int64List: &example.Int64List{Value: out}}}, nil
	default:
		out := make([][]byte, 0, len(values))
		for _, v := range values {
			if _, ok := v.(string); !ok && !isB64(v) {
				return nil, fmt.Errorf("bytes_list needs strings, got %v", v)
			}
			b, err := toBytes(v)
			if err != nil {
				return nil, err
			}
			out = append(out, b)
		}
		return &example.Feature{Kind: &example.Feature_BytesList{BytesList: &example.BytesList{Value: out}}}, nil
	}
}

// FeatureFromJSON converts one decoded JSON value (see tensorjson.Decode): a
// scalar, a flat list, or {"<type>_list": [...]}.
func FeatureFromJSON(value any) (*example.Feature, error) {
	if obj, ok := value.(map[string]any); ok && !isB64(obj) {
		var listType string
		var values any
		for k, v := range obj {
			listType, values = k, v
		}
		if len(obj) != 1 || (listType != floatList && listType != int64List && listType != bytesList) {
			return nil, errors.New("an explicit feature is one of {float_list, int64_list, bytes_list: [...]}")
		}
		list, isList := values.([]any)
		if !isList {
			list = []any{values}
		}
		return fill(listType, list)
	}

	values, ok := value.([]any)
	if !ok {
		values = []any{value}
	}
	if len(values) == 0 {
		return nil, errors.New(`an empty list has no type; write {"float_list": []} (or int64_list, bytes_list)`)
	}
	kinds := map[string]bool{}
	for _, v := range values {
		k, err := leafKind(v)
		if err != nil {
			return nil, err
		}
		kinds[k] = true
	}
	switch {
	case !kinds["bytes"] && !kinds["float"]:
		return fill(int64List, values)
	case !kinds["bytes"]:
		return fill(floatList, values)
	case !kinds["int"] && !kinds["float"]:
		return fill(bytesList, values)
	}
	return nil, errors.New("mixes strings and numbers")
}

// ExampleFromJSON converts an object of features into a tf.Example.
func ExampleFromJSON(value any) (*example.Example, error) {
	obj, ok := value.(map[string]any)
	if !ok {
		return nil, errors.New("must be an object of features")
	}
	features := make(map[string]*example.Feature, len(obj))
	for _, name := range sortedKeys(obj) {
		f, err := FeatureFromJSON(obj[name])
		if err != nil {
			return nil, fmt.Errorf("feature %q: %w", name, err)
		}
		features[name] = f
	}
	return &example.Example{Features: &example.Features{Feature: features}}, nil
}

// FromJSON converts {"examples": [...], "context": {...}} into an Input message.
func FromJSON(doc any) (*apis.Input, error) {
	obj, _ := doc.(map[string]any)
	list, _ := obj["examples"].([]any)
	if len(list) == 0 {
		return nil, errors.New(`request must be an object with a non-empty "examples" array`)
	}
	for _, k := range sortedKeys(obj) {
		if k != "examples" && k != "context" {
			return nil, fmt.Errorf(`unknown key %q; expected "examples" and optional "context"`, k)
		}
	}
	examples := make([]*example.Example, len(list))
	for i, value := range list {
		ex, err := ExampleFromJSON(value)
		if err != nil {
			return nil, fmt.Errorf("example %d: %w", i, err)
		}
		examples[i] = ex
	}

	ctxValue, hasContext := obj["context"]
	if !hasContext {
		return &apis.Input{Kind: &apis.Input_ExampleList{ExampleList: &apis.ExampleList{Examples: examples}}}, nil
	}
	context, err := ExampleFromJSON(ctxValue)
	if err != nil {
		return nil, fmt.Errorf("context: %w", err)
	}
	return &apis.Input{Kind: &apis.Input_ExampleListWithContext{ExampleListWithContext: &apis.ExampleListWithContext{
		Examples: examples,
		Context:  context,
	}}}, nil
}

// ParseInput decodes a request document and converts it with FromJSON.
func ParseInput(data []byte) (*apis.Input, error) {
	doc, err := tensorjson.Decode(data)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON: %w", err)
	}
	return FromJSON(doc)
}

// split returns the examples and the context (nil without one).
func split(inp *apis.Input) ([]*example.Example, *example.Example) {
	if lc := inp.GetExampleListWithContext(); lc != nil {
		return lc.GetExamples(), lc.GetContext()
	}
	return inp.GetExampleList().GetExamples(), nil
}

// kindName is the feature's list type, or "" when unset.
func kindName(f *example.Feature) string {
	switch f.GetKind().(type) {
	case *example.Feature_FloatList:
		return floatList
	case *example.Feature_Int64List:
		return int64List
	case *example.Feature_BytesList:
		return bytesList
	}
	return ""
}

// ExampleToJSON renders {feature: {list_type: [values]}}, for tests and logging.
// Floats are float32 values; bytes are strings, or {"b64": ...} if not UTF-8.
func ExampleToJSON(ex *example.Example) map[string]any {
	out := map[string]any{}
	for name, f := range ex.GetFeatures().GetFeature() {
		kind := kindName(f)
		values := []any{}
		switch kind {
		case floatList:
			for _, v := range f.GetFloatList().GetValue() {
				values = append(values, tensorjson.JSONFloat(v))
			}
		case int64List:
			for _, v := range f.GetInt64List().GetValue() {
				values = append(values, v)
			}
		case bytesList:
			for _, v := range f.GetBytesList().GetValue() {
				if utf8.Valid(v) {
					values = append(values, string(v))
				} else {
					values = append(values, map[string]any{"b64": base64.StdEncoding.EncodeToString(v)})
				}
			}
		default:
			out[name] = map[string]any{}
			continue
		}
		out[name] = map[string]any{kind: values}
	}
	return out
}

// ToJSON renders an Input in the normalized form
// {"examples": [...], "context": {...}} (context only if present).
func ToJSON(inp *apis.Input) map[string]any {
	examples, context := split(inp)
	list := make([]any, len(examples))
	for i, ex := range examples {
		list[i] = ExampleToJSON(ex)
	}
	out := map[string]any{"examples": list}
	if inp.GetExampleListWithContext() != nil {
		out["context"] = ExampleToJSON(context)
	}
	return out
}

// Describe renders one line for stderr: the example count and each feature's
// list type, e.g. "3 examples (petal_length:float, sepal_width:float)".
func Describe(inp *apis.Input) string {
	examples, context := split(inp)
	all := examples
	if inp.GetExampleListWithContext() != nil {
		all = append(append([]*example.Example{}, examples...), context)
	}
	types := map[string]string{}
	for _, ex := range all {
		for name, f := range ex.GetFeatures().GetFeature() {
			if _, seen := types[name]; seen {
				continue
			}
			if kind := kindName(f); kind != "" {
				types[name] = strings.TrimSuffix(kind, "_list")
			} else {
				types[name] = "empty"
			}
		}
	}
	feats := make([]string, 0, len(types))
	for _, name := range sortedKeys(types) {
		feats = append(feats, name+":"+types[name])
	}
	ctx := ""
	if inp.GetExampleListWithContext() != nil {
		ctx = " + context"
	}
	return fmt.Sprintf("%d examples%s (%s)", len(examples), ctx, strings.Join(feats, ", "))
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
