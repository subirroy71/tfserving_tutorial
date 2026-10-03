package tensorjson

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	fw "github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/framework"
)

type sharedCase struct {
	Name   string          `json:"name"`
	Input  json.RawMessage `json:"input"`
	Error  bool            `json:"error"`
	Expect struct {
		DType     string            `json:"dtype"`
		Shape     []int64           `json:"shape"`
		Values    []json.RawMessage `json:"values"`
		B64Values []string          `json:"b64_values"`
	} `json:"expect"`
}

// flatStrings renders the typed field of t as strings for comparison.
func flatStrings(t *fw.TensorProto) []string {
	var out []string
	for _, v := range t.FloatVal {
		out = append(out, strconv.FormatFloat(float64(v), 'g', -1, 64))
	}
	for _, v := range t.DoubleVal {
		out = append(out, strconv.FormatFloat(v, 'g', -1, 64))
	}
	for _, v := range t.IntVal {
		out = append(out, strconv.FormatInt(int64(v), 10))
	}
	for _, v := range t.Int64Val {
		out = append(out, strconv.FormatInt(v, 10))
	}
	for _, v := range t.Uint32Val {
		out = append(out, strconv.FormatUint(uint64(v), 10))
	}
	for _, v := range t.Uint64Val {
		out = append(out, strconv.FormatUint(v, 10))
	}
	for _, v := range t.BoolVal {
		out = append(out, strconv.FormatBool(v))
	}
	for _, v := range t.StringVal {
		out = append(out, string(v))
	}
	return out
}

func TestSharedCases(t *testing.T) {
	data, err := os.ReadFile("../../testdata/cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var file struct {
		Cases []sharedCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &file); err != nil {
		t.Fatal(err)
	}
	if len(file.Cases) == 0 {
		t.Fatal("no cases loaded")
	}
	for _, c := range file.Cases {
		t.Run(c.Name, func(t *testing.T) {
			value, err := Decode(c.Input)
			if err != nil {
				t.Fatal(err)
			}
			got, err := FromJSON(value)
			if c.Error {
				if err == nil {
					t.Fatalf("expected an error, got %v", got)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.Dtype.String() != c.Expect.DType {
				t.Errorf("dtype = %s, want %s", got.Dtype, c.Expect.DType)
			}
			if shape, _ := Shape(got); !reflect.DeepEqual(shape, c.Expect.Shape) {
				t.Errorf("shape = %v, want %v", shape, c.Expect.Shape)
			}
			values := flatStrings(got)
			var want []string
			for _, b := range c.Expect.B64Values {
				raw, _ := base64.StdEncoding.DecodeString(b)
				want = append(want, string(raw))
			}
			for _, raw := range c.Expect.Values {
				var s string
				if json.Unmarshal(raw, &s) != nil {
					s = string(raw)
				}
				want = append(want, s)
			}
			if len(values) != len(want) {
				t.Fatalf("values = %v, want %v", values, want)
			}
			for i := range want {
				if strings.HasPrefix(c.Expect.DType, "DT_FLOAT") || c.Expect.DType == "DT_DOUBLE" {
					g, _ := strconv.ParseFloat(values[i], 64)
					w, _ := strconv.ParseFloat(want[i], 64)
					if math.Abs(g-w) > 1e-6 {
						t.Errorf("value %d = %v, want %v", i, g, w)
					}
				} else if values[i] != want[i] {
					t.Errorf("value %d = %q, want %q", i, values[i], want[i])
				}
			}
		})
	}
}

func shaped(dt fw.DataType, dims ...int64) *fw.TensorProto {
	t := &fw.TensorProto{Dtype: dt, TensorShape: &fw.TensorShapeProto{}}
	for _, d := range dims {
		t.TensorShape.Dim = append(t.TensorShape.Dim, &fw.TensorShapeProto_Dim{Size: d})
	}
	return t
}

func le(v any) []byte {
	var b bytes.Buffer
	binary.Write(&b, binary.LittleEndian, v)
	return b.Bytes()
}

func asJSON(t *testing.T, tensor *fw.TensorProto) string {
	t.Helper()
	v, err := ToJSON(tensor)
	if err != nil {
		t.Fatal(err)
	}
	out, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestToJSON(t *testing.T) {
	content := shaped(fw.DataType_DT_FLOAT, 2, 2)
	content.TensorContent = le([]float32{1, 2, 3, 4.5})
	content64 := shaped(fw.DataType_DT_INT64, 2)
	content64.TensorContent = le([]int64{-1, 1 << 40})
	scalar := shaped(fw.DataType_DT_DOUBLE)
	scalar.DoubleVal = []float64{0.1}
	bools := shaped(fw.DataType_DT_BOOL, 2)
	bools.BoolVal = []bool{true, false}
	repeated := shaped(fw.DataType_DT_INT32, 3)
	repeated.IntVal = []int32{7}
	strs := shaped(fw.DataType_DT_STRING, 2)
	strs.StringVal = [][]byte{[]byte("ok"), {0xff, 0x00}}
	nonFinite := shaped(fw.DataType_DT_FLOAT, 3)
	nonFinite.FloatVal = []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1))}
	short := shaped(fw.DataType_DT_FLOAT, 1)
	short.FloatVal = []float32{0.1}

	for _, c := range []struct {
		name   string
		tensor *fw.TensorProto
		want   string
	}{
		{"float tensor_content", content, `[[1,2],[3,4.5]]`},
		{"int64 tensor_content", content64, `[-1,1099511627776]`},
		{"scalar", scalar, `0.1`},
		{"bools", bools, `[true,false]`},
		{"last value is repeated", repeated, `[7,7,7]`},
		{"no values means zeros", shaped(fw.DataType_DT_FLOAT, 2), `[0,0]`},
		{"strings and bytes", strs, `["ok",{"b64":"/wA="}]`},
		{"non-finite", nonFinite, `["NaN","Infinity","-Infinity"]`},
		{"empty", shaped(fw.DataType_DT_FLOAT, 2, 0), `[[],[]]`},
		{"float32 prints short", short, `[0.1]`},
	} {
		if got := asJSON(t, c.tensor); got != c.want {
			t.Errorf("%s: got %s, want %s", c.name, got, c.want)
		}
	}

	bad := shaped(fw.DataType_DT_FLOAT, 2)
	bad.TensorContent = make([]byte, 7)
	if _, err := ToJSON(bad); err == nil {
		t.Error("expected an error for a truncated tensor_content")
	}
}

func TestRoundTrip(t *testing.T) {
	for _, in := range []string{`[[1,2],[3,4]]`, `["a","b"]`, `[[true],[false]]`, `2.5`, `[[[0.5,1.5]]]`} {
		value, _ := Decode([]byte(in))
		tensor, err := FromJSON(value)
		if err != nil {
			t.Fatal(err)
		}
		if got := asJSON(t, tensor); got != in {
			t.Errorf("round trip of %s gave %s", in, got)
		}
	}
}

func TestParseInputs(t *testing.T) {
	if _, err := ParseInputs([]byte(`{"inputs": {"x": [[1], [2, 3]]}}`)); err == nil || !strings.Contains(err.Error(), `input "x"`) {
		t.Errorf("error should name the input, got %v", err)
	}
	if _, err := ParseInputs([]byte(`{"x": [1]}`)); err == nil {
		t.Error(`expected an error without "inputs"`)
	}
	got, err := ParseInputs([]byte(`{"inputs": {"a": [1], "b": "s"}}`))
	if err != nil || len(got) != 2 {
		t.Errorf("got %v, %v", got, err)
	}
}
