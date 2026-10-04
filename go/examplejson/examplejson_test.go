package examplejson

import (
	"encoding/json"
	"math"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/subirroy71/tfserving_tutorial/go/tensorjson"
)

type sharedCase struct {
	Name   string          `json:"name"`
	Input  json.RawMessage `json:"input"`
	Error  bool            `json:"error"`
	Expect json.RawMessage `json:"expect"`
}

// normalize re-encodes v and decodes it with json.Number, then turns each
// number into an int64 when it is an integer (literal, or a float with no
// fraction such as 1.0) and a float64 otherwise, so that the float32 1 printed
// as "1" equals the expected 1.0 and large integers still compare exactly.
func normalize(t *testing.T, v any) any {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := tensorjson.Decode(data)
	if err != nil {
		t.Fatal(err)
	}
	var walk func(any) any
	walk = func(node any) any {
		switch n := node.(type) {
		case map[string]any:
			for k, v := range n {
				n[k] = walk(v)
			}
		case []any:
			for i, v := range n {
				n[i] = walk(v)
			}
		case json.Number:
			if i, err := strconv.ParseInt(n.String(), 10, 64); err == nil {
				return i
			}
			f, _ := strconv.ParseFloat(n.String(), 64)
			if f == math.Trunc(f) && math.Abs(f) < 1<<53 {
				return int64(f)
			}
			return f
		}
		return node
	}
	return walk(decoded)
}

func TestSharedCases(t *testing.T) {
	data, err := os.ReadFile("../../testdata/example_cases.json")
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
			got, err := ParseInput(c.Input)
			if c.Error {
				if err == nil {
					t.Fatalf("expected an error, got %v", ToJSON(got))
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			var want any
			if want, err = tensorjson.Decode(c.Expect); err != nil {
				t.Fatal(err)
			}
			if g, w := normalize(t, ToJSON(got)), normalize(t, want); !reflect.DeepEqual(g, w) {
				t.Errorf("got  %v\nwant %v", g, w)
			}
		})
	}
}

func TestLargeIntegersAreExact(t *testing.T) {
	inp, err := ParseInput([]byte(`{"examples": [{"n": [9007199254740993, -9223372036854775808]}]}`))
	if err != nil {
		t.Fatal(err)
	}
	got := inp.GetExampleList().GetExamples()[0].GetFeatures().GetFeature()["n"].GetInt64List().GetValue()
	if want := []int64{9007199254740993, -9223372036854775808}; !reflect.DeepEqual(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestErrorNamesTheExampleAndFeature(t *testing.T) {
	for doc, prefix := range map[string]string{
		`{"examples": [{"a": 1}, {"b": []}]}`:              `example 1: feature "b": `,
		`{"examples": [{"a": 1}], "context": {"c": null}}`: `context: feature "c": `,
	} {
		_, err := ParseInput([]byte(doc))
		if err == nil || !strings.HasPrefix(err.Error(), prefix) {
			t.Errorf("%s: error %v, want prefix %q", doc, err, prefix)
		}
	}
}

func TestDescribe(t *testing.T) {
	inp, err := ParseInput([]byte(`{"examples": [{"b": 1.5, "a": "x"}, {"c": {"int64_list": []}}],
		"context": {"d": 3}}`))
	if err != nil {
		t.Fatal(err)
	}
	want := "2 examples + context (a:bytes, b:float, c:int64, d:int64)"
	if got := Describe(inp); got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}
