package responses

import (
	"encoding/json"
	"testing"

	"google.golang.org/protobuf/types/known/anypb"
	"google.golang.org/protobuf/types/known/wrapperspb"

	fw "github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/framework"
	coreprotobuf "github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/protobuf"
	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow_serving/apis"
)

// assertJSON compares the marshaled value with the expected JSON text.
func assertJSON(t *testing.T, got any, want string) {
	t.Helper()
	data, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var g, w any
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatal(err)
	}
	gs, _ := json.Marshal(g)
	ws, _ := json.Marshal(w)
	if string(gs) != string(ws) {
		t.Errorf("got  %s\nwant %s", gs, ws)
	}
}

func TestModelSpecLine(t *testing.T) {
	spec := &apis.ModelSpec{
		Name:          "demo",
		SignatureName: "half_plus_two",
		VersionChoice: &apis.ModelSpec_Version{Version: wrapperspb.Int64(2)},
	}
	if got, want := ModelSpecLine(spec), "model demo version 2 signature half_plus_two"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
	if got, want := ModelSpecLine(&apis.ModelSpec{Name: "demo"}), "model demo"; got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestClassification(t *testing.T) {
	result := &apis.ClassificationResult{Classifications: []*apis.Classifications{{
		Classes: []*apis.Class{{Label: "a", Score: 0.1}, {Label: "b", Score: 0.9}},
	}}}
	// float32 scores print as the shortest round-trip decimal: 0.1, not 0.10000000149011612.
	data, _ := json.Marshal(Classification(result))
	if want := `{"classifications":[[{"label":"a","score":0.1},{"label":"b","score":0.9}]]}`; string(data) != want {
		t.Errorf("got %s, want %s", data, want)
	}
}

func TestRegression(t *testing.T) {
	data, _ := json.Marshal(Regression(&apis.RegressionResult{Regressions: []*apis.Regression{{Value: 0.1}}}))
	if want := `{"regressions":[0.1]}`; string(data) != want {
		t.Errorf("got %s, want %s", data, want)
	}
	assertJSON(t, Regression(&apis.RegressionResult{}), `{"regressions": []}`)
}

func TestStatus(t *testing.T) {
	resp := &apis.GetModelStatusResponse{ModelVersionStatus: []*apis.ModelVersionStatus{
		{Version: 3, State: apis.ModelVersionStatus_LOADING, Status: &apis.StatusProto{}},
	}}
	assertJSON(t, Status(resp),
		`{"versions": [{"version": 3, "state": "LOADING", "error_code": "OK", "error_message": ""}]}`)
}

func TestMetadata(t *testing.T) {
	sig := &coreprotobuf.SignatureDef{
		MethodName: "tensorflow/serving/predict",
		Inputs: map[string]*coreprotobuf.TensorInfo{"x": {
			Dtype:       fw.DataType_DT_FLOAT,
			TensorShape: &fw.TensorShapeProto{Dim: []*fw.TensorShapeProto_Dim{{Size: -1}, {Size: 3}}},
		}},
		Outputs: map[string]*coreprotobuf.TensorInfo{"y": {
			Dtype:       fw.DataType_DT_STRING,
			TensorShape: &fw.TensorShapeProto{UnknownRank: true},
		}},
	}
	packed, err := anypb.New(&apis.SignatureDefMap{SignatureDef: map[string]*coreprotobuf.SignatureDef{"serving_default": sig}})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Metadata(&apis.GetModelMetadataResponse{Metadata: map[string]*anypb.Any{"signature_def": packed}})
	if err != nil {
		t.Fatal(err)
	}
	assertJSON(t, got, `{"signatures": {"serving_default": {
		"method": "tensorflow/serving/predict",
		"inputs": {"x": {"dtype": "DT_FLOAT", "shape": [-1, 3]}},
		"outputs": {"y": {"dtype": "DT_STRING", "shape": null}}}}}`)

	if _, err := Metadata(&apis.GetModelMetadataResponse{}); err == nil {
		t.Error("expected an error without signature_def")
	}
}
