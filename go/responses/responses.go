// Package responses turns TF Serving responses (other than Predict) into
// plain JSON values ready for json.Marshal.
//
// Predict outputs are tensors and go through tensorjson.ToJSON. The other
// APIs answer with their own messages; these functions give each one the
// small, stable JSON shape that the Python and Rust clients print too.
package responses

import (
	"errors"
	"fmt"

	coreprotobuf "github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow/core/protobuf"
	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow_serving/apis"
	"github.com/subirroy71/tfserving_tutorial/go/tensorjson"
)

// ModelSpecLine says which servable answered, e.g.
// "model demo version 2 signature half_plus_two".
func ModelSpecLine(spec *apis.ModelSpec) string {
	line := "model " + spec.GetName()
	if v := spec.GetVersion(); v != nil {
		line += fmt.Sprintf(" version %d", v.GetValue())
	}
	if sig := spec.GetSignatureName(); sig != "" {
		line += " signature " + sig
	}
	return line
}

// Classification renders {"classifications": [[{"label", "score"}, ...] per example]}.
func Classification(result *apis.ClassificationResult) map[string]any {
	perExample := make([]any, 0, len(result.GetClassifications()))
	for _, c := range result.GetClassifications() {
		classes := make([]any, 0, len(c.GetClasses()))
		for _, class := range c.GetClasses() {
			classes = append(classes, map[string]any{
				"label": class.GetLabel(),
				"score": tensorjson.JSONFloat(class.GetScore()),
			})
		}
		perExample = append(perExample, classes)
	}
	return map[string]any{"classifications": perExample}
}

// Regression renders {"regressions": [value per example]}.
func Regression(result *apis.RegressionResult) map[string]any {
	values := make([]any, 0, len(result.GetRegressions()))
	for _, r := range result.GetRegressions() {
		values = append(values, tensorjson.JSONFloat(r.GetValue()))
	}
	return map[string]any{"regressions": values}
}

// Status renders {"versions": [{"version", "state", "error_code", "error_message"}, ...]}
// in the order the server sent them.
func Status(resp *apis.GetModelStatusResponse) map[string]any {
	versions := make([]any, 0, len(resp.GetModelVersionStatus()))
	for _, v := range resp.GetModelVersionStatus() {
		versions = append(versions, map[string]any{
			"version":       v.GetVersion(),
			"state":         v.GetState().String(),
			"error_code":    v.GetStatus().GetErrorCode().String(),
			"error_message": v.GetStatus().GetErrorMessage(),
		})
	}
	return map[string]any{"versions": versions}
}

func tensorInfo(info *coreprotobuf.TensorInfo) map[string]any {
	// null for unknown rank; -1 for an unknown dimension (usually the batch).
	var shape any
	if s := info.GetTensorShape(); !s.GetUnknownRank() {
		dims := make([]int64, 0, len(s.GetDim()))
		for _, d := range s.GetDim() {
			dims = append(dims, d.GetSize())
		}
		shape = dims
	}
	return map[string]any{"dtype": info.GetDtype().String(), "shape": shape}
}

func tensorInfos(infos map[string]*coreprotobuf.TensorInfo) map[string]any {
	out := make(map[string]any, len(infos))
	for name, info := range infos {
		out[name] = tensorInfo(info)
	}
	return out
}

// Metadata renders {"signatures": {name: {"method", "inputs": {name: {dtype, shape}}, "outputs": ...}}}
// from the SignatureDefMap packed in the response's "signature_def" entry.
func Metadata(resp *apis.GetModelMetadataResponse) (map[string]any, error) {
	packed := resp.GetMetadata()["signature_def"]
	if packed == nil {
		return nil, errors.New(`metadata has no "signature_def"`)
	}
	var sigs apis.SignatureDefMap
	if err := packed.UnmarshalTo(&sigs); err != nil {
		return nil, fmt.Errorf("metadata has unexpected type %s", packed.GetTypeUrl())
	}
	signatures := make(map[string]any, len(sigs.GetSignatureDef()))
	for name, sig := range sigs.GetSignatureDef() {
		signatures[name] = map[string]any{
			"method":  sig.GetMethodName(),
			"inputs":  tensorInfos(sig.GetInputs()),
			"outputs": tensorInfos(sig.GetOutputs()),
		}
	}
	return map[string]any{"signatures": signatures}, nil
}
