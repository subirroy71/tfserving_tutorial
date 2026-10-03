// Command client is a TensorFlow Serving gRPC client that builds its request
// tensors from JSON.
//
//	go run ./cmd/client --model demo --input ../examples/applicants.json
//	echo '{"inputs": {"x": [1.0, 2.0]}}' | go run ./cmd/client --signature half_plus_two
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"

	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow_serving/apis"
	"github.com/subirroy71/tfserving_tutorial/go/tensorjson"
)

func main() {
	addr := flag.String("addr", "localhost:8500", "host:port of the gRPC endpoint")
	model := flag.String("model", "demo", "model name")
	signature := flag.String("signature", "serving_default", "signature name")
	version := flag.Int64("version", -1, "model version (default: latest)")
	input := flag.String("input", "-", "JSON request file, or - for stdin")
	timeout := flag.Duration("timeout", 10*time.Second, "RPC deadline")
	flag.Parse()

	os.Exit(run(*addr, *model, *signature, *version, *input, *timeout))
}

func run(addr, model, signature string, version int64, input string, timeout time.Duration) int {
	var data []byte
	var err error
	if input == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(input)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 2
	}
	inputs, err := tensorjson.ParseInputs(data)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 2
	}

	req := &apis.PredictRequest{
		ModelSpec: &apis.ModelSpec{Name: model, SignatureName: signature},
		Inputs:    inputs,
	}
	if version >= 0 {
		req.ModelSpec.VersionChoice = &apis.ModelSpec_Version{Version: wrapperspb.Int64(version)}
	}
	for _, name := range sortedKeys(inputs) {
		fmt.Fprintf(os.Stderr, "input  %s: %s\n", name, tensorjson.Describe(inputs[name]))
	}

	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	resp, err := apis.NewPredictionServiceClient(conn).Predict(ctx, req)
	if err != nil {
		st := status.Convert(err)
		fmt.Fprintf(os.Stderr, "error: Predict failed: %s: %s\n", codeName(st.Code().String()), st.Message())
		return 1
	}

	outputs := map[string]any{}
	for _, name := range sortedKeys(resp.GetOutputs()) {
		t := resp.GetOutputs()[name]
		fmt.Fprintf(os.Stderr, "output %s: %s\n", name, tensorjson.Describe(t))
		if outputs[name], err = tensorjson.ToJSON(t); err != nil {
			fmt.Fprintf(os.Stderr, "error: output %q: %v\n", name, err)
			return 1
		}
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(map[string]any{"outputs": outputs}); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	return 0
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// codeName turns Go's "InvalidArgument" into gRPC's canonical INVALID_ARGUMENT.
func codeName(camel string) string {
	out := make([]rune, 0, len(camel)+4)
	for i, r := range camel {
		if i > 0 && r >= 'A' && r <= 'Z' {
			out = append(out, '_')
		}
		if r >= 'a' && r <= 'z' {
			r -= 'a' - 'A'
		}
		out = append(out, r)
	}
	return string(out)
}
