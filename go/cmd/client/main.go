// Command client is a TensorFlow Serving gRPC client. Requests are built from
// JSON at runtime.
//
//	go run ./cmd/client predict  --model demo --input ../examples/applicants.json
//	go run ./cmd/client predict  --signature half_plus_two --label canary --input -
//	go run ./cmd/client classify --model iris --input ../examples/iris_classify.json
//	go run ./cmd/client regress  --model iris --signature regress --input ../examples/iris_regress.json
//	go run ./cmd/client metadata --model iris
//	go run ./cmd/client status   --model demo
//
// Tensors and examples sent are logged to stderr, along with the servable that
// answered; the result is printed as JSON on stdout. Exit status: 0 on success,
// 1 if the RPC failed, 2 for bad arguments or input.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/wrapperspb"

	"github.com/subirroy71/tfserving_tutorial/go/examplejson"
	"github.com/subirroy71/tfserving_tutorial/go/gen/tensorflow_serving/apis"
	"github.com/subirroy71/tfserving_tutorial/go/responses"
	"github.com/subirroy71/tfserving_tutorial/go/tensorjson"
)

// options holds the parsed flags of one command.
type options struct {
	addr, model, label, signature, input string
	version                              int64
	hasVersion, hasLabel                 bool
	timeout                              float64
}

// usageError means bad arguments or input (exit status 2).
type usageError struct{ error }

func usagef(format string, a ...any) error { return usageError{fmt.Errorf(format, a...)} }

type command struct {
	run        func(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error)
	rpc        string // RPC name for errors
	takesInput bool   // accepts --signature and --input
}

var commands = map[string]command{
	"predict":  {predict, "Predict", true},
	"classify": {classify, "Classify", true},
	"regress":  {regress, "Regress", true},
	"metadata": {metadata, "GetModelMetadata", false},
	"status":   {modelStatus, "GetModelStatus", false},
}

const usage = `usage: client <command> [flags]

commands: predict, classify, regress, metadata, status
run "client <command> -h" for the flags of a command
`

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	if len(args) == 0 {
		fmt.Fprint(os.Stderr, usage+"error: a command is required\n")
		return 2
	}
	name := args[0]
	if name == "-h" || name == "-help" || name == "--help" || name == "help" {
		fmt.Fprint(os.Stdout, usage)
		return 0
	}
	cmd, ok := commands[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "%serror: unknown command %q\n", usage, name)
		return 2
	}
	o, err := parseFlags(name, cmd.takesInput, args[1:])
	if errors.Is(err, flag.ErrHelp) {
		return 0
	}
	if err != nil {
		return 2 // the flag package has already printed the error and usage
	}

	conn, err := grpc.NewClient(o.addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(o.timeout*float64(time.Second)))
	defer cancel()

	result, err := cmd.run(ctx, conn, o)
	var usageErr usageError
	switch {
	case errors.As(err, &usageErr):
		fmt.Fprintln(os.Stderr, "error:", err)
		return 2
	case err != nil:
		if st, ok := status.FromError(err); ok {
			fmt.Fprintf(os.Stderr, "error: %s failed: %s: %s\n", cmd.rpc, codeName(st.Code()), st.Message())
		} else {
			fmt.Fprintln(os.Stderr, "error:", err)
		}
		return 1
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(result); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	return 0
}

func parseFlags(name string, takesInput bool, args []string) (*options, error) {
	o := &options{}
	fs := flag.NewFlagSet("client "+name, flag.ContinueOnError)
	fs.StringVar(&o.addr, "addr", "localhost:8500", "host:port of the gRPC endpoint")
	fs.StringVar(&o.model, "model", "demo", "model name")
	fs.Int64Var(&o.version, "version", 0, "model version (default: latest)")
	fs.StringVar(&o.label, "label", "", "version label, e.g. stable or canary")
	fs.Float64Var(&o.timeout, "timeout", 10, "RPC deadline in seconds")
	if takesInput {
		fs.StringVar(&o.signature, "signature", "serving_default", "signature name")
		fs.StringVar(&o.input, "input", "-", "JSON request file, or - for stdin")
	}
	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if fs.NArg() > 0 {
		err := fmt.Errorf("unexpected argument %q", fs.Arg(0))
		fmt.Fprintln(fs.Output(), err)
		fs.Usage()
		return nil, err
	}
	fs.Visit(func(f *flag.Flag) {
		o.hasVersion = o.hasVersion || f.Name == "version"
		o.hasLabel = o.hasLabel || f.Name == "label"
	})
	return o, nil
}

// modelSpec builds the request's ModelSpec; the signature is left out for
// metadata and status.
func modelSpec(o *options, withSignature bool) (*apis.ModelSpec, error) {
	spec := &apis.ModelSpec{Name: o.model}
	if withSignature {
		spec.SignatureName = o.signature
	}
	switch {
	case o.hasVersion && o.hasLabel:
		return nil, usagef("--version and --label are mutually exclusive")
	case o.hasVersion:
		spec.VersionChoice = &apis.ModelSpec_Version{Version: wrapperspb.Int64(o.version)}
	case o.hasLabel:
		spec.VersionChoice = &apis.ModelSpec_VersionLabel{VersionLabel: o.label}
	}
	return spec, nil
}

func readInput(path string) ([]byte, error) {
	var data []byte
	var err error
	if path == "-" {
		data, err = io.ReadAll(os.Stdin)
	} else {
		data, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, usageError{err}
	}
	return data, nil
}

func predict(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error) {
	data, err := readInput(o.input)
	if err != nil {
		return nil, err
	}
	inputs, err := tensorjson.ParseInputs(data)
	if err != nil {
		return nil, usageError{err}
	}
	spec, err := modelSpec(o, true)
	if err != nil {
		return nil, err
	}
	for _, name := range sortedKeys(inputs) {
		fmt.Fprintf(os.Stderr, "input  %s: %s\n", name, tensorjson.Describe(inputs[name]))
	}

	resp, err := apis.NewPredictionServiceClient(conn).Predict(ctx, &apis.PredictRequest{ModelSpec: spec, Inputs: inputs})
	if err != nil {
		return nil, err
	}
	fmt.Fprintln(os.Stderr, responses.ModelSpecLine(resp.GetModelSpec()))
	outputs := map[string]any{}
	for _, name := range sortedKeys(resp.GetOutputs()) {
		t := resp.GetOutputs()[name]
		fmt.Fprintf(os.Stderr, "output %s: %s\n", name, tensorjson.Describe(t))
		if outputs[name], err = tensorjson.ToJSON(t); err != nil {
			return nil, fmt.Errorf("output %q: %w", name, err)
		}
	}
	return map[string]any{"outputs": outputs}, nil
}

// examplesInput reads the tf.Example input of Classify and Regress.
func examplesInput(o *options) (*apis.Input, error) {
	data, err := readInput(o.input)
	if err != nil {
		return nil, err
	}
	inp, err := examplejson.ParseInput(data)
	if err != nil {
		return nil, usageError{err}
	}
	fmt.Fprintf(os.Stderr, "input  %s\n", examplejson.Describe(inp))
	return inp, nil
}

func classify(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error) {
	spec, err := modelSpec(o, true)
	if err != nil {
		return nil, err
	}
	inp, err := examplesInput(o)
	if err != nil {
		return nil, err
	}
	resp, err := apis.NewPredictionServiceClient(conn).Classify(ctx, &apis.ClassificationRequest{ModelSpec: spec, Input: inp})
	if err != nil {
		return nil, err
	}
	fmt.Fprintln(os.Stderr, responses.ModelSpecLine(resp.GetModelSpec()))
	return responses.Classification(resp.GetResult()), nil
}

func regress(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error) {
	spec, err := modelSpec(o, true)
	if err != nil {
		return nil, err
	}
	inp, err := examplesInput(o)
	if err != nil {
		return nil, err
	}
	resp, err := apis.NewPredictionServiceClient(conn).Regress(ctx, &apis.RegressionRequest{ModelSpec: spec, Input: inp})
	if err != nil {
		return nil, err
	}
	fmt.Fprintln(os.Stderr, responses.ModelSpecLine(resp.GetModelSpec()))
	return responses.Regression(resp.GetResult()), nil
}

func metadata(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error) {
	spec, err := modelSpec(o, false)
	if err != nil {
		return nil, err
	}
	resp, err := apis.NewPredictionServiceClient(conn).GetModelMetadata(ctx, &apis.GetModelMetadataRequest{
		ModelSpec:     spec,
		MetadataField: []string{"signature_def"}, // the only field the server supports
	})
	if err != nil {
		return nil, err
	}
	fmt.Fprintln(os.Stderr, responses.ModelSpecLine(resp.GetModelSpec()))
	return responses.Metadata(resp)
}

func modelStatus(ctx context.Context, conn *grpc.ClientConn, o *options) (any, error) {
	if o.hasLabel {
		// The server would ignore it and report every version.
		return nil, usagef("status does not support --label; use --version or neither")
	}
	spec, err := modelSpec(o, false)
	if err != nil {
		return nil, err
	}
	resp, err := apis.NewModelServiceClient(conn).GetModelStatus(ctx, &apis.GetModelStatusRequest{ModelSpec: spec})
	if err != nil {
		return nil, err
	}
	return responses.Status(resp), nil
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// codeNames are gRPC's canonical code names, indexed by code.
var codeNames = [...]string{
	"OK", "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
	"ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED", "FAILED_PRECONDITION",
	"ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL", "UNAVAILABLE", "DATA_LOSS",
	"UNAUTHENTICATED",
}

// codeName turns Go's codes.InvalidArgument into gRPC's INVALID_ARGUMENT.
func codeName(c codes.Code) string {
	if int(c) < len(codeNames) {
		return codeNames[c]
	}
	return fmt.Sprintf("CODE_%d", c)
}
