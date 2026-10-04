ADDR ?= localhost:8500
PYTHON ?= python3
CARGO ?= cargo

.PHONY: help models serve monitoring down mock py-protos go-protos test test-py test-go test-rust e2e

help:             ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

models:           ## re-export models/ (needs: pip install -r model/requirements.txt)
	$(PYTHON) model/export_model.py models

serve:            ## TensorFlow Serving in Docker: gRPC :8500, REST :8501
	docker compose up -d serving

monitoring:       ## + Prometheus :9090 and Grafana :3000
	docker compose --profile monitoring up -d

down:             ## stop everything started by serve / monitoring
	docker compose --profile monitoring down

mock:             ## Predict-only stand-in server without Docker (needs tensorflow + tensorflow-serving-api)
	$(PYTHON) tools/mock_server.py --model-dir models/demo --port 8500

py-protos:        ## generate python/gen (needs: pip install -r python/requirements.txt)
	PYTHON=$(PYTHON) ./python/gen_protos.sh

go-protos:        ## regenerate go/gen (only after editing proto/)
	./go/gen_protos.sh

test: test-py test-go test-rust  ## unit tests in all three languages

test-py: py-protos
	$(PYTHON) -m unittest discover -s python

test-go:
	cd go && go vet ./... && go test -count=1 ./...

test-rust:
	cd rust && $(CARGO) test

e2e:              ## all three clients against $(ADDR), outputs compared
	PYTHON=$(PYTHON) CARGO=$(CARGO) ./scripts/e2e.sh $(ADDR)
