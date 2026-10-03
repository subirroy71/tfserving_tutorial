ADDR ?= localhost:8500

.PHONY: model serve mock py-protos go-protos test test-py test-go test-rust e2e

model:            ## re-export models/demo/1 (needs: pip install -r model/requirements.txt)
	rm -rf models/demo/1 && python3 model/export_model.py models/demo/1

serve:            ## TensorFlow Serving in Docker: gRPC :8500, REST :8501
	docker compose up

mock:             ## stand-in gRPC server without Docker (needs tensorflow + tensorflow-serving-api)
	python3 tools/mock_server.py --model-dir models/demo --port 8500

py-protos:        ## generate python/gen (needs: pip install -r python/requirements.txt)
	./python/gen_protos.sh

go-protos:        ## regenerate go/gen (only after editing proto/)
	./go/gen_protos.sh

test: test-py test-go test-rust

test-py: py-protos
	python3 -m unittest discover -s python

test-go:
	cd go && go vet ./... && go test -count=1 ./...

test-rust:
	cd rust && cargo test

e2e:              ## run all three clients against $(ADDR) and compare their outputs
	./scripts/e2e.sh $(ADDR)
