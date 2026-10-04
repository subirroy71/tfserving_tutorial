# A self-contained serving image: the model server plus the models and their
# config, so a deployment is one immutable tag.
#
#   docker build -f deploy/Dockerfile -t tfserving-tutorial:1 .
#   docker run --rm -p 8500:8500 -p 8501:8501 tfserving-tutorial:1
FROM tensorflow/serving:2.21.0

COPY models /models
COPY config /config

EXPOSE 8500 8501
ENTRYPOINT ["tensorflow_model_server", \
  "--port=8500", \
  "--rest_api_port=8501", \
  "--model_config_file=/config/models.config", \
  "--model_config_file_poll_wait_seconds=30", \
  "--monitoring_config_file=/config/monitoring.config", \
  "--enable_batching", \
  "--batching_parameters_file=/config/batching.config", \
  "--allow_version_labels_for_unavailable_models=true"]
