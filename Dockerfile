FROM tensorflow/serving:latest

ARG MODEL_NAME=drug_classification
ARG MODEL_BASE_PATH=/models

COPY ./output/serving_model/1759538723 ${MODEL_BASE_PATH}/${MODEL_NAME}/1

ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_BASE_PATH=${MODEL_BASE_PATH}

EXPOSE 8500
EXPOSE 8501

CMD ["tensorflow_model_server", \
     "--rest_api_port=8501", \
     "--grpc_port=8500", \
     "--model_name=drug_classification", \
     "--model_base_path=/models/drug_classification", \
     "--allow_version_labels_for_unavailable_models"]