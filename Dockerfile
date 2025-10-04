FROM tensorflow/serving:latest

# Copy model files
COPY ./serving_model/1759538723 /models/drug_classification/1

# Create model config file
RUN echo "model_config_list: { config: { name: 'drug_classification', base_path: '/models/drug_classification', model_platform: 'tensorflow' } }" > /models/models.config

# Use PORT from Railway environment
CMD tensorflow_model_server \
    --rest_api_port=$PORT \
    --model_config_file=/models/models.config \
    --allow_version_labels_for_unavailable_models