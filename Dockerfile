FROM tensorflow/serving:latest

# Set working directory
WORKDIR /app

# Copy model
COPY ./serving_model/1759538723 /models/drug_classification/1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/v1/models/drug_classification || exit 1

# Expose port (Railway akan override ini)
EXPOSE 8080

# Use PORT from Railway environment
CMD tensorflow_model_server \
    --rest_api_port=$PORT \
    --model_name=drug_classification \
    --model_base_path=/models/drug_classification \
    --allow_version_labels_for_unavailable_models