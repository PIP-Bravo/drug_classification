FROM tensorflow/serving:latest

# Copy model ke path yang dikenali TF Serving
COPY ./serving_model/1 /models/drug_classification/1

# Expose port
EXPOSE 8080

# Start TF Serving
CMD tensorflow_model_server \
    --rest_api_port=$PORT \
    --model_name=drug_classification \
    --model_base_path=/models/drug_classification