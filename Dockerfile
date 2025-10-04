FROM tensorflow/serving:latest

# Copy model ke path yang dikenali TF Serving
COPY ./serving_model/1 /models/drug_classification/1

# Expose port (Railway akan override)
EXPOSE 8080

# Jalankan TensorFlow Serving
ENTRYPOINT ["/usr/bin/tensorflow_model_server"]
CMD ["--rest_api_port=8080", "--model_name=drug_classification", "--model_base_path=/models/drug_classification"]