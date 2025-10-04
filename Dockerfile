FROM tensorflow/serving:latest

# Copy model
COPY ./serving_model/1 /models/drug_classification/1

# Expose port (Railway set sendiri, tapi tetap deklarasikan)
EXPOSE 8080

# Gunakan CMD dengan exec form (bukan shell form)
ENTRYPOINT ["/usr/bin/tensorflow_model_server"]
CMD [
  "--rest_api_port=8080",
  "--model_name=drug_classification",
  "--model_base_path=/models/drug_classification"
]