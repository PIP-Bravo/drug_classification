from prometheus_client import Counter, Histogram, start_http_server
import requests
import time
import base64
import tensorflow as tf

# Metrics
REQUEST_COUNT = Counter("inference_requests_total", "Total number of inference requests")
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Latency of inference requests in seconds")
REQUEST_ERRORS = Counter("inference_request_errors_total", "Total number of inference errors")

MODEL_URL = "https://drugclassification-production.up.railway.app/v1/models/drug_classification:predict"

def make_tf_example(data):
    feature = {
        "Age": tf.train.Feature(int64_list=tf.train.Int64List(value=[data["Age"]])),
        "Sex": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data["Sex"].encode()])),
        "BP": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data["BP"].encode()])),
        "Cholesterol": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data["Cholesterol"].encode()])),
        "Na_to_K": tf.train.Feature(float_list=tf.train.FloatList(value=[data["Na_to_K"]]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def call_model(sample):
    serialized = make_tf_example(sample)
    b64_example = base64.b64encode(serialized).decode("utf-8")
    payload = {"instances": [{"b64": b64_example}]}

    start = time.time()
    try:
        resp = requests.post(MODEL_URL, json=payload)
        latency = time.time() - start
        REQUEST_LATENCY.observe(latency)
        if resp.status_code == 200:
            REQUEST_COUNT.inc()
            return resp.json()
        else:
            REQUEST_ERRORS.inc()
            return {"error": resp.text}
    except Exception as e:
        REQUEST_ERRORS.inc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Start Prometheus server di port 8000
    start_http_server(8000)
    print("Prometheus exporter running at http://localhost:8000/metrics")

    # Contoh dummy request loop (untuk generate metric)
    sample = {"Age": 23, "Sex": "F", "BP": "HIGH", "Cholesterol": "HIGH", "Na_to_K": 25.355}
    while True:
        result = call_model(sample)
        print(result)
        time.sleep(5)  # request tiap 5 detik
