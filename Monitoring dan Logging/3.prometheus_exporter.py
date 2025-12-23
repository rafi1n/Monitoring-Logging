import time
import json
import requests
from prometheus_client import start_http_server, Counter, Histogram, Gauge

MODEL_URL = "http://localhost:5000/invocations"
EXPORTER_PORT = 8000
SCRAPE_INTERVAL_SEC = 2

REQUESTS_TOTAL = Counter("ml_requests_total", "Total requests to model")
REQUEST_ERRORS_TOTAL = Counter("ml_request_errors_total", "Total failed requests to model")
REQUEST_LATENCY = Histogram("ml_request_latency_seconds", "Latency seconds for model request")

MODEL_UP = Gauge("ml_model_up", "1 if model endpoint reachable, else 0")
LAST_PREDICTION = Gauge("ml_last_prediction", "Last prediction value (numeric)")
LAST_STATUS_CODE = Gauge("ml_last_status_code", "Last HTTP status code")

SAMPLE_PAYLOAD = {
    "dataframe_split": {
        "columns": [
            "PassengerId","Pclass","Age","SibSp","Parch","Fare",
            "Sex_female","Sex_male",
            "Embarked_C","Embarked_Q","Embarked_S"
        ],
        "data": [
            [1, 3, 22, 1, 0, 7.25,    0, 1, 0, 0, 1],
            [2, 1, 38, 1, 0, 71.2833, 1, 0, 1, 0, 0]
        ]
    }
}

def ping_model():
    try:
        with REQUEST_LATENCY.time():
            r = requests.post(
                MODEL_URL,
                headers={"Content-Type": "application/json"},
                data=json.dumps(SAMPLE_PAYLOAD),
                timeout=5,
            )
        LAST_STATUS_CODE.set(r.status_code)
        REQUESTS_TOTAL.inc()

        if r.status_code != 200:
            MODEL_UP.set(0)
            REQUEST_ERRORS_TOTAL.inc()
            return

        MODEL_UP.set(1)

        js = r.json()
        preds = js.get("predictions", [])
        if isinstance(preds, list) and len(preds) > 0:
            try:
                LAST_PREDICTION.set(float(preds[0]))
            except Exception:
                pass

    except Exception:
        MODEL_UP.set(0)
        REQUEST_ERRORS_TOTAL.inc()

def main():
    start_http_server(EXPORTER_PORT)
    print(f"Exporter running: http://localhost:{EXPORTER_PORT}/metrics")
    while True:
        ping_model()
        time.sleep(SCRAPE_INTERVAL_SEC)

if __name__ == "__main__":
    main()
