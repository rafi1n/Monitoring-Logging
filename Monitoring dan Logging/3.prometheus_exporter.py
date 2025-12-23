import time
import json
import threading
from typing import Any, Dict

import requests
from prometheus_client import start_http_server, Counter, Gauge, Histogram

MODEL_URL = "http://localhost:5000/invocations"
EXPORTER_PORT = 8000

REQ_TOTAL = Counter("msml_requests_total", "Total request inference yang dikirim")
REQ_SUCCESS = Counter("msml_requests_success_total", "Total request inference yang sukses")
REQ_FAILED = Counter("msml_requests_failed_total", "Total request inference yang gagal")

REQ_LATENCY = Histogram("msml_request_latency_seconds", "Latency request inference (detik)")

LAST_STATUS = Gauge("msml_last_request_status", "Status request terakhir (1=success, 0=fail)")
LAST_HTTP_CODE = Gauge("msml_last_http_status_code", "HTTP status code terakhir")

PRED_0_TOTAL = Counter("msml_pred_0_total", "Total prediksi kelas 0")
PRED_1_TOTAL = Counter("msml_pred_1_total", "Total prediksi kelas 1")

LAST_PRED_CLASS = Gauge("msml_last_pred_class", "Kelas prediksi terakhir (0/1)")

PAYLOAD_SIZE_BYTES = Histogram("msml_payload_size_bytes", "Ukuran payload request (bytes)")
RESPONSE_SIZE_BYTES = Histogram("msml_response_size_bytes", "Ukuran response (bytes)")

UP = Gauge("msml_exporter_up", "Exporter hidup (1=up)")
LAST_ERROR = Gauge("msml_last_error", "1 jika request terakhir error, else 0")

def sample_payload() -> Dict[str, Any]:
    return {
        "dataframe_split": {
            "columns": [
                "PassengerId","Pclass","Age","SibSp","Parch","Fare",
                "Sex_female","Sex_male",
                "Embarked_C","Embarked_Q","Embarked_S"
            ],
            "data": [
                [1,3,22,1,0,7.25, 0,1, 0,0,1],
                [2,1,38,1,0,71.2833, 1,0, 1,0,0]
            ]
        }
    }

def do_inference_loop(interval_sec: int = 10) -> None:
    UP.set(1)
    while True:
        payload = sample_payload()
        payload_bytes = json.dumps(payload).encode("utf-8")
        PAYLOAD_SIZE_BYTES.observe(len(payload_bytes))

        REQ_TOTAL.inc()
        start = time.time()
        try:
            r = requests.post(
                MODEL_URL,
                headers={"Content-Type": "application/json"},
                data=payload_bytes,
                timeout=10
            )
            elapsed = time.time() - start
            REQ_LATENCY.observe(elapsed)

            LAST_HTTP_CODE.set(r.status_code)

            if r.ok:
                REQ_SUCCESS.inc()
                LAST_STATUS.set(1)
                LAST_ERROR.set(0)

                resp_bytes = r.content if r.content is not None else b""
                RESPONSE_SIZE_BYTES.observe(len(resp_bytes))

                resp = r.json()
                preds = resp.get("predictions", [])
                if isinstance(preds, list) and len(preds) > 0:
                    last = preds[-1]
                    try:
                        last_int = int(last)
                        LAST_PRED_CLASS.set(last_int)
                        if last_int == 0:
                            PRED_0_TOTAL.inc()
                        elif last_int == 1:
                            PRED_1_TOTAL.inc()
                    except Exception:
                        pass
            else:
                REQ_FAILED.inc()
                LAST_STATUS.set(0)
                LAST_ERROR.set(1)

        except Exception:
            elapsed = time.time() - start
            REQ_LATENCY.observe(elapsed)
            REQ_FAILED.inc()
            LAST_STATUS.set(0)
            LAST_HTTP_CODE.set(0)
            LAST_ERROR.set(1)

        time.sleep(interval_sec)

if __name__ == "__main__":
    print(f"Starting exporter on :{EXPORTER_PORT} (metrics at /metrics)")
    start_http_server(EXPORTER_PORT)
    t = threading.Thread(target=do_inference_loop, daemon=True)
    t.start()
    while True:
        time.sleep(1)
