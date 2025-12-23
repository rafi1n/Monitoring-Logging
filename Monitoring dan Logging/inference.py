import time
import requests

URL = "http://localhost:5000/invocations"

payload = {
    "dataframe_split": {
        "columns": [
            "PassengerId","Pclass","Age","SibSp","Parch","Fare",
            "Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S"
        ],
        "data": [
            [1,3,22,1,0,7.25,0,1,0,0,1],
            [2,1,38,1,0,71.2833,1,0,1,0,0]
        ]
    }
}

if __name__ == "__main__":
    t0 = time.time()
    r = requests.post(URL, json=payload, timeout=10)
    dt = time.time() - t0
    print("status:", r.status_code, "latency_sec:", round(dt, 4))
    print(r.text)
