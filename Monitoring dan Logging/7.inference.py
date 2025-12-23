import json
import requests

SERVE_URL = "http://localhost:5000/invocations"

def build_payload():
    columns = [
        "PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare",
        "Sex_female", "Sex_male",
        "Embarked_C", "Embarked_Q", "Embarked_S"
    ]

    data = [
        [1, 3, 22, 1, 0, 7.25, 0, 1, 0, 0, 1],
        [2, 1, 38, 1, 0, 71.2833, 1, 0, 1, 0, 0],
    ]

    return {"dataframe_split": {"columns": columns, "data": data}}

def main():
    payload = build_payload()
    headers = {"Content-Type": "application/json"}

    r = requests.post(SERVE_URL, headers=headers, data=json.dumps(payload), timeout=30)
    print("HTTP", r.status_code)
    print(r.text)

if __name__ == "__main__":
    main()
