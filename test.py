import time
import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
headers = {"Authorization": "Bearer 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"}


def query(payload):
    while True:
        response = requests.post(API_URL, headers=headers, json=payload)
        result = response.json()
        if 'error' in result and 'loading' in result['error']:
            print("Model is loading, waiting 20 seconds...")
            time.sleep(20)
        else:
            return result


output = query({
    "inputs": {
        "source_sentence": "That is a happy person",
        "sentences": [
            "3",
            "1",
            "2"
        ]
    },
})

print(output)
