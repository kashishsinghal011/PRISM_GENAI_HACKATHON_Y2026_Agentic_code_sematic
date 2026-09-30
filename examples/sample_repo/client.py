import time

import requests


def call_api(payload, retries=3):
    """POST the payload to the API, retrying failed calls with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.post("https://api.example.com/v1", json=payload, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)
    raise RuntimeError("API unavailable after retries")
