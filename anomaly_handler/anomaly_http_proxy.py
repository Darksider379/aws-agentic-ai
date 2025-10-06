import json, os, urllib.request

API_URL = os.environ.get("API_URL")  # e.g. https://m3ws9g4vfd.execute-api.us-east-1.amazonaws.com/default/anomaly-handler
TIMEOUT = int(os.environ.get("TIMEOUT_SEC", "25"))

def _http_post_json(url: str, payload=None, timeout=TIMEOUT):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        ctype = resp.headers.get("Content-Type", "")
        if "application/json" in ctype:
            return json.loads(body)
        # try parse anyway if API GW missed header
        try:
            return json.loads(body)
        except Exception:
            return {"raw": body}

def lambda_handler(event, context):
    # Bedrock Agent sends these; echo them back
    action_group = event.get("actionGroup") or "anomaly"
    api_path     = event.get("apiPath") or "/anomaly-handler"
    http_method  = event.get("httpMethod") or "POST"

    # Optional: forward user input to your API if you ever need it
    # For now we just call with empty body
    result = _http_post_json(API_URL, payload=None)

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "apiPath": api_path,
            "httpMethod": http_method,
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": result  # IMPORTANT: dict, not string
                }
            }
        }
    }

