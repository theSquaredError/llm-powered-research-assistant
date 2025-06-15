import os
import json
from datetime import datetime

FEEDBACK_LOG = 'data/feedback_log.json'


def store_feedback(payload):
    feedback_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": payload.get("query"),
        "response": payload.get("response"),
        "user_feedback": payload.get("user_feedback")
    }

    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)

    # Append feedback to the log file
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, 'r+', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
            data.append(feedback_entry)
            f.seek(0)
            json.dump(data, f, indent=2)
    else:
        with open(FEEDBACK_LOG, 'w', encoding='utf-8') as f:
            json.dump([feedback_entry], f, indent=2)

    print(f"[✔] Feedback stored at {feedback_entry['timestamp']}")
