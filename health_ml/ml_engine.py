import os
import json
import csv
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.linear_model import LinearRegression

DATA_FILE = "/config/history.csv"
MIN_SAMPLES = 30

# --- Load Add-on Configuration ---
with open("/data/options.json") as f:
    options = json.load(f)

MQTT_HOST = options.get("mqtt_host", "core-mosquitto")
MQTT_USER = options.get("mqtt_user", "")
MQTT_PASS = options.get("mqtt_pass", "")

print("ML Engine started (persistent mode)")

# --- Ensure history file exists ---
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "weight",
            "recovery",
            "cardio",
            "hrv",
            "calories",
            "sleep"
        ])

def load_history():
    X = []
    y = []

    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        for i in range(len(rows) - 1):
            current = rows[i]
            future = rows[i + 1]

            features = [
                float(current["recovery"]),
                float(current["cardio"]),
                float(current["hrv"]),
                float(current["calories"]),
                float(current["sleep"]),
                float(current["weight"])
            ]

            target = float(future["weight"])

            X.append(features)
            y.append(target)

    return np.array(X), np.array(y)

def append_to_history(data):
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            data.get("weight", 0),
            data.get("recovery", 0),
            data.get("cardio", 0),
            data.get("hrv", 0),
            data.get("calories", 0),
            data.get("sleep", 0)
        ])

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)

        append_to_history(data)

        X, y = load_history()

        sample_count = len(X)

        if sample_count < MIN_SAMPLES:
            client.publish(
                "health/ml/model_status",
                json.dumps({"status": f"collecting_data ({sample_count}/{MIN_SAMPLES})"})
            )
            return

        model = LinearRegression()
        model.fit(X, y)

        latest_features = X[-1].reshape(1, -1)

        predicted_next = model.predict(latest_features)[0]

        # 30-day projection (iterative)
        projected_weight = float(latest_features[0][-1])

        temp_features = latest_features.copy()

        for _ in range(30):
            projected_weight = model.predict(temp_features)[0]
            temp_features[0][-1] = projected_weight

        metabolic_probability = min(100, max(0, int((abs(projected_weight - latest_features[0][-1]) * 50))))

        client.publish(
            "health/ml/weight_30d_forecast",
            json.dumps({"forecast": round(projected_weight, 2)})
        )

        client.publish(
            "health/ml/metabolic_probability",
            json.dumps({"probability": metabolic_probability})
        )

        client.publish(
            "health/ml/model_status",
            json.dumps({"status": "active"})
        )

        print("Model updated. Forecast:", projected_weight)

    except Exception as e:
        print("ML ERROR:", e)

# --- MQTT Setup ---
client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_HOST, 1883, 60)

client.subscribe("health/ml/input")
client.on_message = on_message

client.loop_forever()
