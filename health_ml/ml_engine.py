import os
import json
import csv
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.linear_model import LinearRegression

DATA_DIR = "/addon_configs/health_ml"
DATA_FILE = os.path.join(DATA_DIR, "history_v2.csv")

MIN_SAMPLES = 14
ROLLING_WINDOW = 14

# Ensure directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Load config
with open("/data/options.json") as f:
    options = json.load(f)

MQTT_HOST = options.get("mqtt_host", "core-mosquitto")
MQTT_USER = options.get("mqtt_user", "")
MQTT_PASS = options.get("mqtt_pass", "")

print("ML Engine v2 started (Delta Adaptive Mode)")

# Create file if not exists
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date",
            "weight",
            "delta_weight",
            "recovery",
            "delta_recovery",
            "sleep",
            "delta_sleep",
            "calories",
            "delta_calories",
            "hrv",
            "delta_hrv"
        ])

def load_rows():
    with open(DATA_FILE, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)

def append_row(data):
    rows = load_rows()
    today = datetime.now().strftime("%Y-%m-%d")

    weight = float(data.get("weight", 0))
    recovery = float(data.get("recovery", 0))
    sleep = float(data.get("sleep", 0))
    calories = float(data.get("calories", 0))
    hrv = float(data.get("hrv", 0))

    if rows:
        prev = rows[-1]
        delta_weight = weight - float(prev["weight"])
        delta_recovery = recovery - float(prev["recovery"])
        delta_sleep = sleep - float(prev["sleep"])
        delta_calories = calories - float(prev["calories"])
        delta_hrv = hrv - float(prev["hrv"])
    else:
        delta_weight = delta_recovery = delta_sleep = delta_calories = delta_hrv = 0

    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            today,
            weight,
            delta_weight,
            recovery,
            delta_recovery,
            sleep,
            delta_sleep,
            calories,
            delta_calories,
            hrv,
            delta_hrv
        ])

def train_and_predict(client):
    rows = load_rows()

    if len(rows) < MIN_SAMPLES:
        client.publish(
            "health/ml/model_status",
            json.dumps({"status": f"collecting_data ({len(rows)}/{MIN_SAMPLES})"})
        )
        return

    recent = rows[-ROLLING_WINDOW:]

    X = []
    y = []

    for i in range(len(recent)-1):
        X.append([
            float(recent[i]["delta_recovery"]),
            float(recent[i]["delta_sleep"]),
            float(recent[i]["delta_calories"]),
            float(recent[i]["delta_hrv"])
        ])
        y.append(float(recent[i+1]["delta_weight"]))

    X = np.array(X)
    y = np.array(y)

    model = LinearRegression()
    model.fit(X, y)

    last = recent[-1]
    last_features = np.array([[
        float(last["delta_recovery"]),
        float(last["delta_sleep"]),
        float(last["delta_calories"]),
        float(last["delta_hrv"])
    ]])

    predicted_delta = model.predict(last_features)[0]

    current_weight = float(last["weight"])
    projected_weight = current_weight

    for _ in range(30):
        projected_weight += predicted_delta

    slowdown_risk = 0
    if predicted_delta > -0.05:
        slowdown_risk = 70
    elif predicted_delta > -0.1:
        slowdown_risk = 40
    else:
        slowdown_risk = 10

    confidence = min(100, int(len(rows) / 30 * 100))

    client.publish(
        "health/ml/weight_30d_forecast",
        json.dumps({"forecast": round(projected_weight, 2)})
    )

    client.publish(
        "health/ml/metabolic_probability",
        json.dumps({"probability": slowdown_risk})
    )

    client.publish(
        "health/ml/confidence",
        json.dumps({"confidence": confidence})
    )

    client.publish(
        "health/ml/model_status",
        json.dumps({"status": "active"})
    )

    print("Prediction updated:", projected_weight)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload)
        append_row(data)
        train_and_predict(client)
    except Exception as e:
        print("ML ERROR:", e)

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_HOST, 1883, 60)

client.subscribe("health/ml/input")
client.on_message = on_message

client.loop_forever()
