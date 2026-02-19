print("=== TOP OF FILE EXECUTED ===")
print("=== ML ENGINE STARTED ===")
print("STEP 1 - file started")
import json
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest


# --- Load Add-on Configuration ---
with open("/data/options.json") as f:
    options = json.load(f)

MQTT_HOST = options.get("mqtt_host", "core-mosquitto")
MQTT_USER = options.get("mqtt_user", "")
MQTT_PASS = options.get("mqtt_pass", "")
print("STEP 2 - options loaded")
print(options)


# --- ML Model ---
buffer = []
model = IsolationForest(contamination=0.05, random_state=42)


def on_message(client, userdata, msg):
    global buffer

    try:
        data = json.loads(msg.payload)

        features = [
            float(data.get("weight", 0)),
            float(data.get("systolic", 0)),
            float(data.get("heart_rate", 0)),
            float(data.get("sleep", 0))
        ]

        buffer.append(features)

        if len(buffer) >= 50:
            X = np.array(buffer)
            model.fit(X)
            prediction = model.predict([features])[0]
            anomaly = 1 if prediction == -1 else 0

            client.publish(
                "health/ml/anomaly",
                json.dumps({"anomaly": anomaly})
            )

    except Exception as e:
        print("Error processing message:", e)


# --- MQTT Setup ---
print("STEP 3 - about to connect")

try:
    client = mqtt.Client()
    print("STEP 4 - client created")

    client.username_pw_set(MQTT_USER, MQTT_PASS)
    print("STEP 5 - credentials set")

    client.connect(MQTT_HOST, 1883, 60)
    print("STEP 6 - connected")

    client.subscribe("health/ml/input")
    client.on_message = on_message

    print("STEP 7 - entering loop")
    client.loop_forever()

except Exception as e:
    print("MQTT ERROR:", e)
