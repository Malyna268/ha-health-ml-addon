import os
import json
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest

MQTT_HOST = os.environ.get("MQTT_HOST", "core-mosquitto")
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")

buffer = []
model = IsolationForest(contamination=0.05, random_state=42)

def on_message(client, userdata, msg):
    global buffer
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

        client.publish("health/ml/anomaly", json.dumps({"anomaly": anomaly}))

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_HOST, 1883)
client.subscribe("health/ml/input")
client.on_message = on_message
client.loop_forever()

        if len(buffer) >= 50:
            X = np.array(buffer)
            model.fit(X)
            prediction = model.predict([features])[0]
            anomaly = 1 if prediction == -1 else 0

            result = {"anomaly": anomaly}
            client.publish("health/ml/anomaly", json.dumps(result))

    except Exception as e:
        print("Error:", e)

client = mqtt.Client()
client.username_pw_set(MQTT_USER, MQTT_PASS)
client.connect(MQTT_HOST, 1883)
client.subscribe("health/ml/input")
client.on_message = on_message
client.loop_forever()
