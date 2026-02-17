import sys
import json
import numpy as np
import paho.mqtt.client as mqtt
from sklearn.ensemble import IsolationForest

MQTT_HOST = sys.argv[1]
MQTT_USER = sys.argv[2]
MQTT_PASS = sys.argv[3]

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
