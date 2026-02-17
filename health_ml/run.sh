#!/usr/bin/with-contenv bashio

MQTT_HOST=$(bashio::config 'mqtt_host')
MQTT_USER=$(bashio::config 'mqtt_user')
MQTT_PASS=$(bashio::config 'mqtt_pass')

python3 /ml_engine.py "$MQTT_HOST" "$MQTT_USER" "$MQTT_PASS"
