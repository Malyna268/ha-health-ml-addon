ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:latest
FROM $BUILD_FROM

RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    python3-dev \
    py3-numpy \
    py3-scipy \
    py3-scikit-learn

RUN pip3 install --no-cache-dir --break-system-packages \
    paho-mqtt

WORKDIR /app

COPY ml_engine.py /app/ml_engine.py
COPY run.sh /etc/services.d/health_ml/run

RUN chmod a+x /etc/services.d/health_ml/run
