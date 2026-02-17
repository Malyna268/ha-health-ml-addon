ARG BUILD_FROM=ghcr.io/home-assistant/amd64-base:latest
FROM $BUILD_FROM

RUN apk add --no-cache \
    python3 \
    py3-pip

RUN pip3 install \
    paho-mqtt \
    numpy \
    scikit-learn

WORKDIR /app

COPY run.sh /run.sh
COPY ml_engine.py /ml_engine.py

RUN chmod a+x /run.sh

CMD [ "/run.sh" ]
