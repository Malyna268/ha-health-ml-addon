FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    paho-mqtt \
    numpy \
    scikit-learn

WORKDIR /app

COPY ml_engine.py /app/ml_engine.py

CMD ["python", "-u", "/app/ml_engine.py"]
