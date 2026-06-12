FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY exec/ exec/

RUN chmod +x exec/*

ENV PYTHONPATH=src

CMD ["./exec/keep_alive"]