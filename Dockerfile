FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY exec/ exec/
COPY keep_alive.sh .

RUN chmod +x exec/* keep_alive.sh

ENV PYTHONPATH=src

CMD ["./keep_alive.sh"]