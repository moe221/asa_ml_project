FROM python:3.10.12-bookworm

COPY api /api
COPY requirements.txt /requirements.txt
COPY certs /certs

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT
