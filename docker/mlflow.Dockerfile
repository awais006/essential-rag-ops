# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir "mlflow>=3,<4" psycopg2-binary
EXPOSE 5000