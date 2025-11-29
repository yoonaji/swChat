FROM python:3.10-slim

WORKDIR /app

COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/app_gateway.py .

COPY retriever/retriever_pb2.py .
COPY retriever/retriever_pb2_grpc.py .
COPY generator/generator_pb2.py .
COPY generator/generator_pb2_grpc.py .

CMD ["uvicorn", "app_gateway:app", "--host", "0.0.0.0", "--port", "8000"]