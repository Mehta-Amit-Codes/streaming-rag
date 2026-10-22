FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
# sentence-transformers pulls in torch, so this layer is large. For a slim demo image, build with
# EMBEDDER=hash and drop sentence-transformers from requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8005
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"]
