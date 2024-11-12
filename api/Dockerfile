FROM python:3.12-slim

WORKDIR /app/api

# Instalamos dependencias del sistema mínimas
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Instalamos pip y las dependencias mínimas de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torchvision==0.17.0+cpu \
    && pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "pydantic<2.0" \
    pillow \
    python-dotenv \
    python-multipart \
    numpy==1.26.4

# Copiamos solo los archivos necesarios
COPY RelationNet.py /app/api/
COPY routes.py /app/api/
COPY service.py /app/api/
COPY schemas.py /app/api/
COPY main.py /app/api/
COPY best_model.pth /app/api/
COPY support_images /app/api/support_images

ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/app/api/best_model.pth
ENV SUPPORT_FOLDER=/app/api/support_images
ENV DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
