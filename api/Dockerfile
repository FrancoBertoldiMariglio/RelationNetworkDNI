FROM python:3.12-slim
 
WORKDIR /app/api
 
# Instalamos dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
&& rm -rf /var/lib/apt/lists/*
 
# Instalamos pip y las dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0+cpu \
    torchvision==0.17.0+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    pillow \
    python-dotenv \
    python-multipart \
	numpy==1.26.4
 
# Limpiamos dependencias de compilación
RUN apt-get purge -y gcc g++ \
&& apt-get autoremove -y
 
# Copiamos la aplicación
COPY . .
 
ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/app/api/best_model.pth
ENV SUPPORT_FOLDER=/app/api/support_images
ENV DEVICE=cpu
 
EXPOSE 8000
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
