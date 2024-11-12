# Build stage
FROM python:3.12-slim AS builder

# Establece el directorio de trabajo
WORKDIR /app/api

# Instala las dependencias necesarias para construir el proyecto
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    zlib1g-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia y instala las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1+cpu torchvision==0.18.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    && pip install --no-cache-dir -r requirements.txt

# Elimina herramientas de compilación y limpia caché
RUN apt-get remove -y build-essential && apt-get autoremove -y && rm -rf /root/.cache/pip

# Runtime stage
FROM python:3.12-slim

# Establece el directorio de trabajo
WORKDIR /app/api

# Instala solo las dependencias necesarias para la ejecución
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copia solo las dependencias instaladas en la imagen de construcción
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Copia los archivos de la aplicación
COPY . .

# Variables de entorno
ENV PYTHONPATH=/app
ENV CHECKPOINT_PATH=/app/api/best_model.pth
ENV SUPPORT_FOLDER=/app/api/support_images
ENV DEVICE=cpu

# Exposición del puerto para FastAPI
EXPOSE 8000

# Comando de inicio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
