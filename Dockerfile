FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        libxcb1 \
        curl \
        ffmpeg \
        libc++1 \
        libc++abi1 \
        libatomic1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/backend/requirements.txt

# Default is CPU wheels. compose.gpu.yaml passes the CUDA index URL.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# ultralytics is installed with --no-deps so pip does not replace torch
# with CUDA/CPU wheels or the GUI OpenCV package.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r /app/backend/requirements.txt \
    && pip install --no-cache-dir --no-deps ultralytics==8.3.129 \
    && python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY samples /app/samples
COPY backend /app/backend
COPY frontend /app/frontend

ENV PYTHONUNBUFFERED=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    DASHADAS_HOST=0.0.0.0 \
    DASHADAS_PORT=8080 \
    DASHADAS_MODEL=yolo11n.pt \
    DASHADAS_DEVICE=auto

EXPOSE 8080
HEALTHCHECK --interval=20s --timeout=5s --start-period=40s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
