FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY handler.py .

ENV HF_HOME=/runpod-volume/hf \
    PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
