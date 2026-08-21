FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Install lightweight runtime dependencies
RUN pip install --no-cache-dir \
    Flask==3.0.3 \
    gunicorn==22.0.0 \
    tensorflow==2.16.1 \
    keras==3.3.3 \
    numpy==1.26.4 \
    h5py==3.11.0

# Copy application files
COPY . /app

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "app:app"]
