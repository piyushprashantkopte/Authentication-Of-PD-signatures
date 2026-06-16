# Lock to Python 3.11 Bookworm for stable TensorFlow and XGBoost binaries
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside the container
WORKDIR /app

# Install classic system dependencies verified for OpenCV on Bookworm
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your Django project files into the container
COPY . /app/

# Expose port 8000 for web traffic
EXPOSE 8000

# Run using Gunicorn instead of development runserver
CMD ["gunicorn", "signature_website.wsgi:application", "--bind", "0.0.0.0:8000"]