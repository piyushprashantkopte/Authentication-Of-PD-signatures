# Lock to Python 3.11 Bookworm for stable TensorFlow and XGBoost binaries
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside the container
WORKDIR /app

# Install classic system dependencies verified for OpenCV AND curl for downloads
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create the ml_models directory inside the container
RUN mkdir -p /app/ml_models

# Download the 40-class CNN model (95.94% accuracy)
RUN curl -L -o /app/ml_models/signature_cnn_model.h5 "https://drive.google.com/file/d/1rh1G18CXcOWVFDaNqZxNviTt8CmclqJv/view?usp=sharing"

# Download the Random Forest/XGBoost ensemble model (81.67% accuracy)
RUN curl -L -o /app/ml_models/pd_authenticator.pkl "https://drive.google.com/file/d/1YciIQQSg22xDtH1h8IaDzhCf0fDFS7AO/view?usp=sharing"

# Copy the rest of your Django project files into the container
COPY . /app/

# Expose port 8000 for web traffic
EXPOSE 8000

# Run using Gunicorn instead of development runserver
CMD ["gunicorn", "signature_website.wsgi:application", "--bind", "0.0.0.0:8000"]