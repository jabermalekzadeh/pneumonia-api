# Use a Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required dependencies
RUN pip install torch torchvision pillow fastapi uvicorn python-multipart

# Copy model and API script
COPY pneumonia_model.pth /app/
COPY app.py /app/

# Expose the port for API access
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
