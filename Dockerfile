# Use a minimal Python image
FROM python:3.9-slim

# Avoid prompts from Debian tools
ENV DEBIAN_FRONTEND=noninteractive

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only what's needed
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
