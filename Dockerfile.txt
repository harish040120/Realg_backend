# Use lightweight Python image
FROM python:3.10-slim

# Install only required OS packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only what's needed
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code last (prevents rebuilds on every change)
COPY . .

# Expose app port
EXPOSE 5000

# Run your app
CMD ["python", "app.py"]
