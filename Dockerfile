# Use a base Python image
FROM python:3.10-slim

# Install required Python packages
RUN pip install --no-cache-dir pandas nltk

# Download required NLTK data during build (not at runtime)
RUN python -m nltk.downloader punkt stopwords

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 (Flask default)
EXPOSE 5000

# Command to run your app
CMD ["python", "app.py"]
