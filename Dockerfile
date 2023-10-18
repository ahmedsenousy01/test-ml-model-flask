# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install Flask and any other necessary packages
RUN pip install flask numpy pandas scikit-learn joblib

# Copy the Python files into the container at /app
ADD . /app

# Set the working directory in the container
WORKDIR /app

# Generate a self-signed SSL certificate and key (for testing purposes)
RUN openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost"

# Expose port 5000 for the Flask app (HTTP)
EXPOSE 5000

# Expose port 5001 for the Flask app (HTTPS)
EXPOSE 5001

# Run index.py with Gunicorn when the container launches, enabling HTTPS
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--certfile=cert.pem", "--keyfile=key.pem", "index:app"]
