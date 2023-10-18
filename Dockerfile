# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install Flask and any other necessary packages
RUN pip install flask numpy pandas scikit-learn joblib gunicorn

# Copy the Python files into the container at /app
ADD . /app

# Set the working directory in the container
WORKDIR /app

# Expose port 10000 for the Flask app (HTTP)
EXPOSE 10000

# Run index.py with Gunicorn when the container launches, enabling HTTPS
CMD ["gunicorn", "-b", "0.0.0.0:10000", "index:app"]
