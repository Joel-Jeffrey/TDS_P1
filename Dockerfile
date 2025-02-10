FROM python:3.9-slim

WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Expose the correct port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
