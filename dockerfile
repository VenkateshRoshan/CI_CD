FROM python:3.10.0a6-slim-buster

# Set the working directory
WORKDIR /app
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "src/main.py"]