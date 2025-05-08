FROM python:3.11.11

# Set working directory
WORKDIR /app

# Copy project files including hidden files (like .env)
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y espeak alsa-utils
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Gradio default port
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "app.py"]
