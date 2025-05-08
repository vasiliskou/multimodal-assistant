# 🧠 Multimodal AI Assistant

A powerful multimodal assistant powered by LLMs (OpenAI GPT-4o, Claude, Gemini, DeepSeek), with translation, image generation, text-to-speech, and speech-to-text capabilities — deployed via Docker and accessible through NGINX on AWS EC2.

---

## ✨ Features

- ✅ Chat with GPT-4o, Claude, Gemini, or DeepSeek
- 🌐 Language translation using Google Translate
- 🎨 Image generation using OpenAI DALL·E or Pollinations
- 🔊 Text-to-Speech via OpenAI or local TTS (gTTS, pyttsx3)
- 🎙️ Speech-to-Text using OpenAI Whisper
- 📦 Dockerized deployment

---

## 📷 Sample Chatbot Interface Screenshot

![App Screenshot](https://drive.google.com/uc?export=view&id=1Ocbc2tg2wAS5H8BbwxGS0xN7wXr8brca)

---

## 🚀 Run on Google Colab

You can try the full app immediately in your browser using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16R2vVg-8wO15Vodan5DFwMtunJClM-Pl?usp=sharing)

---

## 🐳 Run Locally with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/vasiliskou/multimodal-assistant.git
cd multimodal-assistant
```

### 2. Add Your API Keys

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_claude_key
GOOGLE_API_KEY=your_gemini_key
DEEPSEEK_API_KEY=your_deepseek_key
```

### 3. Define Your Configuration

Edit `config.yaml`:

```yaml
model_gpt: "gpt-4o-mini"
model_claude: "claude-3-haiku-20240307"
model_gemini: "gemini-2.0-flash-exp"
model_deepseek: "deepseek-chat"
tts_model: "tts-1"
image_gen_model: "dall-e-3"
use_openai_api: false  # Set to true if using OpenAI's paid APIs
```

### 4. Build the Docker Image

```bash
docker build -t multimodal .
```

### 5. Run the Container

```bash
docker run -d --name gradio-app -p 7860:7860 --env-file .env multimodal
```

---

## 🌐 Deploy on AWS EC2 with NGINX

### 1. SSH into your EC2 instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### 2. Install Docker & NGINX

```bash
sudo apt update
sudo apt upgrade -y
curl -o get-docker.sh https://get.docker.com/
sudo bash get-docker.sh
sudo apt install nginx -y
```

### 3. Copy the project or push to EC2

> You can SCP or clone the repo directly into EC2.

```bash
git clone https://github.com/vasiliskou/multimodal-assistant.git
```

### 4. Configure NGINX

Create `/etc/nginx/sites-available/gradio`:

```nginx
server {
    listen 80;
    server_name your_domain;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable the config:

```bash
sudo ln -s /etc/nginx/sites-available/gradio /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5. (Optional) Secure with HTTPS

If you don't have a domain yet, you can register one using [AWS Route 53](https://aws.amazon.com/route53/) and point it to your EC2 instance via an A record.

Once your domain is set up and pointing to your EC2 public IP:

Install Certbot and the NGINX plugin:

```bash
sudo apt install certbot python3-certbot-nginx -y
```

Obtain and install a free SSL certificate from Let's Encrypt:

```bash
sudo certbot --nginx -d your_domain
```

> Replace `your_domain` with your actual domain.

---

## 📂 Project Structure

```
.
├── app.py              # Main Gradio app
├── config.yaml         # Model + API configuration
├── Dockerfile          # Docker build instructions
├── requirements.txt    # Python dependencies
├── run.sh              # Shell script to run the app
├── README.md           # Project documentation
├── .gitignore          # Git ignored files
├── .dockerignore       # Docker ignored files
```
