import os                        # Operating system utilities
import sys                       # System-specific parameters and functions
import platform                  # Detect the platform (Windows/Linux/Mac)
import json                      # JSON parsing and formatting
import base64                    # Encoding and decoding base64 binary data
from io import BytesIO            # Handle binary I/O streams

# === LLM APIs ===
from openai import OpenAI         # OpenAI API client
from anthropic import Anthropic   # Anthropic API client

# === Display & Multimedia (Jupyter / Colab) ===
from IPython.display import Markdown, display, update_display, Audio  # Rich display utilities

# === Gradio ===
import gradio as gr               # Gradio UI framework for interactive apps
from gradio import processing_utils  # Gradio utility functions

# === Text-to-Speech (TTS) ===
from pyttsx3 import init          # Free local/offline text-to-speech engine
from gtts import gTTS             # Google Text-to-Speech (useful for headless or Colab environments)

# === Translation ===
from deep_translator import GoogleTranslator  # Free translation using Google Translate

# === HTTP Requests ===
import requests                   # Make HTTP requests (API calls, downloads, etc.)

import yaml
from dotenv import load_dotenv

# ── Environment / configuration ───────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY     = os.getenv("GOOGLE_API_KEY")      # kept for future use
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY")    # kept for future use

# YAML file with model ids & feature flags
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

MODEL_GPT        = cfg["model_gpt"]        # e.g. "gpt-4o"
MODEL_CLAUDE     = cfg["model_claude"]     # e.g. "claude-3-opus-20240229"
MODEL_GEMINI     = cfg["model_gemini"]
MODEL_DEEPSEEK   = cfg["model_deepseek"]
USE_OPENAI_API   = cfg["use_openai_api"]
TTS_MODEL        = cfg["tts_model"]        # e.g. "tts-1-hd"
STT_MODEL        = cfg["stt_model"] 
IMAGE_GEN_MODEL  = cfg["image_gen_model"]  # e.g. "dall-e-3"

# set up environment
openai = OpenAI(api_key=OPENAI_API_KEY)                              # OpenAI (GPT-4o-mini)
claude = Anthropic(api_key=ANTHROPIC_API_KEY)                        # Anthropic (Claude 3 Haiku)
gemini_via_openai_client = OpenAI(                                   # Google Gemini (via OpenAI-compatible client)
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
deepseek_via_openai_client = OpenAI(                                 # DeepSeek (via OpenAI-compatible client)
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

"""### 🧠 Model Constants & System Prompt"""

# constants
MODEL_GPT = 'gpt-4o-mini'
MODEL_CLAUDE = "claude-3-haiku-20240307"
MODEL_GEMINI = "gemini-2.0-flash-exp"
MODEL_DEEPSEEK = "deepseek-chat"

USE_OPENAI_API = False # If True, use OpenAI's paid APIs (for Whisper STT, TTS, Image Generation). If False, use free/local alternatives.
TTS_MODEL = "tts-1"
IMAGE_GEN_MODEL = "dall-e-3"

system_message = "You are an assistant that takes a technical question and respond with an explanation. "
system_message += "Also you can handle image generation and translation through tools."

"""### 📡 Unified Model Streaming Interface for Multi-Provider Chat Models:

This module provides a unified interface for streaming responses from various chat models, including OpenAI's GPT, Google's Gemini (via OpenAI-compatible API), DeepSeek (via OpenAI-compatible API), and Anthropic's Claude.
It supports optional tool use (functions / tools API) and yields streamed responses suitable for interactive UIs like Gradio.
"""

def call_openai(user_history, model, tools=None):
    # Combine system prompt with conversation history
    messages = [{"role": "system", "content": system_message}] + user_history
    client = openai  # Default OpenAI client

    # Select appropriate client and model name based on provider
    if model == "Gemini":
        client = gemini_via_openai_client
        model_name = MODEL_GEMINI
    elif model == "Deepseek":
        client = deepseek_via_openai_client
        model_name = MODEL_DEEPSEEK
    elif model == "GPT":
        model_name = MODEL_GPT
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Set up request with streaming
    request_params = {
        "model": model_name,
        "messages": messages,
        "stream": True,
    }

    # Add tools if available
    if tools:
        request_params["tools"] = tools
        request_params["tool_choice"] = "auto"

    # Start streaming response
    stream_resp = client.chat.completions.create(**request_params)
    for chunk in stream_resp:
        yield chunk


def call_claude(user_history, tools=None):
    # Claude expects system prompt as a separate argument, not in message list
    cleaned_messages = [
        {k: v for k, v in msg.items() if k not in ("metadata", "options")}
        for msg in user_history
        if msg["role"] != "system"
    ]

    # Start Claude stream
    stream_resp = claude.messages.stream(
        model=MODEL_CLAUDE,
        max_tokens=200,
        temperature=0.7,
        system=system_message,
        messages=cleaned_messages
    )

    # Wrap Claude-style stream chunks to mimic OpenAI-like structure
    class ClaudeChunk:
        def __init__(self, content):
            self.choices = [
                type("Choice", (), {
                    "delta": type("DeltaPart", (), {"content": content})(),
                    "finish_reason": None
                })()
            ]

    with stream_resp as stream:
        for text in stream.text_stream:
            if text:
                yield ClaudeChunk(text)


def stream_model(user_history, model, tools=None):
    # Unified interface for OpenAI-compatible and Claude models
    if model in ["GPT", "Gemini", "Deepseek"]:
        return call_openai(user_history, model, tools=tools)
    elif model == "Claude":
        return call_claude(user_history, tools=tools)
    else:
        raise ValueError("Unknown model")

"""### 🌐 Translate & Image Generation Tools (via Function Calling)

This code defines two tools usable via OpenAI-style function calling:

- **translate(message, dest_lang)** – Uses deep_translator to translate text into a target language.
- **image_generation(prompt)** – Generates an image using either OpenAI’s DALL·E or Pollinations API, based on the USE_OPENAI_API flag.

Both tools are registered in the tools list and are automatically callable by supported LLMs (e.g., GPT, Gemini, DeepSeek).

"""

def translate(message, dest_lang):
    # Log the language we're translating to
    print(f"Tool Translate called for language: {dest_lang}")
    try:
        # Use deep_translator to translate text from any language to the target language
        translated_text = GoogleTranslator(source='auto', target=dest_lang).translate(message)
        return translated_text
    except Exception as e:
        # Handle and log errors during translation
        print(f"Translation error: {e}")
        return f"Error translating message: {e}"


def image_generation(prompt):
    """
    Generates an image either via OpenAI DALL-E API or Pollinations API (free)
    depending on the USE_OPENAI_API flag.
    """
    print("Tool Image generation called")

    if USE_OPENAI_API:
        # OpenAI API
        image_response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
        # Return base64 string directly
        return image_response.data[0].b64_json
    else:
        # Free Pollinations API
        url = f"https://pollinations.ai/p/{prompt}"
        response = requests.get(url)
        if response.status_code != 200:
            print("Image generation failed.")
            return None
        # Convert downloaded image to base64 string
        image_base64 = base64.b64encode(response.content).decode("utf-8")
        return image_base64

# Tool specification for OpenAI-compatible function calling
translate_function = {
    "name": "translate",
    "description": "Translate this to a specified language",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message in English to be translated."
            },
            "dest_lang": {
                "type": "string",
                "description": "The language code to translate the message into (e.g., 'fr' for French, 'es' for Spanish, 'de' for German)."
            }
        },
        "required": ["message", "dest_lang"],
        "additionalProperties": False
    }
}

# Tool specification for OpenAI-compatible function calling
image_generation_function = {
    "name": "image_generation",
    "description": "Generate an image",
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "A detailed description of the desired image."
            }
        },
        "required": ["prompt"],
        "additionalProperties": False
    }
}


# Tools List
tools = [
    {"type": "function", "function": translate_function},
    {"type": "function", "function": image_generation_function}
]

"""### 🛠️ Tool Call Extractor from Streaming Chunks:
This function processes streamed OpenAI response chunks to extract structured tool call data. It reconstructs full tool call information (like function name and arguments), handling multi-part streaming, and returns a dictionary that mimics a valid assistant message containing tool calls.
"""

def extract_tool_calls_from_chunks(chunks):
    tool_calls = {}

    for chunk in chunks:
        delta = chunk.choices[0].delta
        # Skip chunks that don't contain tool call data
        if not delta.tool_calls:
            continue

        for tool_call_delta in delta.tool_calls:
            index = tool_call_delta.index  # Index used to group calls across multiple chunks

            # Initialize a new tool call entry if not already present
            if index not in tool_calls:
                tool_calls[index] = {
                    "id": tool_call_delta.id or f"temp_{index}",  # Fallback ID if missing
                    "name": "",
                    "arguments": ""
                }

            func = tool_call_delta.function
            if func:
                # Collect function name if provided
                if func.name:
                    tool_calls[index]["name"] = func.name
                # Append arguments as they arrive chunk by chunk
                if func.arguments:
                    tool_calls[index]["arguments"] += func.arguments

    # Build the list of formatted tool call objects
    tool_call_values = [
        {
            "id": call["id"],
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": call["arguments"]
            }
        }
        for call in tool_calls.values()
        if call["name"]  # Filter out incomplete tool calls
    ]

    # If no tool calls were detected, return None
    if not tool_call_values:
        return None

    # Return a formatted assistant message containing tool calls
    return {
        "role": "assistant",
        "tool_calls": tool_call_values
    }

"""### 🧠 Handle Tool Calls from Assistant Messages

This function extracts and processes tool calls requested by the assistant.  
It automatically detects the tool name (e.g., translate, image_generation) and executes the corresponding local Python function.  
It returns:
- A list of tool response messages formatted for OpenAI-compatible models.
- Optionally, an image path if an image was generated.

"""

def handle_tool_call_data(assistant_tool_message):
    responses = []
    image_path = None

    # Extract tool_calls from the assistant message
    tool_calls = assistant_tool_message.get("tool_calls", [])

    for call in tool_calls:
        function_name = call["function"]["name"]
        tool_call_id = call["id"]

        # Parse JSON arguments safely
        try:
            arguments = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError:
            arguments = {}

        # Handle translation tool
        if function_name == "translate":
            message = arguments.get("message")
            dest_lang = arguments.get("dest_lang")

            # Basic validation
            if not message or not dest_lang:
                result = {"error": "'message' or 'dest_language' is missing"}
            else:
                # Call translation function
                translation = translate(message, dest_lang)
                result = {"dest_lang": dest_lang, "result": translation}

        # Handle image generation tool
        elif function_name == "image_generation":
            prompt = arguments.get("prompt")
            if not prompt:
                result = {"error": "'prompt' is missing"}
            else:
                image_b64 = image_generation(prompt)
                if image_b64:
                    # Decode and save image as a local file
                    image_data = base64.b64decode(image_b64)
                    image_path = f"generated_image_{tool_call_id}.png"
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    result = {"prompt": prompt, "status": "Image generated successfully"}
                else:
                    result = {"error": "Image generation failed"}

        # Unknown tool handler
        else:
            result = {"error": f"Unknown tool: {function_name}"}

        # Add tool response to the return list
        responses.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": json.dumps(result)
        })

    return responses, image_path

"""### 🔊 Text-to-Speech and Speech-to-Text (Paid & Free Options)

This section provides both **paid API-based** and **free offline** options for Text-to-Speech (TTS) and Speech-to-Text (STT):

- **Paid Options**:
   - OpenAI's tts-1 model for high-quality text-to-speech.
   - OpenAI's whisper-1 model for speech-to-text.

- **Free Offline / Open-Source Options**:
   - pyttsx3: Local text-to-speech (offline).
   - gTTS: Google Text-to-Speech (useful for headless environments like Colab).
   - openai-whisper: Open-source speech-to-text (offline).

"""

def text_to_speech_handler(message):
    if USE_OPENAI_API:
        response = openai.audio.speech.create(
            model=TTS_MODEL,
            voice="onyx",
            input=message
        )
        audio_stream = BytesIO(response.content)
        audio_path = "openai_tts_output.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_stream.getbuffer())
        return audio_path

    elif "google.colab" in sys.modules or platform.system() == "Linux":
        print("🔄 Using gTTS (Google Text-to-Speech) for Colab or Linux")
        tts = gTTS(text=message, lang='en')
        audio_path = "gtts_output.mp3"
        tts.save(audio_path)
        return audio_path

    else:
        print("⚠️ Audio playback not supported in this environment.")
        return None



def speech_to_text_handler(audio_path):
    """
    Speech-to-Text function

    """
    if not audio_path or not os.path.exists(audio_path):
        return ""

    # 💸 OpenAI Whisper API (paid)
    try:
        with open(audio_path, "rb") as f:
            transcript = openai.audio.transcriptions.create(
                model= STT_MODEL,
                file=f
            )
        return transcript.text
    except Exception as e:
        print(f"[OpenAI Transcription Error]: {e}")
        return ""


"""### 📝 Main Chat Function with Tool Call and Text-to-Speech Support


This function handles the main chat interaction with the assistant. It streams responses from the selected model.
Handles tool calls (if detected).
Supports optional text-to-speech playback using pyttsx3.
"""
def chat(history, model, text_to_speech):
    # ── 1️⃣ build prompt for the LLM (strip non-strings) ───────────────────
    history = [m for m in history if not isinstance(m["content"], gr.Image)]
    messages = [{"role": "system", "content": system_message}] + [
        m for m in history if isinstance(m["content"], str)
    ]

    # ── 2️⃣ first stream ───────────────────────────────────────────────────
    answer, chunks = "", []
    for chunk in stream_model(messages, model, tools):
        chunks.append(chunk)
        if chunk.choices[0].delta.content:
            answer += chunk.choices[0].delta.content
            yield history + [{"role": "assistant", "content": answer}], None

    # ── 3️⃣ handle tool calls (image etc.) ─────────────────────────────────
    if chunk.choices[0].finish_reason == "tool_calls":
        tc_msg           = extract_tool_calls_from_chunks(chunks)
        messages.append(tc_msg)
        tool_out, img_path = handle_tool_call_data(tc_msg)
        messages += tool_out

        if img_path:                                  # show image once
            history.append({"role": "assistant",
                           "content": gr.Image(value=img_path)})
            yield history, None

        # continue conversation after tool
        answer = ""
        for chunk in stream_model(messages, model):
            if chunk.choices[0].delta.content:
                answer += chunk.choices[0].delta.content
                yield history + [{"role": "assistant", "content": answer}], None

    # ── 4️⃣ optional text-to-speech  (‼️  FIX  ‼️) ─────────────────────────
    if text_to_speech == "Yes":
        audio_path = text_to_speech_handler(answer)
        if audio_path:
            # *don’t* send history again – just update the audio component
            yield gr.update(), audio_path

"""### 🎛️ Gradio Chat Interface with Voice Support
This Gradio UI provides a custom chat interface with:

* Model selection (GPT, Claude, Gemini, Deepseek)

* Text-to-speech toggle

* Speech-to-text via microphone

* A flexible chat display and message input system

It mimics modern assistants like ChatGPT with multimodal input/output capabilities.
"""

# Create the full Gradio interface block
with gr.Blocks() as ui:

    # Row for selecting model and enabling text-to-speech
    with gr.Row():
        model_name = gr.Radio(
            choices=["GPT", "Gemini", "Deepseek", "Claude"],
            label="Select model",
            value="GPT"
        )
        text_to_speech = gr.Radio(
            choices=["No", "Yes"],
            label="🔊 Enable text-to-speech?",
            value="No"
        )

    # Row to show the chat history
    with gr.Row():
        chatbot = gr.Chatbot(height=350, type="messages")

    # Row for text input and microphone-based audio input
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:", scale=4)
        speech_to_text = gr.Audio(
            sources=["microphone"],  # Enable mic recording
            type="filepath",         # Return audio as a file path
            label="🎙️",
            show_label=True,
            scale=1
        )
        audio_btn = gr.Audio(autoplay=True, visible=False)

    # Row with clear button
    with gr.Row():
        clear = gr.Button("Clear")

    #Function to handle text input and update the history
    def do_entry(message, history):
        history += [{"role": "user", "content": message}]
        return "", history

    # When mic input is used, transcribe the audio to text
    speech_to_text.change(
        fn=lambda audio: speech_to_text_handler(audio_path = audio) if audio else "",
        inputs=speech_to_text,
        outputs=entry
    )

    # When user presses Enter in the textbox
    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=[chatbot, model_name, text_to_speech], outputs=[chatbot, audio_btn]
    ).then(
        lambda: gr.update(value=None),  # Clear mic input after chat
        None,
        outputs=speech_to_text
    )

    # Clear button to reset chat
    clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

# Launch Gradio app in the browser
ui.launch()