# Uzbek Speech-to-Text (STT) Fine-Tuning
This project fine-tunes OpenAI's Whisper model on the Uzbek language using Mozilla Common Voice dataset. The goal is to improve automatic speech recognition (ASR) for Uzbek speakers.

Features

Fine-tunes openai/whisper-small on Uzbek speech data

Uses Mozilla Common Voice dataset

Evaluates performance using Word Error Rate (WER)

Runs in a WSL + Ubuntu virtual environment to avoid local installation issues

Installation

1. Set Up WSL and Ubuntu

If not installed, set up WSL (Windows Subsystem for Linux) with Ubuntu:

wsl --install -d Ubuntu

2. Install Dependencies

Inside Ubuntu, install Python and required libraries:

sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip git
pip install torch transformers datasets jiwer openai-whisper

3. Clone the Repository

git clone https://github.com/yourusername/uzbek-stt.git
cd uzbek-stt

Dataset Setup

Log in to Hugging Face to access the Mozilla Common Voice dataset:

huggingface-cli login

Then, download the dataset:

from datasets import load_dataset
dataset = load_dataset("mozilla-foundation/common_voice_13_0", "uz", trust_remote_code=True)

Fine-Tuning the Model

Run the fine-tuning script inside WSL:

python scripts/fine_tune_stt.py

This will train the model and save it in models/whisper-uzbek.

Evaluating the Model

To test the model's accuracy:

python scripts/evaluate_stt.py

Usage

To transcribe an Uzbek audio file:

from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

model = WhisperForConditionalGeneration.from_pretrained("models/whisper-uzbek")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="uz")

def transcribe(audio_path):
    inputs = processor(audio_path, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

transcription = transcribe("sample_uzbek_audio.wav")
print("Transcription:", transcription)

Contributing

Fork the repository

Create a new branch (git checkout -b feature-branch)

Commit changes (git commit -m "Added new feature")

Push to GitHub (git push origin feature-branch)

Open a Pull Request
