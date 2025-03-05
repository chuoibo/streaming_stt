from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import asyncio
import torch
from typing import List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(
    title="Real-Time Audio Processor",
    description="Process and transcribe audio in real-time using Whisper"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model
model = whisper.load_model("model/pytorch_model.bin", device)


