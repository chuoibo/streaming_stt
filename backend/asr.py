import asyncio
import websockets
import pyaudio
import webrtcvad
import time
import json
from collections import deque
import threading
import queue

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_SIZE = 480  # 30ms at 16000 Hz
VAD_FRAME_MS = 30
SILENCE_THRESHOLD = 3.0  # 3 seconds in float

# WebSocket URL
WEBSOCKET_URL = 'wss://asr.gpu.rdhasaki.com/se'

# Initialize VAD
vad = webrtcvad.Vad()
vad.set_mode(1)

# Queue for audio chunks
audio_queue = queue.Queue()

def audio_capture_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
    print("Listening... Speak into the microphone.")
    while True:
        audio_chunk = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_queue.put(audio_chunk)

async def send_audio_chunk(websocket, audio_data):
    st = time.time()
    await websocket.send(audio_data)

async def receive_responses(websocket):
    while True:
        try:
            response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            response = json.loads(response)
            print(response)
        except asyncio.TimeoutError:
            print("No response within timeout period")

async def process_audio_stream():
    threading.Thread(target=audio_capture_thread, daemon=True).start()
    async with websockets.connect(WEBSOCKET_URL) as websocket:
        asyncio.create_task(receive_responses(websocket))
        is_speaking = False
        speech_buffer = deque()
        last_speech_time = None  # Timestamp of last speech chunk

        while True:
            audio_chunk = await asyncio.to_thread(audio_queue.get)
            is_speech = vad.is_speech(audio_chunk, sample_rate=RATE)
            current_time = time.time()

            if is_speech and not is_speaking:
                # Start of speech
                print("Speech detected, starting transcription...")
                is_speaking = True
                last_speech_time = current_time
                speech_buffer.append(audio_chunk)

            elif is_speech and is_speaking:
                # Ongoing speech
                last_speech_time = current_time  # Update last speech time
                speech_buffer.append(audio_chunk)
                if len(b''.join(speech_buffer)) >= 10240:
                    audio_data = b''.join(speech_buffer)[:10240]
                    speech_buffer.clear()
                    await send_audio_chunk(websocket, audio_data)

            elif not is_speech and is_speaking:
                # Silence during speech, check elapsed time
                speech_buffer.append(audio_chunk)  # Keep buffering
                if last_speech_time is not None and (current_time - last_speech_time) >= SILENCE_THRESHOLD:
                    # 3 seconds of silence since last speech
                    print("Speech ended (3s silence), sending final audio...")
                    is_speaking = False
                    last_speech_time = None
                    audio_data = b''.join(speech_buffer)
                    speech_buffer.clear()
                    await send_audio_chunk(websocket, audio_data)
                    await send_audio_chunk(websocket, b'')

            elif not is_speech and not is_speaking:
                # No speech, no action needed
                last_speech_time = None

if __name__ == "__main__":
    try:
        asyncio.run(process_audio_stream())
    except KeyboardInterrupt:
        print("\nStopped by user")