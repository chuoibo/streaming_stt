import asyncio
import websockets
import json
import webrtcvad
import struct
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# External ASR WebSocket endpoint
ASR_WEBSOCKET_URL = 'wss://asr.gpu.rdhasaki.com/se'

class SpeechProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressiveness mode (3 is most aggressive)
        self.rate = 16000     # Sample rate
        self.frame_duration = 30  # Frame duration in ms
        self.chunk = int(self.rate * self.frame_duration / 1000)
        
    def is_speech(self, audio_data):
        """Check if audio frame contains speech."""
        try:
            return self.vad.is_speech(audio_data, self.rate)
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return False
            
    @staticmethod
    def create_wav_header(sample_rate, channels, bits_per_sample, data_length):
        """Create a WAV header for the audio data."""
        header = bytearray()
        header.extend(b'RIFF')
        header.extend(struct.pack('<L', 36 + data_length))
        header.extend(b'WAVE')
        header.extend(b'fmt ')
        header.extend(struct.pack('<L', 16))
        header.extend(struct.pack('<H', 1))
        header.extend(struct.pack('<H', channels))
        header.extend(struct.pack('<L', sample_rate))
        header.extend(struct.pack('<L', sample_rate * channels * bits_per_sample // 8))
        header.extend(struct.pack('<H', channels * bits_per_sample // 8))
        header.extend(struct.pack('<H', bits_per_sample))
        header.extend(b'data')
        header.extend(struct.pack('<L', data_length))
        return bytes(header)

    async def process_audio(self, audio_data):
        """Process audio data and return transcription."""
        if not audio_data:
            return None
            
        wav_header = self.create_wav_header(self.rate, 1, 16, len(audio_data))
        wav_data = wav_header + audio_data
        
        try:
            async with websockets.connect(ASR_WEBSOCKET_URL) as websocket:
                await websocket.send(wav_data)
                
                response = await websocket.recv()
                response_json = json.loads(response)
                
                await websocket.send(b'')
                
                return response_json
                
        except Exception as e:
            logger.error(f"Error in ASR websocket communication: {e}")
            return {"error": str(e)}

# Active client connections
clients = set()
# Speech processor instance
speech_processor = SpeechProcessor()

async def register(websocket):
    """Register a new client."""
    clients.add(websocket)
    logger.info(f"New client connected. Total clients: {len(clients)}")

async def unregister(websocket):
    """Unregister a client."""
    clients.remove(websocket)
    logger.info(f"Client disconnected. Total clients: {len(clients)}")

async def handle_client(websocket):
    """Handle WebSocket connection with a client."""
    await register(websocket)
    
    frames_buffer = b''
    speech_detected = False
    
    try:
        async for message in websocket:
            try:
                # Check if the message is JSON
                if message.startswith('{'):
                    data = json.loads(message)
                    
                    # Handle control messages
                    if data.get('type') == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                        continue
                        
                    # Reset session if requested
                    if data.get('type') == 'reset':
                        frames_buffer = b''
                        speech_detected = False
                        await websocket.send(json.dumps({'type': 'status', 'message': 'Session reset'}))
                        continue
                
                # Handle binary audio data (base64 encoded)
                elif message.startswith('data:audio'):
                    # Extract base64 audio data
                    audio_base64 = message.split(',')[1]
                    audio_data = base64.b64decode(audio_base64)
                    
                    # Check for speech
                    is_speech = speech_processor.is_speech(audio_data)
                    
                    if is_speech:
                        if not speech_detected:
                            speech_detected = True
                            await websocket.send(json.dumps({
                                'type': 'status',
                                'message': 'Speech detected'
                            }))
                        frames_buffer += audio_data
                    else:
                        if speech_detected and len(frames_buffer) > speech_processor.chunk:
                            # Process the speech segment
                            await websocket.send(json.dumps({
                                'type': 'status',
                                'message': 'Processing speech...'
                            }))
                            
                            result = await speech_processor.process_audio(frames_buffer)
                            
                            if result and 'text' in result:
                                await websocket.send(json.dumps({
                                    'type': 'transcription',
                                    'text': result['text']
                                }))
                            elif result and 'error' in result:
                                await websocket.send(json.dumps({
                                    'type': 'error',
                                    'message': result['error']
                                }))
                            
                            frames_buffer = b''
                            speech_detected = False
                
                # End of stream signal
                elif message == 'EOS':
                    if frames_buffer:
                        result = await speech_processor.process_audio(frames_buffer)
                        if result and 'text' in result:
                            await websocket.send(json.dumps({
                                'type': 'transcription',
                                'text': result['text']
                            }))
                        frames_buffer = b''
                        speech_detected = False
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send(json.dumps({
                    'type': 'error',
                    'message': str(e)
                }))
                
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client connection closed")
    finally:
        await unregister(websocket)

async def main():
    """Start the WebSocket server."""
    host = "0.0.0.0"  # Listen on all available interfaces
    port = 8765
    
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    async with websockets.serve(handle_client, host, port):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")