import asyncio
import websockets
import webrtcvad
import json
import struct
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('asr_server')

# ASR service WebSocket URL
ASR_SERVICE_URL = 'wss://asr.gpu.rdhasaki.com/se'

class ASRWebSocketServer:
    def __init__(self, host='localhost', port=5000):
        self.host = host
        self.port = port
        self.sample_rate = 16000
        self.frame_duration = 30  # ms
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        self.clients = {}  # Store client state

    @staticmethod
    def create_wav_header(sample_rate, channels, bits_per_sample, data_length):
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

    def convert_buffer_size(self, audio_frame):
        """Ensure the audio frame has the right size for VAD"""
        if len(audio_frame) < self.frame_size * 2:  # 2 bytes per sample (16-bit)
            audio_frame = audio_frame + b'\x00' * (self.frame_size * 2 - len(audio_frame))
        return audio_frame[:self.frame_size * 2]

    async def process_audio_segment(self, audio_data):
        """Send audio to ASR service and get transcription"""
        if not audio_data:
            return None

        wav_header = self.create_wav_header(self.sample_rate, 1, 16, len(audio_data))
        wav_data = wav_header + audio_data

        try:
            async with websockets.connect(ASR_SERVICE_URL) as websocket:
                await websocket.send(wav_data)
                response = await websocket.recv()
                
                try:
                    response_json = json.loads(response)
                    if "text" in response_json:
                        logger.info(f"Speech recognized: {response_json['text']}")
                        return response_json
                    else:
                        logger.warning(f"Unexpected response format: {response_json}")
                        return None
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: {response}")
                    return None
                
                finally:
                    # Send empty message to close the connection properly
                    await websocket.send(b'')
                
        except Exception as e:
            logger.error(f"Error in ASR service communication: {e}")
            return None

    async def handle_client(self, websocket):
        """Handle WebSocket client connection"""
        client_id = id(websocket)
        
        # Initialize client state
        self.clients[client_id] = {
            'audio_buffer': bytearray(),
            'speech_detected': False,
            'silent_frames': 0,
            'max_silent_frames': 10  # About 300ms of silence (10 * 30ms)
        }
        
        logger.info(f"New client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self.process_frame(websocket, client_id, message)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error occurred with client {client_id}: {e}")
        finally:
            # Clean up client state
            if client_id in self.clients:
                # Process any remaining audio before removing client
                if self.clients[client_id]['audio_buffer'] and self.clients[client_id]['speech_detected']:
                    await self.finalize_audio_segment(websocket, client_id)
                del self.clients[client_id]
            
            logger.info(f"Client {client_id} cleanup complete")

    async def process_frame(self, websocket, client_id, audio_frame):
        """Process incoming audio frame"""
        client = self.clients[client_id]
        
        # Ensure frame is the right size for VAD
        frame_for_vad = self.convert_buffer_size(audio_frame)
        
        try:
            # Check if this frame contains speech
            is_speech = self.vad.is_speech(frame_for_vad, self.sample_rate)
            
            if is_speech:
                # Reset silent frame counter when speech is detected
                client['silent_frames'] = 0
                
                if not client['speech_detected']:
                    logger.debug(f"Speech started for client {client_id}")
                    client['speech_detected'] = True
                
                # Add the frame to the buffer
                client['audio_buffer'].extend(audio_frame)
                
            else:  # Not speech
                if client['speech_detected']:
                    client['audio_buffer'].extend(audio_frame)
                    
                    client['silent_frames'] += 1
                    
                    if client['silent_frames'] >= client['max_silent_frames']:
                        await self.finalize_audio_segment(websocket, client_id)
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")

    async def finalize_audio_segment(self, websocket, client_id):
        """Process the collected audio segment and send results back to client"""
        client = self.clients[client_id]
        
        if len(client['audio_buffer']) > 0 and client['speech_detected']:
            logger.info(f"Processing audio segment of {len(client['audio_buffer'])} bytes")
            
            # Process with ASR
            result = await self.process_audio_segment(bytes(client['audio_buffer']))
            
            # Reset client state
            client['audio_buffer'] = bytearray()
            client['speech_detected'] = False
            client['silent_frames'] = 0
            
            # Send result back to client
            if result and "text" in result:
                await websocket.send(json.dumps(result))
                logger.info(f"Sent transcription to client {client_id}")

    async def start_server(self):
        """Start the WebSocket server"""
        server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        logger.info(f"ASR WebSocket server started at ws://{self.host}:{self.port}")
        
        # Keep the server running
        await server.wait_closed()

def main():
    server = ASRWebSocketServer()
    
    try:
        asyncio.run(server.start_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    main()