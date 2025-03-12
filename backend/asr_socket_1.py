import asyncio
import websockets
import webrtcvad
import json
import struct
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('asr_server')

class ASRWebSocketServer:
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.sample_rate = 16000
        self.frame_duration = 30  
        self.frame_size = int(self.sample_rate * self.frame_duration / 1000)
        self.vad = webrtcvad.Vad(2) 
        self.clients = {}
        self.executors = {}  
        self.loops = {}
        
        # Time interval for sending accumulated speech data (in seconds)
        self.processing_interval = 1.0
        
        # Silence duration threshold to reset buffer (in seconds)
        self.silence_threshold = 1.0

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
        if len(audio_frame) < self.frame_size * 2:  
            audio_frame = audio_frame + b'\x00' * (self.frame_size * 2 - len(audio_frame))
        return audio_frame[:self.frame_size * 2]

    async def process_audio_segment(self, client_id, websocket, audio_data):
        if not audio_data:
            return

        wav_header = self.create_wav_header(self.sample_rate, 1, 16, len(audio_data))
        wav_data = wav_header + audio_data

        try:
            async with websockets.connect('wss://asr.gpu.rdhasaki.com/se') as asr_websocket:
                await asr_websocket.send(wav_data)
                response = await asr_websocket.recv()
                
                try:
                    response_json = json.loads(response)
                    if "text" in response_json:
                        logger.info(f"Speech recognized for client {client_id}: {response_json['text']}")
                        await websocket.send(json.dumps(response_json))
                        logger.info(f"Sent transcription to client {client_id}")
                    else:
                        logger.warning(f"Unexpected response format: {response_json}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON response: {response}")
                
                finally:
                    await asr_websocket.send(b'')
                
        except Exception as e:
            logger.error(f"Error in ASR service communication: {e}")

    def process_audio_frames(self, client_id, websocket, frames):
        if frames and client_id in self.loops:
            asyncio.run_coroutine_threadsafe(
                self.process_audio_segment(client_id, websocket, frames), 
                self.loops[client_id]
            )

    def start_event_loop(self, client_id):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loops[client_id] = loop
        loop.run_forever()
        logger.info(f"Event loop for client {client_id} stopped")

    async def handle_client(self, websocket):
        client_id = id(websocket)
        
        self.clients[client_id] = {
            'speech_detected': False,
            'last_process_time': time.time(),
            'speech_buffer': bytearray(),
            'speech_start_time': None,
            'last_speech_time': None,
            'silence_start_time': None
        }
        
        self.executors[client_id] = ThreadPoolExecutor(max_workers=1)
        self.executors[client_id].submit(self.start_event_loop, client_id)
        
        logger.info(f"New client connected: {client_id}")
        
        try:
            async for message in websocket:
                client = self.clients[client_id]
                current_time = time.time()
                frame_for_vad = self.convert_buffer_size(message)
                
                try:
                    is_speech = self.vad.is_speech(frame_for_vad, self.sample_rate)
                    
                    if is_speech:
                        client['silence_start_time'] = None
                        
                        if not client['speech_detected']:
                            logger.debug(f"Speech started for client {client_id}")
                            client['speech_detected'] = True
                            client['speech_buffer'] = bytearray()
                            client['speech_start_time'] = current_time
                            client['last_process_time'] = current_time
                        
                        client['last_speech_time'] = current_time
                        
                        client['speech_buffer'].extend(message)
                        
                        if current_time - client['last_process_time'] >= self.processing_interval:
                            speech_duration = current_time - client['speech_start_time']
                            logger.info(f"Processing {speech_duration:.2f}s accumulated chunk of {len(client['speech_buffer'])} bytes for client {client_id}")
                            
                            self.process_audio_frames(
                                client_id, 
                                websocket, 
                                bytes(client['speech_buffer'])
                            )
                            
                            client['last_process_time'] = current_time
                    else:
                        logging.info('No speech detected ...')
                        if client['speech_detected']:
                            if client['silence_start_time'] is None:
                                client['silence_start_time'] = current_time
                            
                            if (current_time - client['silence_start_time'] >= 0.5):
                                logger.debug(f"Silence detected for {self.silence_threshold}s, resetting buffer for client {client_id}")
                                client['speech_detected'] = False
                                client['speech_buffer'] = bytearray()
                                client['speech_start_time'] = None
                                client['last_speech_time'] = None
                                client['silence_start_time'] = None
                            else:
                                client['speech_buffer'].extend(message)
                                
                        
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error occurred with client {client_id}: {e}")
        finally:
            # Process any remaining speech data before cleanup
            if client_id in self.clients and len(self.clients[client_id]['speech_buffer']) > self.frame_size:
                self.process_audio_frames(
                    client_id, 
                    websocket, 
                    bytes(self.clients[client_id]['speech_buffer'])
                )
            
            if client_id in self.clients:
                del self.clients[client_id]
            
            if client_id in self.loops:
                self.loops[client_id].call_soon_threadsafe(self.loops[client_id].stop)
            
            if client_id in self.executors:
                self.executors[client_id].shutdown(wait=False)
                del self.executors[client_id]
            
            if client_id in self.loops:
                del self.loops[client_id]
            
            logger.info(f"Client {client_id} cleanup complete")

    async def start_server(self):
        server = await websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        logger.info(f"Realtime ASR WebSocket server started at ws://{self.host}:{self.port}")
        
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