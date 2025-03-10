import pyaudio
import webrtcvad
import struct
import asyncio
import websockets
import json
from concurrent.futures import ThreadPoolExecutor

class VADWebsocketProcessor:
    def __init__(self, device_name='default', websocket_url='wss://asr.gpu.rdhasaki.com/se'):
        self.device_name = device_name
        self.websocket_url = websocket_url
        self.rate = 16000
        self.frame_duration = 30  
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16  
        self.chunk = int(self.rate * self.frame_duration / 1000)  
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        
    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]
        return None

    @staticmethod
    def list_microphones(pyaudio_instance):
        info = pyaudio_instance.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        result = []
        for i in range(0, numdevices):
            if pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
                name = pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('name')
                result += [[i, name]]
        return result

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

    async def process_audio_segment(self, audio_data):
        if not audio_data:
            return
            
        wav_header = self.create_wav_header(self.rate, 1, 16, len(audio_data))
        wav_data = wav_header + audio_data
        
        try:
            async with websockets.connect(self.websocket_url) as websocket:
                await websocket.send(wav_data)
                
                response = await websocket.recv()
                response_json = json.loads(response)
                
                if "text" in response_json:
                    print(f"Speech recognized: {response_json['text']}")
                else:
                    print("Server response:", response_json)
                
                await websocket.send(b'')
                
        except Exception as e:
            print(f"Error in websocket communication: {e}")

    def process_audio_frames(self, frames):
        if frames:
            asyncio.run_coroutine_threadsafe(self.process_audio_segment(frames), self.loop)

    def start_event_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def process_stream(self):
        self.executor.submit(self.start_event_loop)
        
        microphones = self.list_microphones(self.audio)
        selected_input_device_id = self.get_input_device_id(self.device_name, microphones)

        if selected_input_device_id is None:
            print("No matching input device found. Using default.")
            selected_input_device_id = None

        stream = self.audio.open(input_device_index=selected_input_device_id,
                                 format=self.format,
                                 channels=1,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk)

        frames = b''
        speech_detected = False

        print("Starting real-time audio processing...")
        print("Speak into the microphone for real-time speech recognition...")
        
        try:
            while True:
                frame = stream.read(self.chunk, exception_on_overflow=False)
                is_speech = self.vad.is_speech(frame, self.rate)

                if is_speech:
                    if not speech_detected:
                        print("Speech detected, recording...")
                        speech_detected = True
                    frames += frame
                else:
                    if speech_detected and len(frames) > self.chunk:
                        print('Silence detected, processing speech segment')
                        self.process_audio_frames(frames)
                        frames = b''  
                        speech_detected = False

        except KeyboardInterrupt:
            if frames and speech_detected:
                print("Processing final segment before exit...")
                self.process_audio_frames(frames)
            print("Stopping audio processing...")
        finally:
            stream.stop_stream()
            stream.close()
            self.audio.terminate()
            
            # Clean up asyncio loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.executor.shutdown(wait=False)

def run_vad_processor():
    try:
        processor = VADWebsocketProcessor(device_name='default', websocket_url='wss://asr.gpu.rdhasaki.com/se')
        processor.process_stream()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Exiting VAD processor")

if __name__ == "__main__":
    try:
        run_vad_processor()
    except Exception as e:
        print(f"An error occurred: {e}")