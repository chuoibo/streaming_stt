import asyncio
import websockets
import pyaudio
import webrtcvad
import collections
import time
import json
import os

from dotenv import load_dotenv

load_dotenv()

WS = os.getenv('WS')

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  
SILENCE_THRESHOLD = 3.0  
CHUNK_DURATION_MS = 30  
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)  


def get_input_device_id(device_name, microphones):
    for device in microphones:
        if device_name.lower() in device[1].lower():
            print(f"Selected device: {device[1]} (Index: {device[0]})")
            return device[0]
    if microphones:
        print(f"'{device_name}' not found. Using first available: {microphones[0][1]}")
        return microphones[0][0]
    return None


def list_microphones(pyaudio_instance):
    info = pyaudio_instance.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')

    result = []
    for i in range(0, numdevices):
        if (pyaudio_instance.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            name = pyaudio_instance.get_device_info_by_host_api_device_index(
                0, i).get('name')
            result += [[i, name]]
    return result


async def process_audio():
    audio = pyaudio.PyAudio()

    microphones = list_microphones(audio)

    selected_input_device_id = get_input_device_id(device_name='ATR4697-USB: USB Audio (hw:2,0)', microphones=microphones)

    vad = webrtcvad.Vad(3)  
    stream = audio.open(
        input_device_index=selected_input_device_id,
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
        )
        
    audio_buffer = collections.deque(maxlen=100)  
    results = []
    silent_frames = 0
    is_speaking = False
    
    print("Starting real-time voice detection... Speak now!")
    
    async with websockets.connect(WS) as websocket:
        try:
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                print(f"Data type: {type(data)}, Length: {len(data)} bytes")

                is_speech = vad.is_speech(data, RATE)
                
                if is_speech:
                    silent_frames = 0
                    is_speaking = True
                    audio_buffer.append(data)
                    
                    st = time.time()
                    await websocket.send(data)
                    
                    response = await websocket.recv()
                    response = json.loads(response)
                    print(f"Partial result: {response}")
                    print(f"Time: {time.time() - st:.3f}s")
                    
                    if response.get('text'): 
                        results.append(response['text'])
                
                elif is_speaking and not is_speech:
                    silent_frames += 1
                    audio_buffer.append(data)
                    
                    silence_duration = silent_frames * (CHUNK / RATE)
                    if silence_duration >= SILENCE_THRESHOLD:
                        print("Silence detected for 3 seconds, ending...")
                        break
                
                else:
                    audio_buffer.append(data)
        
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await websocket.send(b'')
            await websocket.close()
            
            stream.stop_stream()
            stream.close()
            audio.terminate()
            
            final_result = " ".join(results).strip()
            print(f"Final transcription: {final_result}")
            return final_result

def run_realtime_speech_recognition():
    try:
        result = asyncio.get_event_loop().run_until_complete(process_audio())
        return result
    except KeyboardInterrupt:
        print("\nStopped by user")
        return None
    except Exception as e:
        print(f"Error in execution: {e}")
        return None

if __name__ == "__main__":
    final_text = run_realtime_speech_recognition()