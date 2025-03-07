import threading
import pyaudio
import webrtcvad
from websocket import create_connection

import json
import time

from queue import  Queue, Empty


class VADProcessor:
    """Handles Voice Activity Detection."""
    def __init__(self, device_name='default'):
        self.device_name = device_name
        self.channels = 1
        self.rate = 16000
        self.frame_duration = 30
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.chunk = int(self.rate * self.frame_duration / 1000)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
    

    @staticmethod
    def get_input_device_id(device_name, microphones):
        for device in microphones:
            if device_name in device[1]:
                return device[0]
    

    @staticmethod
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


    def process_stream(self, asr_input_queue):
        """Processes audio from a stream and detects speech."""
        print('Processing audio stream...')
        microphones = VADProcessor.list_microphones(self.audio)

        selected_input_device_id = VADProcessor.get_input_device_id(
            self.device_name, microphones)

        stream = self.audio.open(input_device_index=selected_input_device_id,
                                 format=self.format,
                                 channels=self.channels,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=self.chunk)
        
        frames = b''

        while True:
            if Speech2Txt.exit_event.is_set():
                break
            
            frame = stream.read(self.chunk, exception_on_overflow=False)
            is_speech = self.vad.is_speech(frame, self.rate)

            if is_speech:
                frames += frame
            else:
                if len(frames) > 1:
                    asr_input_queue.put(frames)
                frames = b''


class ASRProcessor:
    """Handles Automatic Speech Recognition."""
    def __init__(self):
        self.ws = create_connection('wss://asr.gpu.rdhasaki.com/se')
    

    def send_audio_file(self, audio_data):
        try:
            self.ws.send(audio_data)
            response = self.ws.recv()
            if response:
                response_obj = json.loads(response)
                text = response_obj.get('text', '')
                return text
            else:
                print("Warning: Empty response received from server")
                return ""
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            print(f"Raw response: {response}")
            return ""
        except Exception as e:
            print(f"Error in send_audio_file: {e}")
            return ""
                

    def process_audio(self, in_queue, out_queue):
        """Processes audio frames and performs ASR."""
        print("Processing audio for ASR...")
        while True:
            audio_frames = in_queue.get()
            if audio_frames == "close":
                break

            text = self.send_audio_file(audio_frames)

            if text:
                out_queue.put(text)
                print(f"Recognized Text: {text}")


class Speech2Txt:
    """Main Speech-to-Text Pipeline."""
    exit_event = threading.Event()

    def __init__(self):
        self.device_name = 'default'
        self.asr_output_queue = Queue()
        self.asr_input_queue = Queue()


        self.vad_processor = VADProcessor()
        self.asr_processor = ASRProcessor()


    def start(self):
        """Start the Speech-to-Text process."""
        print("Starting Speech-to-Text process...")

        self.vad_thread = threading.Thread(
            target=self.vad_processor.process_stream,
            args=(self.asr_input_queue,),
        )
        self.vad_thread.start()

        self.asr_thread = threading.Thread(
            target=self.asr_processor.process_audio,
            args=(self.asr_input_queue, self.asr_output_queue),
        )
        self.asr_thread.start()



    def stop(self):
        """Stop the Speech-to-Text process."""
        print("Stopping Speech-to-Text process...")
        Speech2Txt.exit_event.set()
        self.asr_input_queue.put("close")
        self.asr_output_queue.put(None)


        self.vad_thread.join()
        self.asr_thread.join()


    def run(self):
        """Run the pipeline."""
        start_time = time.time()
        self.start()
        final_text = ""

        try:
            while True:
                try:
                    text = self.asr_output_queue.get(timeout=3.0)
                    if text:
                        print(f"Corrected Text: {text}")
                        final_text += text + " "
                except Empty:
                    print("No more input detected. Stopping.")
                    break
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            self.stop()
        end_time = time.time()

        print(f"Final Text: {final_text} with inference time: {end_time-start_time}s")
        return final_text
    
if __name__ == "__main__":
    speech2txt = Speech2Txt()
    speech2txt.run()