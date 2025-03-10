import asyncio
import websockets
import json
import time

all_time = time.time()
async def send_audio_file():
    async with websockets.connect('wss://asr.gpu.rdhasaki.com/se') as websocket:
        
        with open('recording.wav', 'rb') as f:
            audio_data = f.read()

            await websocket.send(audio_data)

            response = await websocket.recv()
            response = json.loads(response)
            print(response)

        await websocket.send(b'')
        await websocket.close()

asyncio.get_event_loop().run_until_complete(send_audio_file())
        
print('all time = ', time.time() - all_time)