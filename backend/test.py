import asyncio
import websockets
import json
import time

url = ''

all_time = time.time()
async def send_audio_file():
    async with websockets.connect(url) as websocket:
        
        with open('1424.wav', 'rb') as f:
            audio_data = f.read()

        for i in range(0, len(audio_data), 10240):
            st = time.time()
            await websocket.send(audio_data[i:i+10240])

            response = await websocket.recv()
            # load json
            response = json.loads(response)
            print(response)
            print('time = ', time.time() - st)

        # disconnect
        await websocket.send(b'')
        await websocket.close()

asyncio.get_event_loop().run_until_complete(send_audio_file())

print('all time = ', time.time() - all_time)