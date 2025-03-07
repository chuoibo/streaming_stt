import asyncio
import websockets
import json
import time

async def send_audio_file():
    async with websockets.connect('url') as websocket:
        with open('1424.wav', 'rb') as f:
            audio_data = f.read()

        for i in range(0, len(audio_data), 10240):
            st = time.time()
            await websocket.send(audio_data[i:i+10240])

            # Receive and print the response (could be partial transcription)
            try:
                response = await websocket.recv()
                response = json.loads(response)
                print(response)
            except json.JSONDecodeError:
                print("Received invalid JSON from server")
            
            print('Chunk processing time =', time.time() - st)

        # Wait for final transcript
        try:
            final_response = await websocket.recv()
            print("Final response:", json.loads(final_response))
        except json.JSONDecodeError:
            print("Received invalid JSON for final response")

# Run the async function properly
asyncio.run(send_audio_file())