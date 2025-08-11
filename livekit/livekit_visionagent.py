import asyncio
import io
import logging
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (AutoSubscribe, JobContext, WorkerOptions, cli,Agent,AgentSession,JobProcess,
    RoomInputOptions,
    RoomOutputOptions,
    RunContext,metrics)
from PIL import Image
import numpy as np
from livekit.agents.voice import MetricsCollectedEvent
from livekit.plugins import google
from collections import deque
import os 
import sys
import websockets
import base64
import json
import time
import cv2

load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

WIDTH = 640
HEIGHT = 480

# Per-client state
class ClientSession:
    def __init__(self):
        self.short_buffer = deque(maxlen=32)
        self.counter = 0

    def add_frame(self, frame):
        self.short_buffer.append(frame)
        self.counter += 1
        
class VoiceActivityVideoSampler:
    def __init__(self, *, speaking_fps: float = 1.0, silent_fps: float = 0.3):
        if speaking_fps <= 0 or silent_fps <= 0:
            raise ValueError("FPS values must be greater than zero")

        self.speaking_fps = speaking_fps
        self.silent_fps = silent_fps
        self._last_sampled_time: float | None = None

    def __call__(self, frame: rtc.VideoFrame, session: AgentSession) -> bool:
        now = time.time()
        is_speaking = session.user_state == "speaking"
        target_fps = self.speaking_fps if is_speaking else self.silent_fps
        min_frame_interval = 1.0 / target_fps

        if self._last_sampled_time is None:
            self._last_sampled_time = now
            return True

        if (now - self._last_sampled_time) >= min_frame_interval:
            self._last_sampled_time = now
            return True
        print(f"Skipping frame at {now}, last sampled at {self._last_sampled_time}, target FPS: {target_fps}")
        return False

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="you are personal fitness coach,  You will also provide encouragement and motivation.politely ask the user to switch on their camera so that you can watch their workout. You can identify the type of workout, count reps, and provide motivation.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        
        self.session.generate_reply(instructions ="Hello, I am your personal fitness coach. switch on your camera so that i can watch your workour . i can identify your workout,count reps and provide motivation?")
 
class VideoProcessing():
    
    def __init__(self):
        self.session = ClientSession()
        pass
    async def send_frames(self,frames):
            # Encode each frame as JPEG and then base64
            encoded_frames = []
            for frame in frames:
                h, w, c = frame.shape
                print(f"Encoding frame of shape {frame.shape}")
                encoded = base64.b64encode(frame).decode('utf-8')
                encoded_frames.append({'frame':encoded,'height': h,
                    'width': w,
                    'channels': c})
            async with websockets.connect(os.getenv("RAY_SERVE")) as websocket:
                response = None
                for encoded in encoded_frames:
                    print(f"Sending frame to server: {encoded['frame'][:30]}...")  # Print first 30 chars for brevity
                    await websocket.send(  json.dumps({'frame':encoded['frame'],
                                                       'height': encoded['height'],
                    'width': encoded['width'],
                    'channels': encoded['channels']
                    }))
                    response = await websocket.recv()
                    print(f"Received response: {response}")
                return response

    async def do_something(self,track: rtc.RemoteVideoTrack, agentsession: AgentSession):
        video_stream = rtc.VideoStream(track)
        async with websockets.connect(os.getenv("RAY_SERVE")) as websocket:

            async for event in video_stream:
                # Do something here to process event.frame
                
                #print(f"Received frame with timestamp {event.timestamp_us} and type {event.frame.type}")
                frame=event.frame.convert( rtc.VideoBufferType.RGBA)
                #print(f"Frame format: {frame.type}, width: {frame.width}, height: {frame.height}")
                arr = np.asarray(frame.data, dtype=np.uint8)
                #print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
                arr=arr.reshape((frame.height, frame.width, 4))
                arr = arr[:, :, :3]  # Convert RGBA to RGB
                arr =cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
                self.session.add_frame(arr)
                result = {}
                if len(self.session.short_buffer) == 32 and self.session.counter % 32 == 0:
                            #result =   await asyncio.create_task(self.send_frames(self.session.short_buffer))
                            batch = np.stack(self.session.short_buffer, axis=0).astype(np.uint8)
                            with io.BytesIO() as buf:
                                np.save(buf, batch)
                                payload = buf.getvalue()
                                print(f"Sending batch of shape {batch.shape} to server")
                                await websocket.send(payload)
                                result = await websocket.recv()
                                result = json.loads(result)
                                print(f"Action detection result: {result}")
        
                if result:
                    print(f"Result: {result}")
                    if 'action' in result:
                        asyncio.create_task(agentsession.generate_reply(instructions =f"workout: {result['action']}"))
                    if 'count' in result:
                        asyncio.create_task(agentsession.generate_reply(instructions =f"Rep count: {result['count']}"))
            await video_stream.aclose()

async def entrypoint(ctx: JobContext):
    # an rtc.Room instance from the LiveKit Python SDK
    room = ctx.room
    logger.info(f"Connecting to room")
    
    # connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    session = AgentSession(
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=google.LLM(),
        tts=google.TTS(),
        video_sampler = VoiceActivityVideoSampler(speaking_fps=8.0, silent_fps=8)
        
        # use LiveKit's turn detection model
    )
    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # uncomment to enable Krisp BVC noise cancellation
            # noise_cancellation=noise_cancellation.BVC(),
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    # set up listeners on the room before connecting
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            videoprocessor = VideoProcessing()
            asyncio.create_task(videoprocessor.do_something(track,session))

        
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))