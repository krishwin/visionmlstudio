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
sys.path.append('./') 
from models.timesformer.model import ActionDetectorClass
load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

WIDTH = 640
HEIGHT = 480
from concurrent.futures import ThreadPoolExecutor
inference_executor = ThreadPoolExecutor(max_workers=1)

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
        self.action_detector = ActionDetectorClass()  # Initialize the action detector
        self.prewarm()  # Prewarm the model to avoid latency on first inference
        self.reply_queue = asyncio.Queue()
        pass
    def prewarm(self):
        # Prewarm the action detector model
        dummy_frames = np.random.randint(0, 255, (8, 224, 224, 3), dtype=np.uint8)
        self.action_detector(dummy_frames)
    def process_frame(self, frame):
        arr = np.asarray(frame.data, dtype=np.uint8)
        arr = arr.reshape((frame.height, frame.width, 4))[:, :, :3]
        return cv2.resize(arr, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    async def reply_flusher(self, agentsession):
        while True:
            await asyncio.sleep(10)

            # Collect all pending messages
            messages = []
            while not self.reply_queue.empty():
                msg = await self.reply_queue.get()
                messages.append(msg)

            if messages:
                combined = "\n".join(messages)
                try:
                    await agentsession.generate_reply(instructions=combined)
                    print(f"✅ Sent combined reply:\n{combined}")
                except Exception as e:
                    print(f"⚠️ Failed to send reply: {e}")


    async def handle_inference(self, batch, agentsession):
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(inference_executor, self.action_detector, batch)

        except asyncio.TimeoutError:
            print("⚠️ Inference timed out")
            return
        except Exception as e:
            print(f"⚠️ Inference failed: {e}")
            return

        if result:
            print(f"Detected action: {result.get('action', 'Unknown')}")
            if 'action' in result:
                await self.reply_queue.put(f"workout: {result['action']}")
            if 'count' in result:
                await self.reply_queue.put(f"Rep count: {result['count']}")
    
    async def do_something(self,track: rtc.RemoteVideoTrack, agentsession: AgentSession):
            video_stream = rtc.VideoStream(track)
            
            async for event in video_stream:
                # Do something here to process event.frame
                
                #print(f"Received frame with timestamp {event.timestamp_us} and type {event.frame.type}")
                frame=event.frame.convert( rtc.VideoBufferType.RGBA)
                arr = await asyncio.to_thread(self.process_frame, frame)

                self.session.add_frame(arr)
                result = {}
                if len(self.session.short_buffer) == 32 and self.session.counter % 32 == 0:
                            #result =   await asyncio.create_task(self.send_frames(self.session.short_buffer))
                            batch = np.stack(self.session.short_buffer, axis=0).astype(np.uint8)
                            #result = await asyncio.create_task(self.action_detector(batch))
                            #print(f"Detected action: {result.get('action', 'Unknown')}")
                            asyncio.create_task(self.handle_inference(batch, agentsession)) 
        
                #if result:
                #    print(f"Result: {result}")
                #    if 'action' in result:
                #        asyncio.create_task(agentsession.generate_reply(instructions =f"workout: {result['action']}"))
                #    if 'count' in result:
                #        asyncio.create_task(agentsession.generate_reply(instructions =f"Rep count: {result['count']}"))
            await video_stream.aclose()
def prewarm_fnc(proc: JobProcess):
    videoprocessor = VideoProcessing()
    # Prewarm the model to avoid latency on first inference
    #videoprocessor.prewarm()
    proc.userdata["videoprocessor"] = videoprocessor
   
async def entrypoint(ctx: JobContext):
    # an rtc.Room instance from the LiveKit Python SDK
    room = ctx.room
    logger.info(f"Connecting to room")
    videoprocessor = ctx.proc.userdata["videoprocessor"]#VideoProcessing()

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
    asyncio.create_task(videoprocessor.reply_flusher(session))
    # set up listeners on the room before connecting
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(videoprocessor.do_something(track,session))

        
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint,prewarm_fnc=prewarm_fnc,num_idle_processes=1,initialize_process_timeout=20,job_memory_warn_mb=1024))