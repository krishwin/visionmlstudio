import asyncio
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
load_dotenv()

logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)

WIDTH = 640
HEIGHT = 480

class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="you are personal fitness coach,  You will also provide encouragement and motivation.politely ask the user to switch on their camera so that you can watch their workout. You can identify the type of workout, count reps, and provide motivation.",
        )

    async def on_enter(self):
        # when the agent is added to the session, it'll generate a reply
        # according to its instructions
        
        self.session.generate_reply(instructions ="Hello, I am your personal fitness coach. switch on your camera so that i can watch your workour . i can identify your workout,count reps and provide motivation?")
        
async def do_something(track: rtc.RemoteVideoTrack,source: rtc.VideoSource):
    video_stream = rtc.VideoStream(track)
    async for event in video_stream:
        # Do something here to process event.frame
        print(f"Received frame with timestamp {event.timestamp_us} and type {event.frame.type}")
        frame=event.frame.convert( rtc.VideoBufferType.RGBA)
        print(f"Frame format: {frame.type}, width: {frame.width}, height: {frame.height}")
        arr = np.asarray(frame.data, dtype=np.uint8)
        print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
        arr=arr.reshape((frame.height, frame.width, 4))
        img = Image.fromarray(arr)
        #pass
    await video_stream.aclose()

async def entrypoint(ctx: JobContext):
    # an rtc.Room instance from the LiveKit Python SDK
    room = ctx.room
    logger.info(f"Connecting to room {ctx.worker_id}")
    source = rtc.VideoSource(WIDTH, HEIGHT)
    track = rtc.LocalVideoTrack.create_video_track("single-color", source)
    options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    # set up listeners on the room before connecting
    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(do_something(track,source))

    # connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    #publication = await room.local_participant.publish_track(track, options)
    session = AgentSession(
        # any combination of STT, LLM, TTS, or realtime API can be used
        llm=google.LLM(),
        tts=google.TTS(),
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
        
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))