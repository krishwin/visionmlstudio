from decord import VideoReader, cpu

def read_video(given_path):
    """
    Reads a video file from the given path and returns the frames as a list.

    Parameters:
        given_path (str): Path to the video file.

    Returns:
        list: A list of frames from the video.
    """
    try:
        vr = VideoReader(given_path, ctx=cpu(0))
        frames = [frame.asnumpy() for frame in vr]
        return frames
    except Exception as e:
        print(f"Error reading video file: {e}")
        return None