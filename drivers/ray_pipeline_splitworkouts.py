import pyarrow as pa
import pyarrow.fs as fs
import json
import cv2
import numpy as np
import os
from dotenv import load_dotenv
from decord import VideoReader
from decord import cpu
import ray
from typing import Dict, Any

# Load environment variables
load_dotenv()
access_key = os.getenv('access_key')
secret_key = os.getenv('secret_key')
region = os.getenv('region')
namespace = os.getenv('namespace')

# Initialize S3 filesystem with credentials
s3 = fs.S3FileSystem(region=region, access_key=access_key, secret_key=secret_key, endpoint_override=f'https://{namespace}.compat.objectstorage.{region}.oraclecloud.com',)

# Define S3 paths
input_path = 'yt-vtt/RepNetImport/'
output_path = 'yt-vtt/fitclass/'

# List JSON files in the input path
files = s3.get_file_info(fs.FileSelector(input_path))
json_files = [file.path for file in files if file.path.endswith('.json')]

def filter_existing_files(json_files, output_path):
    """
    Filters out JSON files whose corresponding output files already exist in the output path.
    """
    filtered_files = []
    out_files = s3.get_file_info(fs.FileSelector(output_path))
    
    for json_file in json_files:
            videoid = json_file.split('converted_')[1].split('.')[0]
            exists = [file.path for file in out_files if videoid in file.path]
            if(len(exists) == 0):
                filtered_files.append(json_file)
    return filtered_files

# Filter json_files to exclude already processed files
json_files = filter_existing_files(json_files, output_path)

def process_batch(batch:Dict[str, Any]):
            print(batch['video_path'])
            ranges = batch['value'][0]['ranges'][0]
            print(ranges)
            labels = batch['value'][0]['timelinelabels']
            start_frame = ranges['start']
            end_frame = ranges['end']
            vr = batch['video'][0]
            print(vr.shape)
            video_path = batch['video_path'][0]
            fps = batch['fps'][0]
            print(f"Processing video: {video_path}, start: {start_frame}, end: {end_frame}")
            # Create new video segment
            output_file_name = f"{os.path.basename(video_path).split('.')[0]}_{start_frame}_{end_frame}.mp4"
            output_file_path = os.path.join(output_path, output_file_name)

            # Initialize VideoWriter
            frame_width, frame_height = vr[0].shape[1], vr[0].shape[0]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_file = f"/tmp/{output_file_name}"
            vw = cv2.VideoWriter(temp_file, fourcc, fps, (frame_width, frame_height))

            for frame in vr:
                img = frame
                vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            vw.release()
             # Convert the video to H.264 format using ffmpeg
            h264_temp_file = f"/tmp/h264_{output_file_name}"
            os.system(f"ffmpeg -i {temp_file} -c:v libx264 -preset fast -crf 23 -y {h264_temp_file}")

            # Replace the temp_file with the H.264 converted file
            os.remove(temp_file)
            temp_file = h264_temp_file
            
            # Upload video to S3
            with open(temp_file, 'rb') as temp_video:
                with s3.open_output_stream(output_file_path) as output_stream:
                    output_stream.write(temp_video.read())

            os.remove(temp_file)
            print(f"Uploaded video: {output_file_path}")

            # Create JSON metadata
            json_metadata = {
                "data": {
                    "video": f's3://{output_file_path}'
                },
                "annotations": [
                    {
                        "result": [
                            {
                                "type": "number",
                                "value": {
                                    "number": int(labels[0])
                                },
                                "to_name": "video",
                                "from_name": "rep"
                            }
                        ]
                    }
                ]
            }

            # Upload JSON metadata to S3
            json_file_name = f"{os.path.basename(output_file_name).split('.')[0]}.json"
            json_file_path = os.path.join(output_path, json_file_name)
            with s3.open_output_stream(json_file_path) as json_stream:
                json_stream.write(json.dumps(json_metadata).encode('utf-8'))

            print(f"Uploaded JSON: {json_file_path}")
            return batch
            

#@ray.remote
def process_file(json_file):
    # Read JSON file
    print(f"Processing file: {json_file}")
    with s3.open_input_file(json_file) as f:
        data = json.load(f)

    # Extract video path
    video_path = data['data']['video'].split('s3://')[1]

    # Read video using decord
    with s3.open_input_file(video_path) as video_file:
        vr = VideoReader(video_file, ctx=cpu(0))
        fps = vr.get_avg_fps()
        
    # Process annotation results using ray.data with map_batches
    results = ray.data.from_items([
        {
            **result,
            "video_path": video_path,
            "fps": fps,
            "video": np.array([vr[frame].asnumpy() for frame in range(result['value']['ranges'][0]['start'], result['value']['ranges'][0]['end'] + 1)])
        }
        for annotation in data['annotations']
        for result in annotation['result']
        if int(result['value']['timelinelabels'][0]) > 2
    ])
    #results.filter(lambda x:  int(x['value']['timelinelabels'][0]) > 2,concurrency=1,num_cpus=1)#.
    results.map_batches(process_batch,concurrency=1,num_cpus=1,batch_size=1).take_all()
    del results
    del vr
    return
# Initialize Ray
ray.init(ignore_reinit_error=True)

# Process files in parallel
#futures = [process_file.options().remote(json_file) for json_file in json_files]
#ray.get(futures)
for json_file in json_files:
    process_file(json_file)

# Shutdown Ray
ray.shutdown()