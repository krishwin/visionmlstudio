import ray
import os
import pandas as pd
from pyarrow import fs
from torchvision import transforms
from typing import Any, Dict
import numpy as np
import json
import random
from models.RepNet.inference import RepNetEngine

ray.init(ignore_reinit_error=True)

access_key = os.getenv('access_key')
secret_key = os.getenv('secret_key')
region = os.getenv('region')
namespace = os.getenv('namespace')

s3fs = fs.S3FileSystem(
    access_key=access_key,
    secret_key=secret_key,
    region=region,
    endpoint_override=f'https://{namespace}.compat.objectstorage.{region}.oraclecloud.com',
    scheme='https'
)
selector = fs.FileSelector('yt-vtt/converted_videos/',recursive=True)
labels = ["Barbell Jammers", "Bird-dog", "Bodyweight Squat", "BOSU® Lateral Jumps", "BOSU® Squat Jumps", "Childs Pose", "Cobra Exercise", "Crunch", "Decline Plank", "Dirty Dog", "Forward Lunge", "Forward Lunge with Arm Drivers", "Forward Stepping over Cones ", "Front Plank", "Glute Activation Lunges", "Glute Bridge Exercise", "Glute Bridge ", "Half-kneeling Hay Baler", "Hexagon Drill", "High Plank T-spine Rotation", "Hip Rotations (Push-up Position)", "Inchworms", "Inverted Flyers", "Kneeling ABC's", "Lateral Over Unders", "Lunge with Elbow Instep", "Lunge with Overhead Press", "Medicine Ball Lunge to Chest Pass", "Medicine Ball Push-ups", "Partner Assisted Bodyweight Squats", "Partner Tricep Extension", "Plank-ups", "Prone ABC's", "Pull-over Crunch", "Quadruped Bent-knee Hip Extensions", "Reverse Ab Crunch", "Reverse Lunge with Rotation", "Roll Out", "Russian Twist", "Seated Crunch", "Seated Medicine Ball Trunk Rotations", "Seated Side-Straddle Stretch ", "Seated Straddle with Side Reaches ", "Side Plank", "Side Plank (Modified)", "Side Plank - modified", "Side Plank with Straight Leg", "Side-lying Arm Rolls", "Single Arm Plank", "Single Leg Stand", "Single-arm Medicine Ball Push-up", "Single-leg Stand with Reaches", "Squat to Overhead Raise", "Stability Ball Knee Tucks", "Stability Ball Pikes", "Stability Ball Prone Walkout", "Stability Ball Sit-ups / Crunches", "Standing Anti-rotation Press", "Standing Crunch", "Standing Gate Openers (Frankensteins)", "Standing Hamstrings Curl", "Standing Hay Baler", "Standing Hip Abduction", "Standing Hip Adduction", "Standing Leg Extension", "Standing Single-leg Cable Rotation", "Standing Trunk Rotation", "Standing Wood Chop", "Supine Bicycle Crunches", "Supine Dead Bug", "Supine Hollowing with Lower Extremity Movements", "Supine Pelvic Tilts", "Supine Reverse Marches", "Supine Rotator Cuff ", "Supine Shoulder Roll", "Supine Snow Angel (Wipers) Exercise", "Supine Spinal Twist with Rib Grab and Progressions", "Supine Suitcase Pass", "TRX ® Atomic Push-up", "TRX ® Back Row", "TRX ® Biceps Curl", "TRX ® Chest Press", "TRX ® Front Rollout", "TRX ® Single-arm Chest Press", "TRX ® Single-arm Row", "TRX ® Suspended Knee Tucks", "TRX ® Suspended Lunge", "TRX ® Suspended Pike", "Upward Facing Dog", "V Sit Partner Rotations", "V-twist", "V-ups", "Vertical Toe Touches", "Walking Lunges with Twists", "Bent Knee Push-up", "Bent-over Row", "Bicep Curl", "Chest Press", "Chin-ups ", "Close-grip Bench Press", "Downward-facing Dog", "Hammer Curl", "Lying Barbell Triceps Extensions ", "Overhead Triceps Stretch", "Partner Standing Row with Resistance Tubing ", "Power Push Down", "Pull-ups", "Push-up", "Push-up with Single-leg Raise", "Push-up with Staggered Hands", "Reverse Bicep Curl", "Rotational Uppercut", "Seated Bent-Knee Biceps Stretch", "Seated Biceps Curl", "Seated Close-Grip Chest Press", "Seated High Back Row ", "Seated High Back Rows", "Seated Machine Close-Grip Shoulder Press", "Seated Overhead Press", "Seated Row ", "Seated Shoulder Press", "Seated Shoulder Press ", "Spider Walks", "Standing Bicep Curl", "Standing Shoulder Extension", "Standing Shoulder Press", "Tricep Extension", "Tricep Pressdown Exercise", "Triceps Extension", "Triceps Kickback", "Triceps Pressdown", "Triceps Pushdowns ", "TRX ® Overhead Triceps Extension", "TRX ® Suspended Push-up", "Wrist Curl - Extension", "Wrist Curl - Flexion", "Wrist Supination  Pronation Exercises", "90 Lat Stretch", "Cat-Cow", "Contralateral Limb Raises", "Front Squat", "Glute Bridge Single Leg Progression", "Halo", "High Row", "Incline Reverse Fly", "Kneeling Lat Pulldown", "Kneeling Lat Stretch (w/bench)", "Kneeling Reverse Fly", "Overhead Slams", "Prone Scapular (Shoulder) Stabilization Series - I, Y, T, W, O Formation", "Reverse Fly", "Romanian Deadlift", "Seated Lat Pulldown", "Seated Straddle Stretch ", "Shoulder Packing", "Shrug", "Single Arm Overhead Squat", "Single Arm Row", "Single-arm Row", "Single-arm, Single-leg Romanian Dead Lift ", "Spinal Twist with a Push-Pull Movement", "Stability Ball Reverse Extensions", "Stability Ball Shoulder Stabilization", "Standing Row", "Standing Shrug", "Standing Triangle Straddle Bends", "Straight Arm Pressdown", "Supermans", "TRX ® Assisted Cross-over Lunge with Arm Raise", "TRX ® Assisted Side Lunge with Arm Raise", "TRX ® Hip Press", "TRX ® Side-straddle Golf Swings ", "Agility Ladder: Lateral Shuffle", "Alternate Leg Push-off ", "Back Squat", "Box Jumps", "Bulgarian Split Squat", "Cycled Split-Squat Jump", "Elevated Glute Bridge", "Forward Cone Jumps", "Forward Hurdle Run", "Forward Linear Jumps", "Glute Press", "Goblet Squat", "Hip Bridge", "Hip Hinge", "Jump and Reach", "Kneeling Hip-flexor Stretch ", "Kneeling TA Stretch", "Lateral Cone Jumps", "Lateral Lunge", "Lateral Shuffles", "Leg Crossover Stretch ", "Lunge", "Lying Hamstrings Curl", "Mountain Climbers", "Overhead Medicine Ball Throws", "Pistol Squat Workout", "Plank with Knee Drag", "Reverse Lunge", "Reverse Slam", "Seated Leg Press Exercise", "Side Lunge", "Side Lying Hip Abduction", "Side Lying Hip Adduction", "Single Leg Hamstring Curl", "Single Leg Push-off", "Single Leg Romanian Dead Lift", "Single Leg Squat", "Single-leg Romanian Deadlift", "Squat", "Squat Jump", "Squat Jumps", "Stability Ball Hamstring Curl", "Stability Ball Wall Squats", "Standing Lunge Stretch", "Step-up", "Sumo Rotational Squats", "Supine 90-90 Hip Rotator Stretch", "Supine Hip Flexor Stretch", "Thomas Stretch", "Transverse Lunge", "TRX ® Hamstrings Curl ", "Tuck Jump", "Walking Abduction", "Warrior I", "Bottom-up Press", "CKC Parascapular Exercises", "Incline Chest Press", "Lying Chest Fly", "Lying Pullovers", "Offset Single-arm Chest Press", "Seated Cable Press", "Seated Chest Press  ", "Seated Decline Cable Press ", "Seated Incline Cable Press", "Single-arm Chest Press", "Single-arm Rotational Press", "Stability Ball Push-Up", "Standing Chest Fly", "Standing Chest Stretch", "Standing Decline Cable Flyes", "Standing Incline Cable Flyes", "Anti-rotation Reverse Lunge", "Asynchronous Waves", "Bear Crawl Exercise", "Burpee", "Clean and Press", "Deadlift", "Double Push-press", "Double Rotation Waves", "Farmer's Carry", "Figure Eight", "Forward Linear Ladder Drill", "Front Squat to Overhead Press", "Half Turkish Get-up", "Hang Clean", "High Windmill", "High-low Partner See-saw", "Kneeling Hay Baler", "Kneeling Wood Chop", "Lateral Crawls", "Lateral Hurdle Run", "Lateral Lunge Wood Chop", "Lateral Waves", "Low Windmill", "Lunge to Single Arm Row", "Multidirectional Ladder Drill", "Power Clean", "Prone Runner", "Pull to Press", "Push Jerk", "Push Press", "Push-jerk", "Renegade Row", "Rotational Slam", "Side Plank Row", "Simultaneous Wave with Reverse Lunges", "Simultaneous Waves", "Single Arm Overhead Press", "Single Arm Swing", "Single Arm, Single Leg Plank", "Snatch", "Sprinter Pulls", "Squat to Overhead Press", "Squat to Row", "Standing Rotational Chop", "Suitcase Carry", "Swing", "T Drill ", "Toe Taps", "Transverse Lunge to Single Arm Row", "Turkish Get-up", "Waiter's Carry", "Ankle Flexion ", "Calf Raise", "Calf Raises", "Lateral Zig Zags", "Seated Calf Stretch", "Seated Toe Touches ", "Standing Ankle Mobilization", "Standing Calf Raises - Wall", "Standing Dorsi-Flexion (Calf Stretch)", "Step Stretch", "Supine Hamstrings Stretch", "Lateral Neck Flexion", "Neck Flexion and Extension", "Diagonal Raise", "Front Raise", "Lateral Raise", "Lateral Shoulder Raise", "Rotational Overhead Press", "Rotator Cuff External Rotation", "Rotator Cuff Internal Rotation", "Shoulder Stability-Mobility Series - I, Y, T, W Formations ", "Single-arm Lateral Raise", "Supine Shoulder Flexion ", "V-raise", "Modified Hurdler's Stretch", "Prone (Lying) Hamstrings Curl", "Seated Butterfly Stretch ", "Seated Leg Extension ", "Side Lying Quadriceps Stretch ", "Supine IT Band Stretch"]


def transform_frames(row: Dict[str, Any]) -> Dict[str, Any]:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=0.5, std=0.5),
        #transforms.ToTensor()
    ])
    row["frame"] = transform(row["frame"])
    return row

def get_files():
    files_info = s3fs.get_file_info(selector)
    files_info = [file for file in files_info if file.path.endswith(('.mp4', '.mkv'))]
    files_info.sort(key=lambda file: file.size)
    data = [file.path for file in files_info]
    return data

def get_clips_info(file):
    with s3fs.open_input_stream(f'yt-vtt/tsimport/{file.split("/")[-1].split(".")[0]}.json') as s3file:
        data = json.loads(s3file.read())
    print(data)
    return data

def get_clips(row: Dict[str, Any]) -> Dict[str, Any]:
    clips = []
    s3_file_path = f'yt-vtt/tsimport/{row["item"].split("/")[-1].split(".")[0]}.json'
    if s3fs.get_file_info(s3_file_path).type != fs.FileType.NotFound:
        with s3fs.open_input_stream(s3_file_path) as s3file:
            data = json.loads(s3file.read())
        for result in data["annotations"][0]["result"]:
            clips.append([result["value"]["ranges"][0]["start"], result["value"]["ranges"][0]["end"]])
        row["clips"] = clips
    else:
        row["clips"] = []
    return row

files = ['yt-vtt/converted_videos/converted_-cDGcc8Wc70.mp4']#get_files()
ds = ray.data.from_items(files)
clips = ds.map(get_clips).take_all()
print(len(clips))
clips = [clip for clip in clips if clip["clips"]]
print(len(clips))
ds  = ray.data.from_items(clips)
predictions = ds.map_batches(RepNetEngine, concurrency=3, batch_size=1).take_all()
print(predictions)
for index,row in enumerate(predictions):
    task_data = {"data": {"video": ""}, "annotations": [{"result": []}]}
    task_data["data"]["video"] = f's3://{row["item"]}'
    #clips = get_clips_info()
    for idx, results in enumerate(row["scenes"]):
        print(results)
        task_data["annotations"][0]["result"].append({"type": "timelinelabels", "value": {"ranges": [{"start": int(results[0]), "end": int(results[1]),}], "timelinelabels": [int(results[2])]}, "to_name": "video", "from_name": "TimelineLabel"})
    with s3fs.open_output_stream(f'yt-vtt/RepNetImport/{row["item"].split("/")[-1].split(".")[0]}.json') as s3file:
        s3file.write(json.dumps(task_data).encode())
#for file in files:
#    task_data={"data": {"video": ""}, "annotations": [{"result": []}]}
#    task_data["data"]["video"] = f's3://{file}'
#    print(file)
#    ds = ray.data.read_videos(f's3://{file}', filesystem=s3fs,)
#    ds = ds.map(transform_frames)
#    imgarray = [np.stack([row["frame"] for row in ds.take_all()], axis=0)]
#    clips = get_clips_info()
#    ds1 = ray.data.from_numpy([np.stack(imgarray, axis=0).reshape(-1, 3, 112, 112),[clips]])
#    predictions =ds1.map_batches(RepNetEngine,concurrency=1,batch_size=ds1.count()).take_all()
#    print(predictions)
#    for idx,results in enumerate(predictions):
#            print(results)
#            task_data["annotations"][0]["result"].append({"type": "timelinelabels","value": {"ranges": [{"start": int(results['scenes'][0]),"end": int(results['scenes'][1]),}],"timelinelabels": [labels[random.randint(0, len(labels)-1)],]},"to_name": "video","from_name": "TimelineLabel"})
#    with s3fs.open_output_stream(f'yt-vtt/tsimport/{file.split("/")[-1].split(".")[0]}.json') as s3file:
#        s3file.write(json.dumps(task_data).encode())    
