import concurrent.futures
import json
import os
import random
from functools import partial
import numpy as np

import cv2
from gui_agent_data.utils.image import encode_image
from PIL import Image
from tqdm import tqdm

# from moviepy.editor import VideoFileClip

basedir = "datasets/agentnet/raw"

# def get_duration(video_path):
#     clip = VideoFileClip(video_path)
#     duration = clip.duration  # returns duration in seconds
#     clip.close()
#     return duration


def get_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get total number of frames and frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate duration
    duration = total_frames / fps

    cap.release()
    return duration


def extract_frame_at_timestamp(video_path, timestamp):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Calculate relative timestamp in milliseconds
    # timestamp is in seconds
    relative_time = timestamp * 1000  # convert to milliseconds
    # check if the relative_time is extend than the video length, turn to the last frame

    # Set video position to timestamp
    cap.set(cv2.CAP_PROP_POS_MSEC, relative_time)

    # Read frame
    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        return Image.fromarray(frame_rgb)
    return None

def compute_frame_similarity(frame1, frame2):
    """Calculate pixel-wise difference between two frames using L0 norm"""
    # Convert PIL images to numpy arrays if needed
    if isinstance(frame1, Image.Image):
        frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
    if isinstance(frame2, Image.Image):
        frame2 = cv2.cvtColor(np.array(frame2), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate number of different pixels (L0 norm)
    diff = np.sum(gray1 != gray2)
    
    # Convert to similarity score (0 to 1, where 1 means identical)
    total_pixels = gray1.shape[0] * gray1.shape[1]
    similarity = 1 - (diff / total_pixels)
    
    return similarity


def find_loading_complete_time(
    video_path, 
    start_time, 
    end_time, 
    video_start_timestamp,
    sample_interval=0.1, 
    similarity_threshold=0.99
):
    """
    Find the timestamp when page loading completes by comparing frames

    Args:
        video_path: Path to the video file
        start_time: Start time of the event
        end_time: End time of the event
        video_start_timestamp: Video start timestamp for offset calculation
        sample_interval: Time interval between frame samples
        similarity_threshold: Threshold for considering frames similar

    Returns:
        Tuple of (loading_start_time, loading_end_time)
    """
    start_time_relative = start_time - video_start_timestamp
    final_time = max(start_time_relative, end_time - video_start_timestamp - 0.5)
    current_time = max(start_time_relative, final_time) # TODO: maybe delete more time
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    cap.set(cv2.CAP_PROP_POS_MSEC, final_time * 1000)
    ret, final_frame = cap.read()
    if not ret:
        cap.release()
        return None, None

    # Sample frames backward until significant change is detected
    while current_time >= start_time_relative:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        similarity = compute_frame_similarity(frame, final_frame)
        if similarity < similarity_threshold:
            # Found boundary between loading phase and completion
            cap.release()
            return current_time + sample_interval, final_time

        current_time -= sample_interval # maybe more accurate if min(start_time_relative, current_time)

    cap.release()
    return start_time_relative, final_time

def find_terminate_time(
    video_path, 
    start_time, 
    end_time, 
    video_start_timestamp,
    sample_interval=0.1, 
    similarity_threshold=0.99
):
    """
    Find the timestamp when page terminate by comparing frames

    Args:
        video_path: Path to the video file
        start_time: Start time of the event
        end_time: End time of the event
        video_start_timestamp: Video start timestamp for offset calculation
        sample_interval: Time interval between frame samples
        similarity_threshold: Threshold for considering frames similar

    Returns:
        Tuple of (loading_start_time, loading_end_time)
    """
    start_time_relative = start_time - video_start_timestamp
    final_time = max(start_time_relative, end_time - video_start_timestamp)

    if start_time_relative == final_time:
        return start_time_relative, final_time # the terminate time is the same as the last time stamp

    current_time = min(start_time_relative, final_time) # TODO: maybe delete more time
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    cap.set(cv2.CAP_PROP_POS_MSEC, start_time_relative * 1000)
    ret, start_frame = cap.read()
    if not ret:
        cap.release()
        return None, None

    # Sample frames backward until significant change is detected
    while current_time <= final_time:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        similarity = compute_frame_similarity(frame, start_frame)
        if similarity < similarity_threshold:
            # Found boundary between loading phase and completion
            cap.release()
            return current_time + sample_interval, current_time

        current_time += sample_interval # maybe more accurate if min(start_time_relative, current_time)

    cap.release()
    return start_time_relative, final_time


def process_single_directory(dir, num_samples=-1, load_image=True):
    # Skip .DS_Store files
    if dir.startswith(".DS_Store"):
        return None

    raw_traj = {"episode_id": dir}
    print(dir)

    # Process task name
    task_name_path = os.path.join(basedir, dir, "task_name.json")
    try:
        with open(task_name_path, encoding="utf-8-sig") as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: Empty content in {task_name_path}")
                return None
            # taskname = json.loads(content)["task_name"].split(":")
            taskname = json.loads(content)["task_name"]
            raw_traj["task_name"] = taskname
    except Exception as e:
        print(f"Error processing {task_name_path}: {e}")
        return None

    # Process metadata
    metadata_path = os.path.join(basedir, dir, "metadata.json")
    try:
        with open(metadata_path, encoding="utf-8-sig") as f:
            metadata = json.load(f)
            raw_traj["metadata"] = metadata
    except Exception as e:
        print(f"Error processing {metadata_path}: {e}")
        return None

    # Process events
    vis_events_path = os.path.join(basedir, dir, "reduced_events_vis.jsonl")
    complete_events_path = os.path.join(basedir, dir, "reduced_events_complete.jsonl")

    try:
        video_name = [f for f in os.listdir(os.path.join(basedir, dir)) if f.endswith(".mp4")][0]
        video_path = os.path.join(basedir, dir, video_name)
    except Exception as e:
        print(f"Error processing {vis_events_path}: {e}")
        return None

    try:
        events = []
        # Load complete events
        with open(complete_events_path, encoding="utf-8-sig") as f:
            complete_events = [json.loads(line) for line in f if line.strip()]

        # Verify line count
        with open(vis_events_path, encoding="utf-8-sig") as f:
            num_lines = sum(1 for line in f)
            if num_lines != len(complete_events):
                print(
                    f"Warning: Number of lines in {vis_events_path} = {num_lines} and {complete_events_path} = {len(complete_events)} do not match"
                )
                return None

        # Process visual events
        last_time_stamp = None
        with open(vis_events_path, encoding="utf-8-sig") as f:
            for index, line in enumerate(f):
                if line.strip():  # Skip empty lines
                    event = json.loads(line)

                    # Check description match
                    if event["description"] != complete_events[index]["description"]:
                        print(
                            f"Warning: Description mismatch in {vis_events_path} ({event['description']}) and {complete_events_path} ({complete_events[index]['description']}) at index {index}"
                        )
                        event["description"] = complete_events[index]["description"]

                    # Clean description
                    if "\n" in event["description"]:
                        event["description"] = event["description"].split("\n")[0]

                    # Add coordinates for click events
                    if (
                        "click" in complete_events[index]["action"].lower()
                        or "mouse_press" in complete_events[index]["action"].lower()
                        or "click" in complete_events[index]["description"].lower()
                    ) and "(" not in event["description"]:
                        event["description"] = (
                            event["description"]
                            + f" ({complete_events[index]['coordinate']['x']}, {complete_events[index]['coordinate']['y']})"
                        )
                        if "mouse_press" in complete_events[index]["action"].lower():
                            print(event["description"])

                    elif "scroll" in complete_events[index]["action"].lower():
                        event["trace"] = complete_events[index]["trace"]

                    # get the video length
                    video_length = get_duration(video_path)

                    # Calculate timestamp
                    if complete_events[index]["action"].lower() in ["click", "mouse_press","drag"]\
                        and "pre_move" in complete_events[index]:

                        timestamp, loading_end_time = find_loading_complete_time(
                            video_path,
                            start_time=complete_events[index]["pre_move"]["start_time"],
                            end_time=event["start_time"], # fixme: check from here
                            video_start_timestamp=raw_traj["metadata"]["video_start_timestamp"],
                        )
                    else:
                        timestamp = max(0.01, event["start_time"] - raw_traj["metadata"]["video_start_timestamp"] - 0.5) #TODO: modify later

                    if timestamp >= video_length:
                        print(f"Warning: Timestamp {timestamp} is greater than video length {video_length}")
                        timestamp = video_length - 0.01

                    # Extract and encode frame
                    try:
                        frame = extract_frame_at_timestamp(video_path, timestamp)
                        # check the size of the frame
                        # print(f"frame size: {frame.size}")
                        if frame is None:
                            # raise ValueError(f"No frame extracted at timestamp {timestamp}")
                            print(f"No frame extracted at timestamp {timestamp}")

                        if frame:
                            # print(f"Extracted frame at timestamp {timestamp}")
                            event["frame"] = f"data:image/png;base64,{encode_image(frame)}" if load_image else None
                    except Exception as e:
                        print(f"Error extracting frame at timestamp {timestamp}: {e}")

                    last_time_stamp = event["end_time"]

                    # Clear axtree
                    event["axtree"] = None
                    events.append(event)

                else:
                    print(f"Empty line in {vis_events_path} at index {index}")
        
        # add a terminate action
        # extract the frame at the terminate time stamp
        video_length = get_duration(video_path)
        print(f"last_time_stamp: {last_time_stamp - raw_traj['metadata']['video_start_timestamp']}")

        # extract the terminate frame using find_loading_complete_time
        if video_length + raw_traj["metadata"]["video_start_timestamp"] > last_time_stamp:  
            terminate_time_stamp, final_time_stamp = find_terminate_time(
                video_path,
                start_time=last_time_stamp,
                end_time=video_length + raw_traj["metadata"]["video_start_timestamp"],
                # end_time=last_time_stamp + 0.5,
                video_start_timestamp=raw_traj["metadata"]["video_start_timestamp"],
            )
        else:
            print(f"the last time stamp is greater than the video length")
            terminate_time_stamp = video_length - 0.01
        terminate_frame = extract_frame_at_timestamp(video_path, terminate_time_stamp)
        print(f"terminate_time_stamp: {terminate_time_stamp}")
        # print(f"terminate_frame: {terminate_frame}")
        try:
            terminate_event = {
                "action": "terminate",
                "description": "terminate the task",
                "end_time": terminate_time_stamp,
                "id": len(events),
                "start_time": terminate_time_stamp,
                "target": None,
                "time_stamp": terminate_time_stamp,
                "frame": f"data:image/png;base64,{encode_image(terminate_frame)}" if load_image else None,
                "axtree": None,
            }
            events.append(terminate_event)
        except Exception as e:
            print(f"Error extracting frame at timestamp {terminate_time_stamp}: {e}")

        # avoid the first event obs with agnet screenshot
        raw_traj["events"] = events
        if len(events) == 0:
            print(f"Warning: No events found in {vis_events_path}")
            return None
        else:
            return raw_traj

    except Exception as e:
        print(f"Error processing {vis_events_path}: {e}")
        return None


def get_raw_examples_concurrent(num_samples=-1, load_image=True, max_workers=16):
    directories = os.listdir(basedir)[:num_samples] if num_samples != -1 else os.listdir(basedir)

    # Create a partial function with fixed arguments
    process_dir = partial(process_single_directory, num_samples=num_samples, load_image=load_image)

    raw = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a list to store futures with tqdm progress bar
        futures = list(
            tqdm(executor.map(process_dir, directories), total=len(directories), desc="Processing directories")
        )

        # Filter out None results and add valid results to raw list
        raw = [result for result in futures if result is not None]

    return raw


def get_raw_examples(num_samples=-1, load_image=True):
    return get_raw_examples_concurrent(num_samples, load_image, max_workers=16)
