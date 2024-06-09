# -*- coding: utf-8 -*-
"""
AutoCropper_Improved
Repurpose long videos into shorts with automatic cropping based on detected faces.
"""

# Cell 1: Import necessary libraries and set up environment

import cv2
import subprocess
import openai
import numpy as np
import json
import math
import os
from pytube import YouTube
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

load_dotenv(".env")
api_key = os.getenv("google-api-key")

# Cell 2: Download YouTube Video function

def download_video(url, filename):
    yt = YouTube(url)
    video = yt.streams.filter(file_extension='mp4', resolution=yt.streams.get_highest_resolution().resolution).first()
    
    video.download(filename=filename)
    print(f"Video downloaded as {filename}")
    fps = video.fps
    return float(fps)

# Cell 3: Segment Video function

def segment_video(response, input_file):
    for i, segment in enumerate(response):
        start_time = math.floor(float(segment.get("start_time", 0)))
        end_time = math.ceil(float(segment.get("end_time", 0))) + 2
        output_file = f"output{str(i).zfill(3)}.mp4"
        
        # Extract video segment
        command_video = f"ffmpeg -i {input_file} -ss {start_time} -to {end_time} -c copy -an temp_video.mp4"
        subprocess.call(command_video, shell=True)
        
        # Extract audio segment
        command_audio = f"ffmpeg -i {input_file} -ss {start_time} -to {end_time} -c copy -vn temp_audio.aac"
        subprocess.call(command_audio, shell=True)
        
        # Combine video and audio segments
        command_combine = f"ffmpeg -i temp_video.mp4 -i temp_audio.aac -c:v copy -c:a aac {output_file}"
        subprocess.call(command_combine, shell=True)
        
        # Clean up temporary files
        os.remove("temp_video.mp4")
        os.remove("temp_audio.aac")
        
        print(f"Segment {i} saved as {output_file}")

# Cell 4: Face Detection function (Modified for stability)

def detect_faces(video_file):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_file)
    faces = []
    frame_count = 0
    face_history = []  # Track faces over time for stability

    while len(faces) < 5:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Add detected faces to history, removing duplicates
        for face in detected_faces:
            if not any(np.array_equal(face, f) for f in face_history):
                face_history.append(face)

        # Check if same faces appear for a few frames for stability
        if len(face_history) > 10:  # Adjust this threshold as needed
            unique_faces = set(tuple(face) for face in face_history)
            faces = [list(face) for face in unique_faces]
            face_history = []  # Reset history

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, {len(faces)} unique faces detected.")

    cap.release()

    if len(faces) > 0:
        return faces
    else:
        print("No faces detected.")
        return None

# Cell 5: Crop Video function (Improved cropping and stabilization)

def crop_video(faces, input_file, output_file, fps):
    if not faces:
        print("No faces detected in the video.")
        return

    CROP_RATIO = 0.9
    VERTICAL_RATIO = 9 / 16
    SMOOTHING_WINDOW = 15  # Adjust for more or less smoothing

    cap = cv2.VideoCapture(input_file)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_height = int(frame_height * CROP_RATIO)
    target_width = int(target_height * VERTICAL_RATIO)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video_file = "temp_cropped_video.mp4"
    output_video = cv2.VideoWriter(temp_video_file, fourcc, fps, (target_width, target_height))

    positions = []
    frame_count = 0

    # Track face positions over time for smoother transitions
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        face_positions = []
        for face in faces:
            x, y, w, h = face
            face_positions.append((x, y, w, h))

        if face_positions:
            avg_face = np.mean(face_positions, axis=0).astype(int)
            positions.append(avg_face)

        frame_count += 1

    # Smooth out face positions using a sliding window average
    smoothed_positions = np.copy(positions)
    for i in range(len(positions)):
        start = max(0, i - SMOOTHING_WINDOW // 2)
        end = min(len(positions), i + SMOOTHING_WINDOW // 2)
        smoothed_positions[i] = np.mean(positions[start:end], axis=0).astype(int)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0

    # Crop and write frames to the output video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        x, y, w, h = smoothed_positions[idx]
        # Calculate cropping coordinates to center the face
        crop_x = max(0, x + (w - target_width) // 2)
        crop_y = max(0, y + (h - target_height) // 2)
        crop_x2 = min(crop_x + target_width, frame_width)
        crop_y2 = min(crop_y + target_height, frame_height)

        # Crop the frame
        cropped_frame = frame[crop_y:crop_y2, crop_x:crop_x2]
        # No resizing for auto-reframing
        output_video.write(cropped_frame)

        idx += 1

    cap.release()
    output_video.release()

    temp_audio_file = "temp_audio.aac"

    # Extract audio from the original video
    subprocess.call(f"ffmpeg -i {input_file} -q:a 0 -map a {temp_audio_file}", shell=True)

    # Combine the cropped video with the extracted audio
    subprocess.call(f"ffmpeg -i {temp_video_file} -i {temp_audio_file} -map 0:v? -map 1:a -c:v libx264 -c:a aac -strict experimental -shortest {output_file}", shell=True)
    # Clean up temporary files
    os.remove(temp_video_file)
    os.remove(temp_audio_file)

    print(f"Video cropped and saved as {output_file}")

# Cell 6: Get Transcript function (No changes)

def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=('en', 'hi'))
    formatted_transcript = ''

    for entry in transcript:
        start_time = "{:.2f}".format(entry['start'])
        end_time = "{:.2f}".format(entry['start'] + entry['duration'])
        text = entry['text']
        formatted_transcript += f"{start_time} --> {end_time} : {text}\n"

    return formatted_transcript

# Cell 7: Analyze Transcript with GPT function (No changes)

response_obj = '''[
  {
    "start_time": 97.19, 
    "end_time": 127.43,
    "description": "Spoken Text here",
    "duration": 36
  },
  {
    "start_time": 169.58,
    "end_time": 199.10,
    "description": "Spoken Text here",
    "duration": 33 
  }
]'''

def analyze_transcript(transcript):
    prompt = f"This is a transcript of a video. Please identify the 3 most viral sections from the whole, make sure they are more than 30 seconds in duration. Respond only in this format: {response_obj}.\nHere is the transcription:\n{transcript}"
    instruction = "You are a ViralGPT helpful assistant. You are master at reading YouTube transcripts and identifying the most interesting and viral content."
    safe = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
    config = {
        "max_output_tokens": 512
    }
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash", generation_config=config, system_instruction=instruction)
    response = model.generate_content(prompt)
    response = response.text.replace("```json", "").replace("```", "")
    print(response)
    return json.loads(response)

# Main function and execution

def main(video_url, video_id, output_prefix="output"):
    # Step 1: Download the video
    video_filename = f"{output_prefix}_input.mp4"
    try:
        fps = download_video(video_url, video_filename)
    except Exception as e:
        print(f"Failed to download video: {e}")
        return

    # Step 2: Detect faces in the video
    try:
        faces = detect_faces(video_filename)
    except Exception as e:
        print(f"Failed to detect faces: {e}")
        return

    # Step 3: Crop the video around the detected faces
    if faces:
        print(fps)
        cropped_filename = f"{output_prefix}_cropped.mp4"
        try:
            crop_video(faces, video_filename, cropped_filename, fps=fps)
        except Exception as e:
            print(f"Failed to crop video: {e}")
            return

    # Step 4: Get the transcript of the video
    try:
        transcript = get_transcript(video_id)
    except Exception as e:
        print(f"Failed to get transcript: {e}")
        return

    # Step 5: Analyze the transcript to find interesting segments
    try:
        response = analyze_transcript(transcript)
        print("Interesting Segments:", response)
    except Exception as e:
        print(f"Failed to analyze transcript: {e}")
        return

    # Step 6: Segment the video based on the analysis
    try:
        segment_video(response, cropped_filename if faces else video_filename)
    except Exception as e:
        print(f"Failed to segment video: {e}")
        return

if __name__ == "__main__":
    video_url = input("Video URL: ")
    video_id = video_url.split("https://www.youtube.com/watch?v=")[-1]
    main(video_url, video_id)
