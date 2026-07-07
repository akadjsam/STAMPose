import argparse
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input video path')
args = parser.parse_args()

# Load the MP4 video file
clip = VideoFileClip(args.input)

# Get the width and height of the video
width = clip.w
height = clip.h

# Print the width and height of the video
print(f"Video width: {width}px")
print(f"Video height: {height}px")
