import argparse
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input video path')
parser.add_argument('--output', type=str, required=True, help='output video path')
parser.add_argument('--width', type=int, default=1280, help='target width')
parser.add_argument('--height', type=int, default=720, help='target height')
args = parser.parse_args()

# Load the MP4 video file
clip = VideoFileClip(args.input)

# Resize the video
resized_clip = clip.resize((args.width, args.height))

# Write to file
resized_clip.write_videofile(args.output)
