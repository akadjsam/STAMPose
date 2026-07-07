import argparse
from moviepy.video.io.VideoFileClip import VideoFileClip

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input video path')
parser.add_argument('--output', type=str, required=True, help='output video path')
parser.add_argument('--start', type=float, required=True, help='start time to cut (in seconds)')
parser.add_argument('--end', type=float, required=True, help='end time to cut (in seconds)')
args = parser.parse_args()

# Load the video file
clip = VideoFileClip(args.input)
print(clip.duration)

# Cut the video for the specified time range
sub_clip = clip.subclip(args.start, args.end)

# Save the cut video clip
sub_clip.write_videofile(args.output)

# Close the video file
clip.close()
