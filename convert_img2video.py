import cv2
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='s9_greeting_243', help='input video')
parser.add_argument('--gpu', type=str, default='0', help='input video')
args = parser.parse_args()
# Image folder path
image_folder = args.img_dir
print(image_folder)
# Video output path
video_name = 's9_greeting_243.mp4'

# Get all image files in the image folder
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
# images = ['%d.jpg'%(idx+1) for idx in range(1600)]
# images = ['%08d.jpg'%(idx+1) for idx in range(len(os.listdir(image_folder)))]
print(images)
# Get the width and height of the first image
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# Create video encoder object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 60
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
# Write images to video frame by frame
for image in images:
    print(f'Writing {image} to video')
    video.write(cv2.imread(os.path.join(image_folder, image)))

# Release resources
cv2.destroyAllWindows()
video.release()
