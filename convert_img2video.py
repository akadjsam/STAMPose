import cv2
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='s9_greeting_243', help='input video')
parser.add_argument('--gpu', type=str, default='0', help='input video')
args = parser.parse_args()
# 图片文件夹路径
image_folder = args.img_dir
print(image_folder)
# 视频输出路径
video_name = 's9_greeting_243.mp4'

# 获取图片文件夹中的所有图片文件
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
# images = ['%d.jpg'%(idx+1) for idx in range(1600)]
# images = ['%08d.jpg'%(idx+1) for idx in range(len(os.listdir(image_folder)))]
print(images)
# 获取第一张图片的宽度和高度
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# 创建视频编码器对象
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 60
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
# 逐帧写入图片到视频
for image in images:
    print(f'将{image}写入视频')
    video.write(cv2.imread(os.path.join(image_folder, image)))

# 释放资源
cv2.destroyAllWindows()
video.release()
