# from moviepy.editor import VideoFileClip

# # 加载MP4视频文件
# clip = VideoFileClip('/home/hyl/data/PoseMamba/demo/output/sample_video/sample_video.mp4')

# # 使用原视频的帧率
# fps = clip.fps

# # 设置GIF的持续时间（以秒为单位），起始时间（以秒为单位）
# # 如果您想转换整个视频，可以省略start和duration参数
# start = 0  # 起始时间
# duration = clip.duration  # 视频持续时间
# clip = clip.subclip(start, duration)

# # 设置GIF的质量，0到100之间，数值越高，质量越好，文件也越大
# quality = 10  # 质量参数，可以根据需要调整

# # 将视频转换为GIF，使用原视频的帧率
# clip.write_gif('/home/hyl/data/PoseMamba/demo/output/sample_video/sample_video.gif', fps=60)
import imageio
from tqdm import tqdm
# 视频文件的路径
video_path = '/home/hyl/data/PoseMamba/demo/output/sample_video/sample_video.mp4'
# 输出GIF的路径
gif_path = '/home/hyl/data/PoseMamba/demo/output/sample_video/sample_video.gif'
 
# 读取视频
video = imageio.get_reader(video_path, 'ffmpeg')
# 创建一个空列表来存储帧
frames = []
 
# 遍历视频中的每一帧，并将其添加到列表中
for frame in video:
    frames.append(frame)
 
# 将帧列表保存为GIF
imageio.mimsave(gif_path, frames, duration=0.03)
