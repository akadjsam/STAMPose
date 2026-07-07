from moviepy.editor import VideoFileClip

# 加载MP4视频文件
clip = VideoFileClip("/home/hyl/data/MotionBERT/demo/video/chuyin1.mp4")

# 获取视频的宽度和高度
width = clip.w
height = clip.h

# 打印视频的宽度和高度
print(f"视频宽度: {width}px")
print(f"视频高度: {height}px")
'''
一定要是
视频宽度: 1280px
视频高度: 720px
'''