from moviepy.editor import VideoFileClip

# 加载MP4视频文件
clip = VideoFileClip("/home/hyl/data/PoseMamba/demo/video/tj.mp4")

# 设置目标宽度和高度
target_width = 1280
target_height = 720

# 调整视频尺寸
resized_clip = clip.resize((target_width, target_height))

# 写入文件
resized_clip.write_videofile("/home/hyl/data/PoseMamba/demo/video/tj_1280_720.mp4")
