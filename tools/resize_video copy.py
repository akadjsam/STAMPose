from moviepy.editor import VideoFileClip

# 加载MP4视频文件
clip = VideoFileClip("/home/hyl/data/PoseMamba/demo/video/xrxjwd.mp4")

# 设置目标宽度和高度
target_width = 1280
target_height = 720

# 调整视频尺寸
resized_clip = clip.resize((target_width, target_height))
# 指定开始截取的时间（单位：秒）
start_time = 60

# 指定结束截取的时间（单位：秒）
end_time = 68

# 截取指定时间段的视频
sub_clip = resized_clip.subclip(start_time, end_time)

# 保存截取的视频片段
sub_clip.write_videofile('/home/hyl/data/PoseMamba/demo/video/xrxjwd_cut.mp4')

# 关闭视频文件
clip.close()
