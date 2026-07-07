from moviepy.video.io.VideoFileClip import VideoFileClip

# 视频文件的路径
video_path = '/home/hyl/data/PoseMamba/demo/video/drh.mp4'

# 指定开始截取的时间（单位：秒）
start_time = 60

# 指定结束截取的时间（单位：秒）
end_time = 68

# 加载视频文件
clip = VideoFileClip(video_path)
print(clip.duration)
# 截取指定时间段的视频
sub_clip = clip.subclip(start_time, end_time)

# 保存截取的视频片段
sub_clip.write_videofile('/home/hyl/data/PoseMamba/demo/video/drh_cut.mp4')

# 关闭视频文件
clip.close()
