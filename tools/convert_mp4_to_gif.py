import imageio
from tqdm import tqdm

video_path = 'demo/output/sample/sample.mp4'
gif_path = 'demo/output/sample/sample.gif'

# GIF은 프레임 지연시간을 1/100초 단위 정수로만 저장하므로,
# 100의 약수가 아닌 fps(예: 30, 29.97)를 그대로 쓰면 반올림 오차로 재생 속도가 어긋난다.
# 100의 약수인 fps로 프레임을 리샘플링해 원본 재생시간에 맞춘다.
target_fps = 25

video = imageio.get_reader(video_path, 'ffmpeg')
original_fps = video.get_meta_data()['fps']
frames = [frame for frame in video]

duration = len(frames) / original_fps
n_out = max(1, round(duration * target_fps))
indices = [min(round(i * original_fps / target_fps), len(frames) - 1) for i in range(n_out)]
resampled_frames = [frames[i] for i in tqdm(indices, desc="Resampling frames")]

imageio.mimsave(gif_path, resampled_frames, fps=target_fps, loop=0)
