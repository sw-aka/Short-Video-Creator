from moviepy.editor import VideoFileClip
import os
VideoFileClip('video1.mp4')#.close()
os.close('video1.mp4')
os.remove('video1.mp4')