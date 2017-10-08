from moviepy.editor import VideoClip, ImageSequenceClip
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoFileClip
from moviepy.video.VideoClip import ImageClip
import os
import _init_

path = "G:\\Scene\\video\\del\\"
image_list = []
for im in os.listdir(path):
    image_list.append(path + im)

print(image_list)
clip = ImageSequenceClip(image_list, fps=24)
clip.write_gif("data\\test_on_test_10.gif")
