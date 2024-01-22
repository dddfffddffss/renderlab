from IPython.core.display import Video, display
from moviepy.editor import *

import gymnasium as gym
import time
import cv2
import os

class RenderFrame(gym.Wrapper):
    def __init__(self, env, directory, auto_release=True, size=None, fps=None, rgb=True):
        super().__init__(env)
        self.cliptime = time.time()
        self.directory = directory
        self.path = f'{self.directory}/{self.cliptime}.mp4'
        self.auto_release = auto_release
        self.active = False
        self.rgb = rgb
        
        if env.render_mode != "rgb_array":
            raise Exception("RenderFrame requires environment render mode configured to rgb_array")

        os.makedirs(self.directory, exist_ok = True)

        if size is None:
            self.env.reset()
            self.size = self.env.render().shape[:2][::-1]
        else:
            self.size = size

        if fps is None:
            if 'video.frames_per_second' in self.env.metadata:
                self.fps = self.env.metadata['video.frames_per_second']
            else:
                self.fps = 30
        else:
            self.fps = fps

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        return observation, info
    
    def isActive(self):
        return self.active

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        self._writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        self.active = True

    def _write(self):
        if self.active:
            frame = self.env.render()
            if self.rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._writer.write(frame)

    def step(self, *args, **kwargs):
        observation, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        self._write()

        return observation, reward, terminated, truncated, info
    
    def _release(self):
        self._writer.release()

    def play(self):
        self._write()
        self._release()
        
        filename = 'temp.mp4'
        clip = VideoFileClip(self.path)
        clip.write_videofile(filename, verbose = False)
        display(Video(filename, embed = True))
        os.remove(filename)

    def __del__(self):
        pass