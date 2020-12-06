import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os


class PlainModel:
    def __init__(self, segments, anchor=None):
        self.dimension = 2
       
        if anchor is None:
            anchor = [0,0]
        assert self.dimension == len(anchor)

        self.anchor = np.array(anchor)
        self.segments = segments
        self.joints = np.array(segments) * 0

        self.pos = self.get_pos(self.joints)
    
    def set_pos(self, joints):
        self.pos = self.get_pos(joints)
        return self.pos
        
    def get_pos(self, joints: "Array[rads]"=None):
        if joints is None:
            return self.pos
         
        assert len(joints) == len(self.segments), "You passed invalid joint value:"+str(joints)
        all_pos = np.zeros((self.dimension, len(self.segments)))
        absangle = 0
        for ind, (seg, angle) in enumerate(zip(self.segments, joints)):
            absangle += angle
            if ind == 0:
                prev_pos = self.anchor.copy()
            else:
                prev_pos = all_pos[:, ind-1]
           
            print(f"Prev: {prev_pos}")
            x = math.sin(absangle)*seg
            y = math.cos(absangle)*seg
            all_pos[:, ind] = prev_pos + [x, -y]

        return all_pos


if __name__ == "__main__":
    model = PlainModel([15, 10, 10, 5])
    print("Model loaded")
    print(f"Model pos:\n{model.pos}")

    
