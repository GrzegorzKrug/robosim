import pygame
from pygame.locals import *

import numpy as np
import math
import time
import os

from itertools import product
from OpenGL.GL import *
from OpenGL.GLU import *

vertices = np.array([*product([-1, 1], repeat=3)])

edges = [(ind, ind2) 
        for ind, vertx in enumerate(vertices)
        for ind2 in np.where(
            np.absolute((vertices - vertx)).sum(axis=1)==2)[0]
        if ind2 > ind]
        

for ind,vert in enumerate(vertices):
    print(ind, vert)


for ind, ed in enumerate(edges):
    v1,v2 = ed
    vert1 = vertices[v1]
    vert2 = vertices[v2]
    # print(f"{v1}:{v2} = ",vert1, vert2, vert1.sum(), vert2.sum() )


def cube():
    glBegin(GL_LINES)
    
    for edge in edges:
        v1,v2 = edge
        vert1 = vertices[v1]
        vert2 = vertices[v2]

        glVertex3fv(vert1)
        glVertex3fv(vert2)

    glEnd()

def main():
    pygame.init()
    display = (600, 800)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0, 0, -10)
    glRotatef(0, 0, 0, 0)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                break
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit()
                    quit()
        
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        cube()

        pygame.display.flip()
        pygame.time.wait(50)
        
        glRotatef(1, 0, 1, 0)


if __name__ == '__main__':
    main()


