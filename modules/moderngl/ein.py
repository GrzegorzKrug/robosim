import moderngl_window as mglw
import numpy as np
import os

from functools import wraps


def pre_run(parent):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pass


class Example(mglw.WindowConfig):
    window_size = (800, 600)
    pass


class HelloWorld(Example):
    gl_version = (3, 3)
    title = "Hello World"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_vert;
                out float dist;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    dist = sqrt(pow(in_vert[0], 2) + pow(in_vert[1], 2));
                }
            ''',
             geometry_shader="""
                #version 330

                layout(triangles) in;
                layout(triangle_strip, max_vertices=3) out;

                flat out vec2 center;
                out vec2 pos;

                void main() {
                    vec2 p1 = gl_in[0].gl_Position.xy;
                    vec2 p2 = gl_in[1].gl_Position.xy;
                    vec2 p3 = gl_in[2].gl_Position.xy;

                    vec2 c = vec2(0.0, 0.0);
                    c[0] = (p1[0] + p2[0] + p3[0])/3.0;
                    c[1] = (p1[1] + p2[1] + p3[1])/3.0;

                    gl_Position = vec4(p1, 0.0, 1.0);
                    center = c;
                    pos = p1;
                    EmitVertex();

                    gl_Position = vec4(p2, 0.0, 1.0);
                    center = c;
                    pos = p2;
                    EmitVertex();

                    gl_Position = vec4(p3, 0.0, 1.0);
                    center = c;
                    pos = p3;
                    EmitVertex();

                    EndPrimitive();
                }
            """,
            fragment_shader="""
                #version 330

                in vec2 pos;
                out vec4 f_color;
                flat in vec2 center;

                void main() {
                    float d = distance(center, pos);
                    f_color = vec4(0.0, 0.0, d, 1.0);
                }
            """,
        )

        vertices = np.array([
            0, 1,
            -0.5, 0,
            0.5, 0,
        ], dtype='f4')
        vert = vertices.reshape(-1, 2)
        print(vert)
        print(vert.mean(axis=0))

        self.vbo = self.ctx.buffer(vertices)
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_vert')

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 0.5)
        self.vao.render()

    def handle_key_press():
        print("Handle")

    def key_event(self, key, action, modifiers):
        keys = self.wnd.keys
        if key == keys.Q:
            print(dir(self))
            self.quit()


if __name__ == '__main__':
    HelloWorld.run()

