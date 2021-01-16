import moderngl_window as mglw
import numpy as np
import os

from functools import wraps


def pre_run(parent):
    def decorator(func):
        def wrapper(*args, **kwargs):
            pass


class Example(mglw.WindowConfig):
    window_size = (600, 800)
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
                layout(triangle_strip, max_vertices=5) out;
                out vec3 color;

                void main() {
                    vec3 p1 = gl_in[0].gl_Position.xyz;
                    vec3 p2 = gl_in[1].gl_Position.xyz;
                    vec3 p3 = gl_in[2].gl_Position.xyz;

                    gl_Position = vec4(p1, 1.0);
                    color = vec3(1, 0, 0);
                    EmitVertex();

                    gl_Position = vec4(p2, 1.0);
                    color = vec3(0, 1, 0);
                    EmitVertex();

                    gl_Position = vec4(0,-0.3,0, 1.0);
                    color = vec3(0, 0, 0);
                    EmitVertex();

                    gl_Position = vec4(p3, 1.0);
                    color = vec3(0, 0, 1);
                    EmitVertex();

                    gl_Position = vec4(p1, 1.0);
                    color = vec3(1, 0, 0);
                    EmitVertex();

                    EndPrimitive();
                }
            """,
            fragment_shader='''
                #version 330
                out vec4 f_color;
                in vec3 color;

                void main() {
                    f_color = vec4(color, 1);
                }
            ''',
        )

        vertices = np.array([
            0.0, 0.8,
            -0.6, -0.8,
            0.6, -0.8,
        ], dtype='f4')

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
            self.quit()


if __name__ == '__main__':
    HelloWorld.run()

