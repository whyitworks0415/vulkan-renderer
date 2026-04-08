#version 450

layout(location = 0) out vec2 fragUV;

// Full-screen triangle: 3 vertices, no vertex buffer
// gl_VertexIndex 0,1,2 → covers the entire viewport
void main() {
    vec2 uv     = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV      = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
