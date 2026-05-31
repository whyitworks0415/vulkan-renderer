#version 450

layout(location = 0) out vec2 fragUV;

// 버텍스 버퍼 없이 전체화면 삼각형을 생성한다.
// 세 정점만으로 전체 뷰포트를 덮는다.
void main() {
    vec2 uv     = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    fragUV      = uv;
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
