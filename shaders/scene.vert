#version 450

// ── Vertex input ──────────────────────────────────────────────────────────────
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV;

// ── Outputs to fragment shader ────────────────────────────────────────────────
layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragColor;
layout(location = 3) out vec2 fragUV;

// ── Uniforms ──────────────────────────────────────────────────────────────────
layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz = position, .w = unused
} cam;

// Per-draw model matrix + material packed in push constants (128 bytes max)
layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;        // rgb + unused
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;     // -1.0 = no texture, >= 0.0 = index into textures[]
} pc;

void main() {
    vec4 worldPos = pc.model * vec4(inPosition, 1.0);
    fragPos       = worldPos.xyz;
    fragNormal    = normalize(mat3(transpose(inverse(pc.model))) * inNormal);
    fragColor     = inColor * pc.baseColor.rgb;
    fragUV        = inUV;

    gl_Position   = cam.proj * cam.view * worldPos;
}
