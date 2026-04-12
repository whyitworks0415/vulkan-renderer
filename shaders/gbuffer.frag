#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

// G-Buffer outputs
layout(location = 0) out vec4 gAlbedo;   // RGB = albedo,  A = specularStrength
layout(location = 1) out vec4 gNormal;   // RGB = normal (encoded 0..1),  A = shininess/256
layout(location = 2) out vec4 gPosition; // RGB = world position,  A = 1.0 + emissiveLum (valid 플래그 겸)

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz = position, .w = darkFloor flag
    vec4 lightDir;
} cam;

layout(set = 0, binding = 1) uniform sampler2D textures[64];

layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive;   // rgb=emissive color, a=emissive luminance (G-Buffer 패킹용)
} pc;

void main() {
    vec3 N = normalize(fragNormal);

    // ── Base albedo ───────────────────────────────────────────────────────────
    int texIdx = int(pc.textureIndex + 0.5);
    vec3 albedo;
    if (texIdx >= 0) {
        albedo = texture(textures[texIdx], fragUV).rgb * pc.baseColor.rgb;
    } else {
        albedo = fragColor * pc.baseColor.rgb;
    }

    // ── Checkerboard floor (텍스처 없는 바닥면에만 적용) ─────────────────────
    const float TILE = 1.5;
    if (texIdx < 0 && N.y > 0.95 && fragPos.y < 0.01) {
        vec2  uv      = fragPos.xz / TILE;
        vec2  checker = floor(uv);
        float parity  = mod(checker.x + checker.y, 2.0);
        bool darkFloor = cam.cameraPos.w > 0.5;
        albedo = darkFloor
            ? ((parity < 1.0) ? vec3(0.10) : vec3(0.02))
            : ((parity < 1.0) ? vec3(0.90) : vec3(0.38));
    }

    // ── Write G-Buffer ────────────────────────────────────────────────────────
    gAlbedo   = vec4(albedo, pc.specularStrength);
    gNormal   = vec4(N * 0.5 + 0.5, pc.shininess / 256.0);
    // gPosition.w = 1.0 + emissiveLum  →  >= 1.0 이면 유효 픽셀 (배경 픽셀은 0.0)
    gPosition = vec4(fragPos, 1.0 + pc.emissive.a);
}
