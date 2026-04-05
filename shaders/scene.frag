#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz = position, .w = darkFloor flag
    vec4 lightDir;  // .xyz = directional light direction (toward scene)
} cam;

// Texture array: up to 64 textures per scene (unused slots filled with 1×1 white)
layout(set = 0, binding = 1) uniform sampler2D textures[64];

layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength; // kept for ABI compat (unused after reflection removal)
    float textureIndex;    // -1.0 = no texture, >= 0.0 = index into textures[]
} pc;

// ── Scene lights ──────────────────────────────────────────────────────────────
// Directional light direction is passed dynamically via CameraUBO.lightDir
// (scroll wheel rotates it; cam.lightDir.xyz points FROM light TOWARD scene)
const vec3  DIR_LIGHT_COLOR = vec3(1.0, 0.95, 0.85);
const float DIR_LIGHT_INT   = 0.9;

// Point lights
const int   NUM_POINT = 3;
const vec3  PT_POS[NUM_POINT]   = vec3[](vec3( 4.0, 2.5,  4.0),
                                          vec3(-4.0, 2.5, -4.0),
                                          vec3( 0.0, 4.0,  0.0));
const vec3  PT_COLOR[NUM_POINT] = vec3[](vec3(0.8, 0.6, 1.0),   // purple
                                          vec3(0.4, 0.8, 1.0),   // cyan
                                          vec3(1.0, 1.0, 0.9));  // white
const float PT_INT[NUM_POINT]   = float[](12.0, 12.0, 20.0);
const float PT_LIN  = 0.22;
const float PT_QUAD = 0.20;

// ── Blinn-Phong helper ────────────────────────────────────────────────────────
vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, float intensity, vec3 albedo) {
    float diff = max(dot(N, L), 0.0);
    vec3 H     = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), pc.shininess) * pc.specularStrength;
    return lightColor * intensity * (diff * albedo + spec * vec3(1.0));
}

// ── Fake ground reflection (Fresnel-style tint on floor-facing surfaces) ──────
vec3 fresnelReflect(vec3 N, vec3 V) {
    float cosTheta = max(dot(N, V), 0.0);
    float f = pow(1.0 - cosTheta, 3.0); // Schlick approx
    vec3 skyColor = mix(vec3(0.4, 0.55, 0.75), vec3(0.7, 0.8, 0.95),
                        clamp(0.5 + 0.5 * (-cam.lightDir.y), 0.0, 1.0));
    return f * skyColor * pc.reflectStrength;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cam.cameraPos.xyz - fragPos);

    // ── Base albedo: texture or vertex color ──────────────────────────────────
    int texIdx = int(pc.textureIndex + 0.5); // round float index to int
    vec3 albedo;
    if (texIdx >= 0) {
        albedo = texture(textures[texIdx], fragUV).rgb * pc.baseColor.rgb;
    } else {
        albedo = fragColor * pc.baseColor.rgb;
    }

    // ── Checkerboard floor ────────────────────────────────────────────────────
    // Detected by upward-facing normal (floor surface).
    // Pattern computed from world XZ position → sharp, interpolation-free tiles.
    const float TILE = 1.5;   // tile size in scene units
    if (N.y > 0.95 && fragPos.y < 0.01) {
        // Use fract-based checker to avoid int overflow on large coordinates
        vec2  uv      = fragPos.xz / TILE;
        vec2  checker = floor(uv);
        float parity  = mod(checker.x + checker.y, 2.0);
        // cam.cameraPos.w == 1.0 → dark floor,  0.0 → bright floor
        bool darkFloor = cam.cameraPos.w > 0.5;
        albedo = darkFloor
            ? ((parity < 1.0) ? vec3(0.10) : vec3(0.02))
            : ((parity < 1.0) ? vec3(0.90) : vec3(0.38));
    }

    // ── Lighting ──────────────────────────────────────────────────────────────
    vec3 ambient = vec3(0.08) * albedo;
    vec3 result  = ambient;

    // Directional light (direction from UBO, scroll-wheel controlled)
    result += blinnPhong(N, -cam.lightDir.xyz, V, DIR_LIGHT_COLOR, DIR_LIGHT_INT, albedo);

    // Point lights
    for (int i = 0; i < NUM_POINT; ++i) {
        vec3  toLight = PT_POS[i] - fragPos;
        float dist    = length(toLight);
        vec3  L       = toLight / dist;
        float atten   = 1.0 / (1.0 + PT_LIN * dist + PT_QUAD * dist * dist);
        result += blinnPhong(N, L, V, PT_COLOR[i], PT_INT[i] * atten, albedo);
    }

    // Fresnel sky tint (all surfaces — sky-bounce ambient)
    result += fresnelReflect(N, V);

    // Tone-map (Reinhard) + gamma
    result = result / (result + vec3(1.0));
    result = pow(result, vec3(1.0 / 2.2));

    outColor = vec4(result, pc.baseColor.a);
}
