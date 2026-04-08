#version 450

layout(location = 0) in  vec2 fragUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz = position, .w = unused
    vec4 lightDir;  // .xyz = directional light direction (toward scene)
} cam;

// G-Buffer textures (set 1)
layout(set = 1, binding = 0) uniform sampler2D gAlbedo;   // RGB=albedo,  A=specularStrength
layout(set = 1, binding = 1) uniform sampler2D gNormal;   // RGB=normal(0..1), A=shininess/256
layout(set = 1, binding = 2) uniform sampler2D gPosition; // RGB=worldPos, A=valid(1.0)

// ── Scene lights ──────────────────────────────────────────────────────────────
const vec3  DIR_LIGHT_COLOR = vec3(1.0, 0.95, 0.85);
const float DIR_LIGHT_INT   = 0.9;

const int   NUM_POINT = 3;
const vec3  PT_POS[NUM_POINT]   = vec3[](vec3( 4.0, 2.5,  4.0),
                                          vec3(-4.0, 2.5, -4.0),
                                          vec3( 0.0, 4.0,  0.0));
const vec3  PT_COLOR[NUM_POINT] = vec3[](vec3(0.8, 0.6, 1.0),
                                          vec3(0.4, 0.8, 1.0),
                                          vec3(1.0, 1.0, 0.9));
const float PT_INT[NUM_POINT]   = float[](12.0, 12.0, 20.0);
const float PT_LIN  = 0.22;
const float PT_QUAD = 0.20;

// ── Blinn-Phong ───────────────────────────────────────────────────────────────
vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, float intensity,
                vec3 albedo, float shininess, float specStr) {
    float diff = max(dot(N, L), 0.0);
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), shininess) * specStr;
    return lightColor * intensity * (diff * albedo + spec * vec3(1.0));
}

// ── Fresnel sky tint ──────────────────────────────────────────────────────────
vec3 fresnelReflect(vec3 N, vec3 V) {
    float cosTheta = max(dot(N, V), 0.0);
    float f = pow(1.0 - cosTheta, 3.0);
    vec3 skyColor = mix(vec3(0.4, 0.55, 0.75), vec3(0.7, 0.8, 0.95),
                        clamp(0.5 + 0.5 * (-cam.lightDir.y), 0.0, 1.0));
    return f * skyColor;
}

void main() {
    vec4 gbPos = texture(gPosition, fragUV);

    // Background pixel (not covered by any geometry)
    if (gbPos.w < 0.5) {
        outColor = vec4(0.53, 0.68, 0.85, 1.0);  // sky clear color
        return;
    }

    // ── Unpack G-Buffer ───────────────────────────────────────────────────────
    vec3  fragPos   = gbPos.rgb;
    vec4  gbAlb     = texture(gAlbedo,  fragUV);
    vec4  gbN       = texture(gNormal,  fragUV);

    vec3  albedo    = gbAlb.rgb;
    float specStr   = gbAlb.a;
    vec3  N         = normalize(gbN.rgb * 2.0 - 1.0);
    float shininess = max(gbN.a * 256.0, 1.0);

    vec3 V = normalize(cam.cameraPos.xyz - fragPos);

    // ── Lighting ──────────────────────────────────────────────────────────────
    vec3 ambient = vec3(0.08) * albedo;
    vec3 result  = ambient;

    result += blinnPhong(N, -cam.lightDir.xyz, V,
                         DIR_LIGHT_COLOR, DIR_LIGHT_INT,
                         albedo, shininess, specStr);

    for (int i = 0; i < NUM_POINT; ++i) {
        vec3  toLight = PT_POS[i] - fragPos;
        float dist    = length(toLight);
        vec3  L       = toLight / dist;
        float atten   = 1.0 / (1.0 + PT_LIN * dist + PT_QUAD * dist * dist);
        result += blinnPhong(N, L, V, PT_COLOR[i], PT_INT[i] * atten,
                             albedo, shininess, specStr);
    }

    result += fresnelReflect(N, V);

    // Tone-map (Reinhard) + gamma
    result   = result / (result + vec3(1.0));
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, 1.0);
}
