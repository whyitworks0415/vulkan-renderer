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

layout(set = 0, binding = 1) uniform sampler2D textures[64];

// ── 씬 조명 UBO (GLTF KHR_lights_punctual) ───────────────────────────────────
struct GpuLight {
    vec4 posRange;    // pt/spot: xyz=position w=range(0=inf)  /  dir: xyz=direction w=0
    vec4 dirType;     // xyz=spot direction, w=type (0=Point, 1=Directional, 2=Spot)
    vec4 colorEnab;   // xyz=color*intensity, w=enabled (0.0 or 1.0)
};
layout(set = 0, binding = 2) uniform SceneLightUBO {
    int numLights;
    int useSceneLights; // 1=GLTF 조명, 0=fallback 하드코딩
    int ambientOn;
    int emissiveOn;
    GpuLight lights[8];
} sl;

layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive;         // rgb=emissive color, a=unused in forward path
} pc;

// ── Blinn-Phong ───────────────────────────────────────────────────────────────
vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, vec3 albedo) {
    float diff = max(dot(N, L), 0.0);
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), pc.shininess) * pc.specularStrength;
    return lightColor * (diff * albedo + spec * vec3(1.0));
}

// ── Fresnel sky tint ──────────────────────────────────────────────────────────
vec3 fresnelReflect(vec3 N, vec3 V) {
    float cosTheta = max(dot(N, V), 0.0);
    float f = pow(1.0 - cosTheta, 3.0);
    vec3 skyColor = mix(vec3(0.4, 0.55, 0.75), vec3(0.7, 0.8, 0.95),
                        clamp(0.5 + 0.5 * (-cam.lightDir.y), 0.0, 1.0));
    return f * skyColor * pc.reflectStrength;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cam.cameraPos.xyz - fragPos);

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

    // ── Ambient ───────────────────────────────────────────────────────────────
    vec3 result = (sl.ambientOn != 0) ? vec3(0.08) * albedo : vec3(0.0);

    // ── Lighting ──────────────────────────────────────────────────────────────
    if (sl.useSceneLights != 0 && sl.numLights > 0) {
        // GLTF 동적 조명 사용
        for (int i = 0; i < sl.numLights && i < 8; ++i) {
            if (sl.lights[i].colorEnab.w < 0.5) continue; // disabled
            int  ltype  = int(sl.lights[i].dirType.w + 0.5);
            vec3 lcolor = sl.lights[i].colorEnab.rgb;

            if (ltype == 1) {
                // Directional: posRange.xyz = direction toward scene
                vec3 L = normalize(-sl.lights[i].posRange.xyz);
                result += blinnPhong(N, L, V, lcolor, albedo);
            } else {
                // Point or Spot: posRange.xyz = position
                vec3  toLight = sl.lights[i].posRange.xyz - fragPos;
                float dist    = max(length(toLight), 0.001);
                vec3  L       = toLight / dist;
                float range   = sl.lights[i].posRange.w;
                float atten;
                if (range > 0.0) {
                    // KHR_lights_punctual 감쇄 공식
                    float ratio = clamp(dist / range, 0.0, 1.0);
                    atten = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                    atten = atten * atten / (dist * dist + 1.0);
                } else {
                    atten = 1.0 / (dist * dist + 1.0);
                }
                if (ltype == 2) {
                    // Spot: cone attenuation
                    vec3  spotDir   = normalize(sl.lights[i].dirType.xyz);
                    float cosAngle  = dot(-L, spotDir);
                    // outer cone = 45도 기본, 조명마다 다를 수 있음
                    atten *= clamp((cosAngle - 0.5) / 0.2, 0.0, 1.0);
                }
                result += blinnPhong(N, L, V, lcolor * atten, albedo);
            }
        }
    } else {
        // Fallback: 하드코딩 방향광 + 3개 포인트 라이트
        result += blinnPhong(N, -cam.lightDir.xyz, V,
                             vec3(0.9, 0.855, 0.765), albedo);

        const vec3  PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                        vec3(-4.0, 2.5, -4.0),
                                        vec3( 0.0, 4.0,  0.0));
        const vec3  PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                        vec3(4.8, 9.6, 12.0),
                                        vec3(20.0,20.0,18.0));
        for (int i = 0; i < 3; ++i) {
            vec3  toLight = PT_POS[i] - fragPos;
            float dist    = length(toLight);
            float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
            result += blinnPhong(N, toLight / dist, V, PT_COL[i] * atten, albedo);
        }
    }

    // ── Fresnel sky tint ──────────────────────────────────────────────────────
    result += fresnelReflect(N, V);

    // ── Emissive ──────────────────────────────────────────────────────────────
    if (sl.emissiveOn != 0) {
        result += pc.emissive.rgb;
    }

    // ── Tone-map (Reinhard) + gamma ───────────────────────────────────────────
    result   = result / (result + vec3(1.0));
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, pc.baseColor.a);
}
