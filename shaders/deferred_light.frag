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
layout(set = 1, binding = 0) uniform sampler2D gAlbedo;   // RGB=albedo, A=specStr
layout(set = 1, binding = 1) uniform sampler2D gNormal;   // RGB=normal(0..1), A=shininess/256
layout(set = 1, binding = 2) uniform sampler2D gPosition; // RGB=worldPos, A=1.0+emissiveLum (0=bg)

// ── 씬 조명 UBO (set 0, binding 2) ───────────────────────────────────────────
struct GpuLight {
    vec4 posRange;    // pt/spot: xyz=position w=range  /  dir: xyz=direction w=0
    vec4 dirType;     // xyz=spot direction, w=type (0=Point,1=Dir,2=Spot)
    vec4 colorEnab;   // xyz=color*intensity, w=enabled
};
layout(set = 0, binding = 2) uniform SceneLightUBO {
    int numLights;
    int useSceneLights;
    int ambientOn;
    int emissiveOn;
    GpuLight lights[8];
} sl;

// ── Blinn-Phong ───────────────────────────────────────────────────────────────
vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, vec3 albedo,
                float shininess, float specStr) {
    float diff = max(dot(N, L), 0.0);
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), shininess) * specStr;
    return lightColor * (diff * albedo + spec * vec3(1.0));
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

    // gPosition.w < 0.5  →  배경 픽셀 (지오메트리 없음)
    if (gbPos.w < 0.5) {
        outColor = vec4(0.53, 0.68, 0.85, 1.0);  // sky clear color
        return;
    }

    // ── G-Buffer 언팩 ─────────────────────────────────────────────────────────
    vec3  fragPos     = gbPos.rgb;
    float emissiveLum = max(0.0, gbPos.w - 1.0);  // gPosition.w = 1.0 + lum

    vec4  gbAlb     = texture(gAlbedo, fragUV);
    vec4  gbN       = texture(gNormal, fragUV);

    vec3  albedo    = gbAlb.rgb;
    float specStr   = gbAlb.a;
    vec3  N         = normalize(gbN.rgb * 2.0 - 1.0);
    float shininess = max(gbN.a * 256.0, 1.0);

    vec3 V = normalize(cam.cameraPos.xyz - fragPos);

    // ── Ambient ───────────────────────────────────────────────────────────────
    vec3 result = (sl.ambientOn != 0) ? vec3(0.08) * albedo : vec3(0.0);

    // ── Lighting ──────────────────────────────────────────────────────────────
    if (sl.useSceneLights != 0 && sl.numLights > 0) {
        // GLTF 동적 조명
        for (int i = 0; i < sl.numLights && i < 8; ++i) {
            if (sl.lights[i].colorEnab.w < 0.5) continue;
            int  ltype  = int(sl.lights[i].dirType.w + 0.5);
            vec3 lcolor = sl.lights[i].colorEnab.rgb;

            if (ltype == 1) {
                // Directional
                vec3 L = normalize(-sl.lights[i].posRange.xyz);
                result += blinnPhong(N, L, V, lcolor, albedo, shininess, specStr);
            } else {
                // Point or Spot
                vec3  toLight = sl.lights[i].posRange.xyz - fragPos;
                float dist    = max(length(toLight), 0.001);
                vec3  L       = toLight / dist;
                float range   = sl.lights[i].posRange.w;
                float atten;
                if (range > 0.0) {
                    float ratio = clamp(dist / range, 0.0, 1.0);
                    atten = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
                    atten = atten * atten / (dist * dist + 1.0);
                } else {
                    atten = 1.0 / (dist * dist + 1.0);
                }
                if (ltype == 2) {
                    vec3  spotDir  = normalize(sl.lights[i].dirType.xyz);
                    float cosAngle = dot(-L, spotDir);
                    atten *= clamp((cosAngle - 0.5) / 0.2, 0.0, 1.0);
                }
                result += blinnPhong(N, L, V, lcolor * atten, albedo, shininess, specStr);
            }
        }
    } else {
        // Fallback: 하드코딩 조명
        result += blinnPhong(N, -cam.lightDir.xyz, V,
                             vec3(0.9, 0.855, 0.765), albedo, shininess, specStr);

        const vec3 PT_POS[3] = vec3[](vec3( 4.0, 2.5,  4.0),
                                       vec3(-4.0, 2.5, -4.0),
                                       vec3( 0.0, 4.0,  0.0));
        const vec3 PT_COL[3] = vec3[](vec3(9.6, 7.2, 12.0),
                                       vec3(4.8, 9.6, 12.0),
                                       vec3(20.0,20.0,18.0));
        for (int i = 0; i < 3; ++i) {
            vec3  toLight = PT_POS[i] - fragPos;
            float dist    = length(toLight);
            float atten   = 1.0 / (1.0 + 0.22 * dist + 0.20 * dist * dist);
            result += blinnPhong(N, toLight / dist, V, PT_COL[i] * atten,
                                 albedo, shininess, specStr);
        }
    }

    // ── Fresnel sky tint ──────────────────────────────────────────────────────
    result += fresnelReflect(N, V);

    // ── Emissive (G-Buffer의 gPosition.w 에서 추출한 발광 강도) ───────────────
    if (sl.emissiveOn != 0 && emissiveLum > 0.0) {
        result += albedo * emissiveLum;
    }

    // ── Tone-map (Reinhard) + gamma ───────────────────────────────────────────
    result   = result / (result + vec3(1.0));
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, 1.0);
}
