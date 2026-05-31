#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 darkFloor 플래그
    vec4 lightDir; // .xyz는 씬을 향하는 방향광 벡터
    mat4 lightVP; // 방향광 시점 뷰·투영 (섀도맵 샘플링용)
} cam;

layout(set = 0, binding = 1) uniform sampler2D textures[64];

// 씬 조명 UBO(GLTF KHR_lights_punctual)
struct GpuLight {
    vec4 posRange; // Point/Spot: xyz=위치, w=범위(0=무제한) / Directional: xyz=방향, w=0
    vec4 dirType; // xyz=스팟 방향, w=타입(0=Point, 1=Directional, 2=Spot)
    vec4 colorEnab; // xyz=색상*강도, w=활성 여부(0 또는 1)
};
layout(set = 0, binding = 2) uniform SceneLightUBO {
    int numLights;
    int useSceneLights; // 1=GLTF 조명, 0=셰이더 기본 조명
    int ambientOn;
    int emissiveOn;
    GpuLight lights[8];
} sl;

// 방향광 섀도맵 (set 0, binding 3)
layout(set = 0, binding = 3) uniform sampler2D shadowMap;

layout(push_constant) uniform PushConstants {
    mat4  model;
    vec4  baseColor;
    float shininess;
    float specularStrength;
    float reflectStrength;
    float textureIndex;
    vec4  emissive; // rgb=발광 색상, a=forward 경로에서는 사용하지 않음
} pc;

// Blinn-Phong 조명 계산
vec3 blinnPhong(vec3 N, vec3 L, vec3 V, vec3 lightColor, vec3 albedo) {
    float diff = max(dot(N, L), 0.0);
    vec3  H    = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), pc.shininess) * pc.specularStrength;
    return lightColor * (diff * albedo + spec * vec3(1.0));
}

// Fresnel 하늘색 반사
vec3 fresnelReflect(vec3 N, vec3 V) {
    float cosTheta = max(dot(N, V), 0.0);
    float f = pow(1.0 - cosTheta, 3.0);
    vec3 skyColor = mix(vec3(0.4, 0.55, 0.75), vec3(0.7, 0.8, 0.95),
                        clamp(0.5 + 0.5 * (-cam.lightDir.y), 0.0, 1.0));
    return f * skyColor * pc.reflectStrength;
}

// 방향광 섀도맵 가시성(3x3 PCF). 1.0=완전히 밝음, 0.0=완전히 그림자.
float shadowVisibility(vec3 worldPos, vec3 N, vec3 L) {
    vec4 lc   = cam.lightVP * vec4(worldPos, 1.0);
    vec3 proj = lc.xyz / lc.w;
    vec2 uv   = proj.xy * 0.5 + 0.5;

    // 라이트 절두체 밖이거나 최대 깊이를 넘으면 그림자 없음으로 본다.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || proj.z > 1.0) {
        return 1.0;
    }

    // 표면 기울기에 따른 깊이 바이어스로 셰도 acne를 줄인다.
    float bias    = max(0.0015 * (1.0 - dot(N, L)), 0.0005);
    float current = proj.z - bias;

    // 3x3 PCF 평균
    float lit   = 0.0;
    vec2  texel = 1.0 / vec2(textureSize(shadowMap, 0));
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closest = texture(shadowMap, uv + vec2(x, y) * texel).r;
            lit += current <= closest ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(cam.cameraPos.xyz - fragPos);

    // 기본 albedo 계산
    int texIdx = int(pc.textureIndex + 0.5);
    vec3 albedo;
    if (texIdx >= 0) {
        albedo = texture(textures[texIdx], fragUV).rgb * pc.baseColor.rgb;
    } else {
        albedo = fragColor * pc.baseColor.rgb;
    }

    // 텍스처가 없는 바닥에만 체커보드 패턴을 적용한다.
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

    // 환경광 계산
    vec3 result = (sl.ambientOn != 0) ? vec3(0.08) * albedo : vec3(0.0);

    // 직접 조명 계산
    if (sl.useSceneLights != 0 && sl.numLights > 0) {
        // GLTF 동적 조명 사용
        for (int i = 0; i < sl.numLights && i < 8; ++i) {
            if (sl.lights[i].colorEnab.w < 0.5) continue; // 비활성 조명은 건너뛴다.
            int  ltype  = int(sl.lights[i].dirType.w + 0.5);
            vec3 lcolor = sl.lights[i].colorEnab.rgb;

            if (ltype == 1) {
                // 방향광 처리 (섀도맵 가시성 적용)
                vec3  L   = normalize(-sl.lights[i].posRange.xyz);
                float vis = shadowVisibility(fragPos, N, L);
                result += blinnPhong(N, L, V, lcolor, albedo) * vis;
            } else {
                // 포인트/스팟 조명 처리
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
                    // 스팟 조명 콘 감쇠
                    vec3  spotDir   = normalize(sl.lights[i].dirType.xyz);
                    float cosAngle  = dot(-L, spotDir);
                    // outer cone은 45도를 기본값으로 쓰고, 조명별 값으로 대체할 수 있다.
                    atten *= clamp((cosAngle - 0.5) / 0.2, 0.0, 1.0);
                }
                result += blinnPhong(N, L, V, lcolor * atten, albedo);
            }
        }
    } else {
        // GLTF 조명이 없으면 기본 방향광과 포인트 라이트를 사용한다.
        vec3  sunL   = -cam.lightDir.xyz;
        float sunVis = shadowVisibility(fragPos, N, normalize(sunL));
        result += blinnPhong(N, sunL, V,
                             vec3(0.9, 0.855, 0.765), albedo) * sunVis;

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

    // Fresnel 하늘색 반사
    result += fresnelReflect(N, V);

    // 발광 성분을 더한다.
    if (sl.emissiveOn != 0) {
        result += pc.emissive.rgb;
    }

    // Reinhard 톤매핑과 감마 보정을 적용한다.
    result   = result / (result + vec3(1.0));
    result   = pow(result, vec3(1.0 / 2.2));
    outColor = vec4(result, pc.baseColor.a);
}
