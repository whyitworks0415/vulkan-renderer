#version 450

layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragColor;
layout(location = 3) in vec2 fragUV;

// 지오메트리 패스가 기록하는 G-Buffer 출력
layout(location = 0) out vec4 gAlbedo; // RGB=기본색, A=스페큘러 강도
layout(location = 1) out vec4 gNormal; // RGB=0..1로 인코딩한 법선, A=shininess/256
layout(location = 2) out vec4 gPosition; // RGB=월드 위치, A=1.0+발광 강도(유효 픽셀 표시)
layout(location = 3) out vec4 gMaterial; // R=반사 강도

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos; // .xyz는 카메라 위치, .w는 darkFloor 플래그
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
    vec4  emissive; // rgb=발광 색상, a=발광 강도
} pc;

void main() {
    vec3 N = normalize(fragNormal);

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

    // 발광 강도는 gPosition.a에 패킹해 조명 패스로 넘긴다.
    gAlbedo   = vec4(albedo, pc.specularStrength);
    gNormal   = vec4(N * 0.5 + 0.5, pc.shininess / 256.0);
    // gPosition.w가 1.0 이상이면 유효 픽셀이고, 0.0이면 배경이다.
    gPosition = vec4(fragPos, 1.0 + pc.emissive.a);
    gMaterial = vec4(pc.reflectStrength, 0.0, 0.0, 0.0);
}
