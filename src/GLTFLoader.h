#pragma once
#include <string>
#include <vector>
#include <cstdint>
#include <glm/glm.hpp>

struct Vertex;
struct DrawObject;

// ─────────────────────────────────────────────────────────────────────────────
//  GLTFTextureData  –  CPU 메모리에 디코딩된 텍스처 한 장
//  tinygltf + stb_image 가 GLB 내장 이미지를 RGBA8 로 자동 디코딩한다.
// ─────────────────────────────────────────────────────────────────────────────
struct GLTFTextureData {
    int width  = 0;
    int height = 0;
    std::vector<uint8_t> pixels; // RGBA8 순서, 크기 = width * height * 4
};

// ─────────────────────────────────────────────────────────────────────────────
//  GLTFSceneDesc  –  GLTF 씬에서 추출한 메타데이터
//  .scene 파일의 SceneDesc 와 대응하는 역할을 한다.
// ─────────────────────────────────────────────────────────────────────────────
struct GLTFSceneDesc {
    std::string name        = "GLTF Scene";
    glm::vec3   cameraPos   = {0.f, 5.f, -15.f};
    float       cameraYaw   = 90.f;
    float       cameraPitch = -10.f;
};

// ─────────────────────────────────────────────────────────────────────────────
//  loadGLTFScene
//
//  .glb / .gltf 파일을 읽어 각 노드(Blender 오브젝트)를 별도의 DrawObject 로
//  변환한다. 오브젝트들은 합쳐지지 않으므로 Frustum Culling 등 최적화 기법을
//  오브젝트 단위로 적용할 수 있다.
//
//  인자:
//    path      – .glb 또는 .gltf 파일 경로
//    verts     – 글로벌 버텍스 배열 (데이터를 뒤에 추가함)
//    inds      – 글로벌 인덱스 배열 (데이터를 뒤에 추가함)
//    objects   – DrawObject 목록 (항목을 뒤에 추가함)
//    sceneDesc – 씬 이름·카메라 초기 위치 등 메타데이터 반환
//    textures  – RGBA8 디코딩된 텍스처 목록 (항목을 뒤에 추가함)
//
//  반환: 성공 true, 실패 false
// ─────────────────────────────────────────────────────────────────────────────
bool loadGLTFScene(const std::string&          path,
                   std::vector<Vertex>&         verts,
                   std::vector<uint32_t>&       inds,
                   std::vector<DrawObject>&     objects,
                   GLTFSceneDesc&               sceneDesc,
                   std::vector<GLTFTextureData>& textures);
