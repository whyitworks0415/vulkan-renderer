// tinygltf – 헤더 온리 라이브러리. 이 .cpp 파일 하나에서만 구현체를 정의한다.
// stb_image 로 GLB 내장 텍스처를 RGBA8 로 자동 디코딩한다.
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include "GLTFLoader.h"
#include "VulkanApp.h"   // Vertex, DrawObject, PushConstants

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>

// ─────────────────────────────────────────────────────────────────────────────
//  computeBoundSphereLocal
//  로컬 공간 AABB 와 월드 변환 행렬로 바운딩 구를 계산한다.
//  VulkanApp::computeBoundSphere 와 동일한 알고리즘 (AABB 8 코너 변환).
// ─────────────────────────────────────────────────────────────────────────────
static void computeBoundSphereLocal(DrawObject&      obj,
                                    const glm::vec3& bmin,
                                    const glm::vec3& bmax)
{
    const glm::mat4& M = obj.push.model;
    glm::vec3 corners[8];
    int k = 0;
    for (float x : {bmin.x, bmax.x})
        for (float y : {bmin.y, bmax.y})
            for (float z : {bmin.z, bmax.z})
                corners[k++] = glm::vec3(M * glm::vec4(x, y, z, 1.f));

    glm::vec3 center{};
    for (auto& c : corners) center += c;
    center /= 8.f;

    float radius = 0.f;
    for (auto& c : corners)
        radius = std::max(radius, glm::length(c - center));

    obj.boundCenter = center;
    obj.boundRadius = radius;
}
static float determinant3x3(const glm::mat4& m)
{
    const glm::vec3 c0(m[0]);
    const glm::vec3 c1(m[1]);
    const glm::vec3 c2(m[2]);
    return glm::dot(c0, glm::cross(c1, c2));
}

static void alignTriangleWindingToNormals(const std::vector<Vertex>& verts,
                                          std::vector<uint32_t>&     inds,
                                          uint32_t                   idxStart,
                                          uint32_t                   idxCount)
{
    for (uint32_t i = 0; i + 2 < idxCount; i += 3) {
        const Vertex& v0 = verts[inds[idxStart + i + 0]];
        const Vertex& v1 = verts[inds[idxStart + i + 1]];
        const Vertex& v2 = verts[inds[idxStart + i + 2]];

        glm::vec3 faceN = glm::cross(v1.pos - v0.pos, v2.pos - v0.pos);
        float faceLen2 = glm::dot(faceN, faceN);
        if (faceLen2 <= 1e-12f) continue;

        glm::vec3 avgN = v0.normal + v1.normal + v2.normal;
        float avgLen2 = glm::dot(avgN, avgN);
        if (avgLen2 <= 1e-12f) continue;

        if (glm::dot(faceN, avgN) < 0.f)
            std::swap(inds[idxStart + i + 1], inds[idxStart + i + 2]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  nodeLocalTransform
//  GLTF 노드의 로컬 변환 행렬을 반환한다.
//  matrix 가 있으면 그대로 사용, 없으면 TRS 에서 조합.
// ─────────────────────────────────────────────────────────────────────────────
static glm::mat4 nodeLocalTransform(const tinygltf::Node& node)
{
    if (!node.matrix.empty()) {
        // GLTF 행렬은 열 우선(column-major), double 타입
        glm::mat4 m;
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                m[col][row] = static_cast<float>(node.matrix[col * 4 + row]);
        return m;
    }

    glm::mat4 T(1.f), R(1.f), S(1.f);

    if (!node.translation.empty())
        T = glm::translate(glm::mat4(1.f),
                           glm::vec3(static_cast<float>(node.translation[0]),
                                     static_cast<float>(node.translation[1]),
                                     static_cast<float>(node.translation[2])));

    if (!node.rotation.empty()) {
        // GLTF 쿼터니언 순서: [x, y, z, w]
        glm::quat q(static_cast<float>(node.rotation[3]),  // w
                    static_cast<float>(node.rotation[0]),  // x
                    static_cast<float>(node.rotation[1]),  // y
                    static_cast<float>(node.rotation[2])); // z
        R = glm::mat4_cast(q);
    }

    if (!node.scale.empty())
        S = glm::scale(glm::mat4(1.f),
                       glm::vec3(static_cast<float>(node.scale[0]),
                                 static_cast<float>(node.scale[1]),
                                 static_cast<float>(node.scale[2])));

    return T * R * S; // 오른쪽부터 적용: Scale → Rotate → Translate
}

// ─────────────────────────────────────────────────────────────────────────────
//  processPrimitive
//  GLTF 메시의 프리미티브 하나를 글로벌 verts/inds 에 추가하고
//  해당 DrawObject 를 objects 에 추가한다.
//
//  texIndexMap: GLTF image index → 로컬 텍스처 배열 인덱스 (없으면 -1)
// ─────────────────────────────────────────────────────────────────────────────
static void processPrimitive(const tinygltf::Model&              model,
                              const tinygltf::Primitive&           prim,
                              const glm::mat4&                     worldTransform,
                              const std::unordered_map<int, int>&  texIndexMap,
                              std::vector<Vertex>&                 verts,
                              std::vector<uint32_t>&               inds,
                              std::vector<DrawObject>&             objects)
{
    // 삼각형 프리미티브만 처리 (라인, 포인트 등은 렌더러가 지원하지 않음)
    if (prim.mode != TINYGLTF_MODE_TRIANGLES) return;

    auto posIt = prim.attributes.find("POSITION");
    if (posIt == prim.attributes.end()) return; // 위치 없으면 건너뜀

    // ── POSITION 버퍼 ────────────────────────────────────────────────────────
    const tinygltf::Accessor&   posAcc  = model.accessors[posIt->second];
    const tinygltf::BufferView& posView = model.bufferViews[posAcc.bufferView];
    const uint8_t* posPtr = model.buffers[posView.buffer].data.data()
                          + posView.byteOffset + posAcc.byteOffset;
    int posStride = posAcc.ByteStride(posView);

    // ── NORMAL 버퍼 (선택적) ─────────────────────────────────────────────────
    const uint8_t* nrmPtr    = nullptr;
    int            nrmStride = 12;
    auto nrmIt = prim.attributes.find("NORMAL");
    if (nrmIt != prim.attributes.end()) {
        const tinygltf::Accessor&   nrmAcc  = model.accessors[nrmIt->second];
        const tinygltf::BufferView& nrmView = model.bufferViews[nrmAcc.bufferView];
        nrmPtr    = model.buffers[nrmView.buffer].data.data()
                  + nrmView.byteOffset + nrmAcc.byteOffset;
        nrmStride = nrmAcc.ByteStride(nrmView);
    }

    // ── TEXCOORD_0 버퍼 (선택적) ─────────────────────────────────────────────
    const uint8_t* uvPtr    = nullptr;
    int            uvStride = 8; // 기본값: float2 (8 bytes)
    auto uvIt = prim.attributes.find("TEXCOORD_0");
    if (uvIt != prim.attributes.end()) {
        const tinygltf::Accessor&   uvAcc  = model.accessors[uvIt->second];
        const tinygltf::BufferView& uvView = model.bufferViews[uvAcc.bufferView];
        uvPtr    = model.buffers[uvView.buffer].data.data()
                 + uvView.byteOffset + uvAcc.byteOffset;
        uvStride = uvAcc.ByteStride(uvView);
    }

    // ── 재질 베이스 컬러 + 텍스처 인덱스 ────────────────────────────────────
    glm::vec3 matColor(0.8f);
    int       texIdx = -1;
    bool      twoSided = false;
    if (prim.material >= 0) {
        const auto& material = model.materials[prim.material];
        const auto& pbr = material.pbrMetallicRoughness;
        twoSided = material.doubleSided;

        // 베이스 컬러 팩터 (텍스처 없을 때 색상으로 사용)
        if (pbr.baseColorFactor.size() >= 3)
            matColor = glm::vec3(static_cast<float>(pbr.baseColorFactor[0]),
                                 static_cast<float>(pbr.baseColorFactor[1]),
                                 static_cast<float>(pbr.baseColorFactor[2]));

        // 베이스 컬러 텍스처
        int baseColorTexIdx = pbr.baseColorTexture.index;
        if (baseColorTexIdx >= 0 && baseColorTexIdx < (int)model.textures.size()) {
            int imageIdx = model.textures[baseColorTexIdx].source;
            auto it = texIndexMap.find(imageIdx);
            if (it != texIndexMap.end())
                texIdx = it->second;
        }
    }

    // ── 버텍스 생성 ──────────────────────────────────────────────────────────
    uint32_t  vertBase = static_cast<uint32_t>(verts.size());
    glm::vec3 bboxMin( std::numeric_limits<float>::max());
    glm::vec3 bboxMax(-std::numeric_limits<float>::max());

    for (int vi = 0; vi < static_cast<int>(posAcc.count); ++vi) {
        const float* pf = reinterpret_cast<const float*>(posPtr + vi * posStride);
        Vertex vtx{};
        vtx.pos   = glm::vec3(pf[0], pf[1], pf[2]);
        vtx.color = matColor;

        if (nrmPtr) {
            const float* nf = reinterpret_cast<const float*>(nrmPtr + vi * nrmStride);
            vtx.normal = glm::vec3(nf[0], nf[1], nf[2]);
        } else {
            vtx.normal = glm::vec3(0.f, 1.f, 0.f);
        }

        if (uvPtr) {
            const float* uf = reinterpret_cast<const float*>(uvPtr + vi * uvStride);
            vtx.uv = glm::vec2(uf[0], uf[1]);
        } else {
            vtx.uv = glm::vec2(0.f, 0.f);
        }

        bboxMin = glm::min(bboxMin, vtx.pos);
        bboxMax = glm::max(bboxMax, vtx.pos);
        verts.push_back(vtx);
    }

    // ── 인덱스 생성 ──────────────────────────────────────────────────────────
    uint32_t idxStart = static_cast<uint32_t>(inds.size());

    if (prim.indices >= 0) {
        const tinygltf::Accessor&   idxAcc  = model.accessors[prim.indices];
        const tinygltf::BufferView& idxView = model.bufferViews[idxAcc.bufferView];
        const uint8_t* idxPtr = model.buffers[idxView.buffer].data.data()
                              + idxView.byteOffset + idxAcc.byteOffset;
        int idxStride = idxAcc.ByteStride(idxView);

        for (int ii = 0; ii < static_cast<int>(idxAcc.count); ++ii) {
            const uint8_t* p = idxPtr + ii * idxStride;
            uint32_t idx = 0;
            switch (idxAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                idx = *reinterpret_cast<const uint32_t*>(p); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                idx = *reinterpret_cast<const uint16_t*>(p); break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                idx = *p; break;
            }
            inds.push_back(vertBase + idx);
        }
    } else {
        for (uint32_t i = 0; i < static_cast<uint32_t>(posAcc.count); ++i)
            inds.push_back(vertBase + i);
    }

    uint32_t idxCount = static_cast<uint32_t>(inds.size()) - idxStart;
    if (idxCount == 0) return;

    if (nrmPtr)
        alignTriangleWindingToNormals(verts, inds, idxStart, idxCount);

    // ── DrawObject 생성 ──────────────────────────────────────────────────────
    DrawObject obj{};
    obj.indexStart            = idxStart;
    obj.indexCount            = idxCount;
    obj.push.model            = worldTransform;
    obj.push.baseColor        = glm::vec4(1.f);
    obj.push.shininess        = 32.f;
    obj.push.specularStrength = 0.3f;
    obj.push.reflectStrength  = 0.f;
    obj.push.textureIndex     = static_cast<float>(texIdx);
    obj.textureIndex          = texIdx;
    obj.instanceGroupId       = -1;
    obj.twoSided              = twoSided;
    obj.reverseFrontFace      = (determinant3x3(worldTransform) < 0.f);

    computeBoundSphereLocal(obj, bboxMin, bboxMax);
    objects.push_back(obj);
}

// ─────────────────────────────────────────────────────────────────────────────
//  traverseNode
//  GLTF 씬 그래프를 재귀 순회한다.
// ─────────────────────────────────────────────────────────────────────────────
static void traverseNode(const tinygltf::Model&              model,
                          int                                  nodeIdx,
                          const glm::mat4&                     parentTransform,
                          const std::unordered_map<int, int>&  texIndexMap,
                          std::vector<Vertex>&                 verts,
                          std::vector<uint32_t>&               inds,
                          std::vector<DrawObject>&             objects)
{
    const tinygltf::Node& node = model.nodes[nodeIdx];
    glm::mat4 worldTransform = parentTransform * nodeLocalTransform(node);

    if (node.mesh >= 0) {
        const tinygltf::Mesh& mesh = model.meshes[node.mesh];
        for (const auto& prim : mesh.primitives)
            processPrimitive(model, prim, worldTransform, texIndexMap, verts, inds, objects);
    }

    for (int child : node.children)
        traverseNode(model, child, worldTransform, texIndexMap, verts, inds, objects);
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadGLTFScene  – 공개 진입점
// ─────────────────────────────────────────────────────────────────────────────
bool loadGLTFScene(const std::string&           path,
                   std::vector<Vertex>&          verts,
                   std::vector<uint32_t>&        inds,
                   std::vector<DrawObject>&      objects,
                   GLTFSceneDesc&                sceneDesc,
                   std::vector<GLTFTextureData>& textures)
{
    tinygltf::Model    model;
    tinygltf::TinyGLTF loader;
    std::string        err, warn;

    // 확장자로 바이너리(.glb) 와 텍스트(.gltf) 구분
    auto dotPos = path.find_last_of('.');
    std::string ext = (dotPos != std::string::npos) ? path.substr(dotPos + 1) : "";
    for (auto& c : ext) c = static_cast<char>(tolower(c));

    bool ok = (ext == "glb")
            ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
            : loader.LoadASCIIFromFile (&model, &err, &warn, path);

    if (!warn.empty()) std::cerr << "[GLTFLoader] Warning: " << warn << "\n";
    if (!ok) {
        std::cerr << "[GLTFLoader] Failed to load: " << path << "\n";
        if (!err.empty()) std::cerr << "  " << err << "\n";
        return false;
    }

    // ── 씬 이름 ──────────────────────────────────────────────────────────────
    if (!model.scenes.empty() && !model.scenes[0].name.empty())
        sceneDesc.name = model.scenes[0].name;
    else {
        auto slash = path.find_last_of("/\\");
        sceneDesc.name = (slash != std::string::npos) ? path.substr(slash + 1) : path;
    }

    // ── GLTF 카메라 노드에서 초기 위치 추출 ──────────────────────────────────
    for (const auto& node : model.nodes) {
        if (node.camera >= 0 && !node.translation.empty()) {
            sceneDesc.cameraPos = glm::vec3(
                static_cast<float>(node.translation[0]),
                static_cast<float>(node.translation[1]),
                static_cast<float>(node.translation[2]));
            break;
        }
    }

    // ── 텍스처 디코딩 ─────────────────────────────────────────────────────────
    // tinygltf + stb_image 가 GLB 내장 이미지를 RGBA8 로 자동 디코딩한다.
    // model.images[i].image 에 RGBA8 픽셀 데이터가 들어있다.
    std::unordered_map<int, int> texIndexMap; // GLTF image index → 로컬 인덱스
    for (int i = 0; i < static_cast<int>(model.images.size()); ++i) {
        const auto& img = model.images[i];
        if (!img.image.empty() && img.width > 0 && img.height > 0) {
            texIndexMap[i] = static_cast<int>(textures.size());
            GLTFTextureData td;
            td.width  = img.width;
            td.height = img.height;
            td.pixels.assign(img.image.begin(), img.image.end());
            textures.push_back(std::move(td));
        }
    }

    if (!textures.empty())
        std::cout << "[GLTFLoader] Decoded " << textures.size() << " texture(s)\n";

    // ── 씬 그래프 순회 ────────────────────────────────────────────────────────
    int sceneIdx = (model.defaultScene >= 0) ? model.defaultScene : 0;
    if (sceneIdx >= static_cast<int>(model.scenes.size())) {
        std::cerr << "[GLTFLoader] No scenes found in: " << path << "\n";
        return false;
    }

    size_t objsBefore = objects.size();
    for (int rootNode : model.scenes[sceneIdx].nodes)
        traverseNode(model, rootNode, glm::mat4(1.f), texIndexMap, verts, inds, objects);

    std::cout << "[GLTFLoader] Loaded '" << sceneDesc.name << "' from " << path
              << "  (" << (objects.size() - objsBefore) << " objects, "
              << verts.size() << " vertices)\n";
    return true;
}
