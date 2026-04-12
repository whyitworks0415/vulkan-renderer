#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include "OBJLoader.h"
#include "GLTFLoader.h"   // GLTFSceneDesc, GLTFTextureData
#include "VulkanApp.h"    // Vertex, DrawObject, PushConstants

// stb_image 구현체는 stb_image_impl.cpp 에서 정의됨 — 여기서는 선언만 사용
#include <stb_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  헬퍼: 로컬 공간 AABB → 바운딩 구 (GLTFLoader 와 동일 알고리즘)
// ─────────────────────────────────────────────────────────────────────────────
static void computeBound(DrawObject&      obj,
                         const glm::vec3& bmin,
                         const glm::vec3& bmax)
{
    const glm::mat4& M = obj.push.model;
    glm::vec3 corners[8]; int k = 0;
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

// ─────────────────────────────────────────────────────────────────────────────
//  헬퍼: stb_image 로 텍스처 파일 → GLTFTextureData
//  이미 로드된 경로는 texPathToIndex 맵에서 재사용한다.
// ─────────────────────────────────────────────────────────────────────────────
static int loadTexture(const std::string& absPath,
                       std::vector<GLTFTextureData>&           textures,
                       std::unordered_map<std::string, int>&   texPathToIndex)
{
    auto it = texPathToIndex.find(absPath);
    if (it != texPathToIndex.end()) return it->second;

    int w = 0, h = 0, ch = 0;
    unsigned char* data = stbi_load(absPath.c_str(), &w, &h, &ch, 4);
    if (!data) {
        std::cerr << "[OBJLoader] stbi_load failed: " << absPath << "\n";
        return -1;
    }

    GLTFTextureData td;
    td.width  = w;
    td.height = h;
    td.pixels.assign(data, data + (size_t)w * h * 4);
    stbi_image_free(data);

    int idx = (int)textures.size();
    textures.push_back(std::move(td));
    texPathToIndex[absPath] = idx;
    return idx;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadOBJScene
// ─────────────────────────────────────────────────────────────────────────────
bool loadOBJScene(const std::string&           path,
                  std::vector<Vertex>&          verts,
                  std::vector<uint32_t>&        inds,
                  std::vector<DrawObject>&      objects,
                  GLTFSceneDesc&                sceneDesc,
                  std::vector<GLTFTextureData>& textures)
{
    tinyobj::ObjReaderConfig cfg;
    cfg.mtl_search_path = std::filesystem::path(path).parent_path().string();
    cfg.triangulate     = true;

    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(path, cfg)) {
        std::cerr << "[OBJLoader] Parse error: " << reader.Error() << "\n";
        return false;
    }
    if (!reader.Warning().empty())
        std::cerr << "[OBJLoader] Warning: " << reader.Warning() << "\n";

    const auto& attrib    = reader.GetAttrib();
    const auto& shapes    = reader.GetShapes();
    const auto& materials = reader.GetMaterials();

    if (shapes.empty()) {
        std::cerr << "[OBJLoader] No shapes in: " << path << "\n";
        return false;
    }

    // 씬 이름 = 파일명 (확장자 제외)
    sceneDesc.name = std::filesystem::path(path).stem().string();

    // 텍스처 경로 캐시 (globalTextures 인덱스는 textures.size() 기준 오프셋 필요)
    std::unordered_map<std::string, int> texPathToIndex;
    const int texOffset = (int)textures.size(); // 이미 로드된 텍스처 수

    std::string baseDir = std::filesystem::path(path).parent_path().string();
    if (!baseDir.empty() && baseDir.back() != '/' && baseDir.back() != '\\')
        baseDir += '/';

    // 재질별 텍스처 인덱스 미리 로드
    std::vector<int> matTexIdx(materials.size(), -1);
    for (int mi = 0; mi < (int)materials.size(); ++mi) {
        const auto& mat = materials[mi];
        if (!mat.diffuse_texname.empty()) {
            std::string texPath = mat.diffuse_texname;
            // 절대 경로가 아닌 경우 .obj 파일 기준 상대 경로로 처리
            if (!std::filesystem::path(texPath).is_absolute())
                texPath = baseDir + texPath;
            int localIdx = loadTexture(texPath, textures, texPathToIndex);
            matTexIdx[mi] = (localIdx >= 0) ? (localIdx) : -1;
        }
    }

    // ── shape(=오브젝트) 별로 DrawObject 생성 ─────────────────────────────────
    // 같은 재질 내에서도 face마다 다른 재질을 쓸 수 있으므로,
    // 재질별로 face를 묶어 같은 shape 내에서도 DrawObject를 분리한다.
    // (대부분의 .obj는 shape당 재질 1개이므로 결과적으로 shape = DrawObject)

    for (const auto& shape : shapes) {
        // shape 내 face들을 재질 ID 기준으로 그룹화
        std::unordered_map<int, std::vector<int>> matToFaces; // matId → face 인덱스 목록
        for (int fi = 0; fi < (int)shape.mesh.material_ids.size(); ++fi) {
            int mid = shape.mesh.material_ids[fi];
            if (mid < 0) mid = -1; // 재질 없음
            matToFaces[mid].push_back(fi);
        }

        for (auto& [matId, faceList] : matToFaces) {
            // 이 그룹의 버텍스·인덱스를 로컬로 수집 (중복 버텍스 병합)
            std::unordered_map<uint64_t, uint32_t> indexCache;
            std::vector<Vertex>   localVerts;
            std::vector<uint32_t> localInds;

            glm::vec3 bmin{ 1e9f}, bmax{-1e9f};

            // 재질 색상·질감
            glm::vec3 diffuseColor{1.f};
            float     shininess     = 32.f;
            float     specStr       = 0.3f;
            int       texIdx        = -1;

            if (matId >= 0 && matId < (int)materials.size()) {
                const auto& mat = materials[matId];
                diffuseColor = { mat.diffuse[0], mat.diffuse[1], mat.diffuse[2] };
                if (mat.shininess > 0.f) shininess = mat.shininess;
                float specLum = (mat.specular[0] + mat.specular[1] + mat.specular[2]) / 3.f;
                specStr = glm::clamp(specLum, 0.f, 1.f);
                texIdx  = matTexIdx[matId];
            }

            for (int fi : faceList) {
                int faceVerts = 3; // triangulated
                int idxBase   = fi * 3;
                for (int v = 0; v < faceVerts; ++v) {
                    const auto& idx = shape.mesh.indices[idxBase + v];

                    // 64-bit 키: (vertex_index << 32 | normal_index << 16 | texcoord_index)
                    // 각 인덱스는 최대 16비트로 가정 (실제로는 32비트지만 대부분 충분)
                    uint64_t key = (uint64_t)(uint32_t)idx.vertex_index
                                 | ((uint64_t)(uint32_t)idx.normal_index   << 20)
                                 | ((uint64_t)(uint32_t)idx.texcoord_index << 40);

                    auto it = indexCache.find(key);
                    if (it != indexCache.end()) {
                        localInds.push_back(it->second);
                    } else {
                        Vertex vt{};
                        int vi = idx.vertex_index * 3;
                        vt.pos = { attrib.vertices[vi],
                                   attrib.vertices[vi + 1],
                                   attrib.vertices[vi + 2] };

                        if (idx.normal_index >= 0) {
                            int ni = idx.normal_index * 3;
                            vt.normal = { attrib.normals[ni],
                                          attrib.normals[ni + 1],
                                          attrib.normals[ni + 2] };
                        }

                        if (idx.texcoord_index >= 0) {
                            int ti = idx.texcoord_index * 2;
                            vt.uv = { attrib.texcoords[ti],
                                      1.f - attrib.texcoords[ti + 1] }; // Y 반전 (OpenGL→Vulkan)
                        }

                        // 버텍스 컬러: attrib.colors 있으면 사용, 없으면 diffuse
                        if (!attrib.colors.empty()) {
                            int ci = idx.vertex_index * 3;
                            vt.color = { attrib.colors[ci],
                                         attrib.colors[ci + 1],
                                         attrib.colors[ci + 2] };
                        } else {
                            vt.color = diffuseColor;
                        }

                        bmin = glm::min(bmin, vt.pos);
                        bmax = glm::max(bmax, vt.pos);

                        uint32_t newIdx = (uint32_t)localVerts.size();
                        indexCache[key] = newIdx;
                        localVerts.push_back(vt);
                        localInds.push_back(newIdx);
                    }
                }
            }

            if (localVerts.empty()) continue;

            // 법선이 없는 face → 플랫 법선 계산
            for (size_t i = 0; i + 2 < localInds.size(); i += 3) {
                Vertex& a = localVerts[localInds[i]];
                Vertex& b = localVerts[localInds[i + 1]];
                Vertex& c = localVerts[localInds[i + 2]];
                if (a.normal == glm::vec3{} && b.normal == glm::vec3{} && c.normal == glm::vec3{}) {
                    glm::vec3 n = glm::normalize(glm::cross(b.pos - a.pos, c.pos - a.pos));
                    a.normal = b.normal = c.normal = n;
                }
            }

            // 글로벌 버퍼에 추가
            uint32_t idxStart = (uint32_t)inds.size();
            uint32_t vtxBase  = (uint32_t)verts.size();
            for (auto& lv : localVerts) verts.push_back(lv);
            for (auto  li : localInds)  inds.push_back(vtxBase + li);

            // DrawObject 구성
            DrawObject obj{};
            obj.indexStart = idxStart;
            obj.indexCount = (uint32_t)localInds.size();
            obj.push.model = glm::mat4(1.f);
            obj.push.baseColor = glm::vec4(diffuseColor, 1.f);
            obj.push.shininess        = shininess;
            obj.push.specularStrength = specStr;
            obj.push.reflectStrength  = 0.2f;
            obj.push.textureIndex     = (texIdx >= 0) ? (float)(texIdx) : -1.f;
            obj.push.emissive         = glm::vec4(0.f);
            obj.textureIndex          = texIdx;
            obj.twoSided              = false;

            computeBound(obj, bmin, bmax);
            objects.push_back(obj);
        }
    }

    if (objects.empty()) {
        std::cerr << "[OBJLoader] No drawable objects extracted from: " << path << "\n";
        return false;
    }

    // 씬 전체 AABB 기반 카메라 초기 위치 추정
    glm::vec3 sceneMin{ 1e9f}, sceneMax{-1e9f};
    for (auto& obj : objects) {
        sceneMin = glm::min(sceneMin, obj.boundCenter - glm::vec3(obj.boundRadius));
        sceneMax = glm::max(sceneMax, obj.boundCenter + glm::vec3(obj.boundRadius));
    }
    glm::vec3 sceneCenter = (sceneMin + sceneMax) * 0.5f;
    float     sceneSize   = glm::length(sceneMax - sceneMin);
    sceneDesc.cameraPos   = sceneCenter + glm::vec3(0.f, sceneSize * 0.3f, -sceneSize * 0.7f);
    sceneDesc.cameraYaw   = 90.f;
    sceneDesc.cameraPitch = -15.f;

    std::cout << "[OBJLoader] Loaded " << path
              << "  objects=" << objects.size()
              << "  verts=" << verts.size()
              << "  tris=" << inds.size() / 3 << "\n";
    return true;
}
