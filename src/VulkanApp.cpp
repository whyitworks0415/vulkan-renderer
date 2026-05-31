#include "VulkanApp.h"
#include "MeshLoader.h"
#include "GLTFLoader.h"
#include "OBJLoader.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

// 상수와 검증 레이어 설정
static const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
static const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef ENABLE_VALIDATION_LAYERS
constexpr bool kEnableValidation = true;
#else
constexpr bool kEnableValidation = false;
#endif

// Vulkan 검증 메시지 콜백
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::cerr << "[Vulkan] " << data->pMessage << "\n";
    return VK_FALSE;
}

static VkResult CreateDebugMessenger(VkInstance inst,
    const VkDebugUtilsMessengerCreateInfoEXT* ci,
    VkDebugUtilsMessengerEXT* out)
{
    auto fn = (PFN_vkCreateDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(inst, "vkCreateDebugUtilsMessengerEXT");
    return fn ? fn(inst, ci, nullptr, out) : VK_ERROR_EXTENSION_NOT_PRESENT;
}
static void DestroyDebugMessenger(VkInstance inst, VkDebugUtilsMessengerEXT m)
{
    auto fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
              vkGetInstanceProcAddr(inst, "vkDestroyDebugUtilsMessengerEXT");
    if (fn) fn(inst, m, nullptr);
}

// 버텍스 입력 레이아웃
VkVertexInputBindingDescription Vertex::getBindingDesc() {
    return {0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX};
}
std::array<VkVertexInputAttributeDescription, 4> Vertex::getAttrDescs() {
    return {{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal)},
        {2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, uv)},
    }};
}

// 절차적 씬 지오메트리 생성
namespace {


static std::string quoteCmdArg(const std::string& arg)
{
    bool needsQuote = arg.empty();
    for (char c : arg) {
        if (std::isspace(static_cast<unsigned char>(c)) || c == '"') {
            needsQuote = true;
            break;
        }
    }
    if (!needsQuote) return arg;

    std::string out = "\"";
    for (char c : arg) {
        if (c == '"') out += "\\\"";
        else out += c;
    }
    out += "\"";
    return out;
}

static std::string pythonStringLiteral(const std::filesystem::path& path)
{
    std::string s = path.generic_string();
    std::string out = "\"";
    for (char c : s) {
        if (c == '\\' || c == '"') out += '\\';
        out += c;
    }
    out += "\"";
    return out;
}

static std::string findBlenderExecutable()
{
    namespace fs = std::filesystem;

    for (const char* envName : {"BLENDER_EXE", "BLENDER_PATH"}) {
        if (const char* env = std::getenv(envName)) {
            fs::path p(env);
            if (!p.empty() && fs::exists(p))
                return p.string();
        }
    }

#ifdef _WIN32
    std::vector<fs::path> candidates;
    fs::path root("C:/Program Files/Blender Foundation");
    std::error_code ec;
    if (fs::exists(root, ec)) {
        for (const auto& entry : fs::directory_iterator(root, ec)) {
            if (!entry.is_directory()) continue;
            fs::path exe = entry.path() / "blender.exe";
            if (fs::exists(exe, ec)) candidates.push_back(exe);
        }
    }
    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end());
        return candidates.back().string();
    }
#endif

    return "blender";
}

static bool convertBlendToGlb(const std::string& blendPath, std::string& outGlbPath)
{
    namespace fs = std::filesystem;
    std::error_code ec;

    fs::path input(blendPath);
    fs::path output = input;
    output.replace_extension(".glb");
    outGlbPath = output.string();

    const bool outputFresh =
        fs::exists(output, ec) &&
        fs::last_write_time(output, ec) >= fs::last_write_time(input, ec);
    if (outputFresh)
        return true;

    fs::path script = output;
    script.replace_extension(".export_glb.py");

    {
        std::ofstream py(script);
        if (!py.is_open()) {
            std::cerr << "[BlendLoader] Cannot write export script: "
                      << script.string() << "\n";
            return false;
        }

        py << "import bpy\n";
        py << "out_path = " << pythonStringLiteral(output) << "\n";
        py << "try:\n";
        py << "    bpy.ops.wm.addon_enable(module='io_scene_gltf2')\n";
        py << "except Exception:\n";
        py << "    pass\n";
        py << "kwargs = dict(filepath=out_path, export_format='GLB', use_selection=False)\n";
        py << "try:\n";
        py << "    bpy.ops.export_scene.gltf(**kwargs, export_apply=True)\n";
        py << "except TypeError:\n";
        py << "    bpy.ops.export_scene.gltf(**kwargs)\n";
    }

    std::string blender = findBlenderExecutable();
    std::string cmd = quoteCmdArg(blender)
                    + " --background " + quoteCmdArg(input.string())
                    + " --python " + quoteCmdArg(script.string());

    std::cout << "[BlendLoader] Exporting " << input.string()
              << " -> " << output.string() << "\n";
    int rc = std::system(cmd.c_str());
    fs::remove(script, ec);

    if (rc != 0 || !fs::exists(output, ec)) {
        std::cerr << "[BlendLoader] Blender export failed. Install Blender or set BLENDER_EXE/BLENDER_PATH.\n";
        return false;
    }

    return true;
}

// Y-up 원점 기준 축 정렬 박스를 만든다.
// hw/hh/hd는 각 축의 반 길이이며, 정점 순서는 Vulkan 전면 판정에 맞춘다.
// 각 면의 정점 순서가 바깥쪽 법선을 만들도록 유지한다.
static void addBoxFace(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                       glm::vec3 n,
                       glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3,
                       glm::vec3 color)
{
    uint32_t base = static_cast<uint32_t>(verts.size());
    verts.push_back({v0, n, color});
    verts.push_back({v1, n, color});
    verts.push_back({v2, n, color});
    verts.push_back({v3, n, color});
    inds.insert(inds.end(), {base, base+1, base+2,  base, base+2, base+3});
}

static MeshRange addBox(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                        float hw, float hh, float hd, glm::vec3 color)
{
    uint32_t iBase = static_cast<uint32_t>(inds.size());

    // 각 면의 정점 순서가 바깥쪽 법선을 만들도록 유지한다.
    //  +X 면 법선
    addBoxFace(verts,inds,{1,0,0},   {hw,hh,hd},{hw,-hh,hd},{hw,-hh,-hd},{hw,hh,-hd}, color);
    // -X 면 법선
    addBoxFace(verts,inds,{-1,0,0},  {-hw,hh,-hd},{-hw,-hh,-hd},{-hw,-hh,hd},{-hw,hh,hd}, color);
    //  +Y 면 법선
    addBoxFace(verts,inds,{0,1,0},   {-hw,hh,hd},{hw,hh,hd},{hw,hh,-hd},{-hw,hh,-hd}, color);
    // -Y 면 법선
    addBoxFace(verts,inds,{0,-1,0},  {-hw,-hh,-hd},{hw,-hh,-hd},{hw,-hh,hd},{-hw,-hh,hd}, color);
    //  +Z 면 법선
    addBoxFace(verts,inds,{0,0,1},   {hw,hh,hd},{-hw,hh,hd},{-hw,-hh,hd},{hw,-hh,hd}, color);
    // -Z 면 법선
    addBoxFace(verts,inds,{0,0,-1},  {-hw,hh,-hd},{hw,hh,-hd},{hw,-hh,-hd},{-hw,-hh,-hd}, color);

    MeshRange r{};
    r.indexStart = iBase;
    r.indexCount = static_cast<uint32_t>(inds.size()) - iBase;
    r.bboxMin = {-hw, -hh, -hd};
    r.bboxMax = { hw,  hh,  hd};
    return r;
}

static MeshRange addFloorMesh(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                              float size, int divs, glm::vec3 color)
{
    uint32_t iBase = static_cast<uint32_t>(inds.size());
    float half = size * 0.5f;
    int n = std::max(1, divs);
    float step = size / static_cast<float>(n);

    for (int z = 0; z < n; ++z) {
        for (int x = 0; x < n; ++x) {
            float x0 = -half + x * step;
            float x1 = x0 + step;
            float z0 = -half + z * step;
            float z1 = z0 + step;
            uint32_t base = static_cast<uint32_t>(verts.size());
            verts.push_back({{x0, 0.0f, z0}, {0, 1, 0}, color, {0, 0}});
            verts.push_back({{x1, 0.0f, z0}, {0, 1, 0}, color, {1, 0}});
            verts.push_back({{x1, 0.0f, z1}, {0, 1, 0}, color, {1, 1}});
            verts.push_back({{x0, 0.0f, z1}, {0, 1, 0}, color, {0, 1}});
            inds.insert(inds.end(), {base, base + 1, base + 2, base, base + 2, base + 3});
        }
    }

    MeshRange r{};
    r.indexStart = iBase;
    r.indexCount = static_cast<uint32_t>(inds.size()) - iBase;
    r.bboxMin = {-half, 0.0f, -half};
    r.bboxMax = { half, 0.0f,  half};
    return r;
}

// 절차적 박스 메시에서 DrawObject를 구성한다.
// center는 박스 중심의 월드 좌표다.
static DrawObject makeBoxObj(const MeshRange& mesh, glm::vec3 center,
                              float shininess, float specStr, float reflStr,
                              glm::vec4 baseColor = glm::vec4(1.f))
{
    DrawObject obj{};
    obj.indexStart            = mesh.indexStart;
    obj.indexCount            = mesh.indexCount;
    obj.push.model            = glm::translate(glm::mat4(1.f), center);
    obj.push.baseColor        = baseColor;
    obj.push.shininess        = shininess;
    obj.push.specularStrength = specStr;
    obj.push.reflectStrength  = reflStr;
    // 박스를 감싸는 외접구를 바운딩 구로 사용한다.
    float hw = (mesh.bboxMax.x - mesh.bboxMin.x) * 0.5f;
    float hh = (mesh.bboxMax.y - mesh.bboxMin.y) * 0.5f;
    float hd = (mesh.bboxMax.z - mesh.bboxMin.z) * 0.5f;
    obj.boundCenter = center;
    obj.boundRadius = std::sqrt(hw*hw + hh*hh + hd*hd);
    return obj;
}

// 원점 중심의 절차적 UV 구를 만든다.
// rings는 위도 분할 수, segments는 경도 분할 수다.
// 외부에서 볼 때 CCW가 되도록 삼각형 순서를 만든다.
static MeshRange makeProcSphere(std::vector<Vertex>& verts, std::vector<uint32_t>& inds,
                                float radius, int rings, int segments, glm::vec3 color)
{
    const float PI = std::acos(-1.0f);
    uint32_t vBase = static_cast<uint32_t>(verts.size());
    uint32_t iBase = static_cast<uint32_t>(inds.size());

    for (int r = 0; r <= rings; ++r) {
        float phi = PI * r / rings; // 북극점
        float y   = radius * std::cos(phi);
        float rr  = radius * std::sin(phi); // XZ 평면에서의 링 반지름
        for (int s = 0; s <= segments; ++s) {
            float theta = 2.f * PI * s / segments;
            glm::vec3 pos = {rr * std::cos(theta), y, rr * std::sin(theta)};
            glm::vec3 n   = (radius > 1e-6f) ? glm::normalize(pos) : glm::vec3(0,1,0);
            if (r == 0)     n = {0.f,  1.f, 0.f}; // 북극점
            if (r == rings) n = {0.f, -1.f, 0.f}; // 남극점
            verts.push_back({pos, n, color});
        }
    }

    for (int r = 0; r < rings; ++r) {
        for (int s = 0; s < segments; ++s) {
            uint32_t v0 = vBase +  r      * (segments + 1) + s;
            uint32_t v1 = vBase +  r      * (segments + 1) + s + 1;
            uint32_t v2 = vBase + (r + 1) * (segments + 1) + s;
            uint32_t v3 = vBase + (r + 1) * (segments + 1) + s + 1;
            // 각 면의 정점 순서가 바깥쪽 법선을 만들도록 유지한다.
            inds.push_back(v0); inds.push_back(v1); inds.push_back(v2);
            inds.push_back(v1); inds.push_back(v3); inds.push_back(v2);
        }
    }

    MeshRange mr{};
    mr.indexStart = iBase;
    mr.indexCount = static_cast<uint32_t>(inds.size()) - iBase;
    mr.bboxMin = {-radius, -radius, -radius};
    mr.bboxMax = { radius,  radius,  radius};
    return mr;
}

} // 익명 namespace 종료

bool VulkanApp::sampleObjectFrontFaceIsClockwise(const DrawObject& obj,
                                                 const Camera&     cam,
                                                 bool&             outClockwise) const
{
    if (indices.empty() || vertices.empty() || obj.indexCount < 3) return false;

    glm::mat4 view = cam.getViewMatrix();
    glm::mat4 proj = glm::perspective(glm::radians(60.f),
                                      scExtent.width / (float)scExtent.height,
                                      0.1f, getFarPlane());
    proj[1][1] *= -1.f;
    glm::mat4 vp = proj * view;

    int cwCount  = 0;
    int ccwCount = 0;
    int samples  = 0;

    for (uint32_t i = 0; i + 2 < obj.indexCount && samples < 128; i += 3) {
        uint32_t ia = indices[obj.indexStart + i + 0];
        uint32_t ib = indices[obj.indexStart + i + 1];
        uint32_t ic = indices[obj.indexStart + i + 2];
        if (ia >= vertices.size() || ib >= vertices.size() || ic >= vertices.size()) continue;

        glm::vec3 wa = glm::vec3(obj.push.model * glm::vec4(vertices[ia].pos, 1.f));
        glm::vec3 wb = glm::vec3(obj.push.model * glm::vec4(vertices[ib].pos, 1.f));
        glm::vec3 wc = glm::vec3(obj.push.model * glm::vec4(vertices[ic].pos, 1.f));

        glm::vec3 center = (wa + wb + wc) / 3.f;
        glm::mat3 normalM = glm::transpose(glm::inverse(glm::mat3(obj.push.model)));
        glm::vec3 avgN = vertices[ia].normal + vertices[ib].normal + vertices[ic].normal;
        glm::vec3 viewDir = cam.position - center;
        if (glm::dot(avgN, avgN) > 1e-10f) {
            glm::vec3 worldN = glm::normalize(normalM * avgN);
            if (glm::dot(worldN, viewDir) <= 1e-5f) continue;
        } else {
            glm::vec3 faceN = glm::cross(wb - wa, wc - wa);
            if (glm::dot(faceN, faceN) <= 1e-10f) continue;
            if (glm::dot(faceN, viewDir) <= 1e-5f) continue;
        }

        glm::vec4 ca = vp * glm::vec4(wa, 1.f);
        glm::vec4 cb = vp * glm::vec4(wb, 1.f);
        glm::vec4 cc = vp * glm::vec4(wc, 1.f);
        if (ca.w <= 1e-5f || cb.w <= 1e-5f || cc.w <= 1e-5f) continue;

        glm::vec2 aNdc = glm::vec2(ca) / ca.w;
        glm::vec2 bNdc = glm::vec2(cb) / cb.w;
        glm::vec2 cNdc = glm::vec2(cc) / cc.w;

        // Vulkan 화면 좌표계의 전면 판정과 맞춰 winding을 판별한다.
        // Y-down 화면 좌표에서 signed area가 양수이면 시계 방향이다.
        glm::vec2 a((aNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (aNdc.y * 0.5f + 0.5f) * scExtent.height);
        glm::vec2 b((bNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (bNdc.y * 0.5f + 0.5f) * scExtent.height);
        glm::vec2 c((cNdc.x * 0.5f + 0.5f) * scExtent.width,
                    (cNdc.y * 0.5f + 0.5f) * scExtent.height);
        float area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
        if (std::abs(area2) <= 1e-7f) continue;

        if (area2 > 0.f) ++cwCount;
        else             ++ccwCount;
        ++samples;
    }

    if (samples == 0) return false;
    outClockwise = cwCount >= ccwCount;
    return true;
}

void VulkanApp::refreshFrontFaceHints(const Camera& cam)
{
    for (auto& obj : drawObjects) {
        if (obj.twoSided) {
            obj.reverseFrontFace = false;
            continue;
        }

        bool clockwise = false;
        if (sampleObjectFrontFaceIsClockwise(obj, cam, clockwise)) {
            obj.reverseFrontFace = !clockwise; // 기본 파이프라인은 시계 방향을 전면으로 본다.
        } else {
            obj.reverseFrontFace = glm::determinant(glm::mat3(obj.push.model)) < 0.f;
        }
    }
}


void VulkanApp::buildScene() {
    using Clock = std::chrono::steady_clock;
    auto elapsed = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    // 확장자에 따라 적절한 로더로 씬 파일 로드
    auto t_parse = Clock::now();
    GLTFSceneDesc gDesc;
    sceneTextures.clear();

    // 확장자 추출 (소문자)
    auto extOf = [](const std::string& p) {
        auto dot = p.rfind('.');
        if (dot == std::string::npos) return std::string{};
        std::string e = p.substr(dot);
        std::transform(e.begin(), e.end(), e.begin(), ::tolower);
        return e;
    };
    std::string ext = extOf(currentMapFile);

    bool loadOk = false;
    if (ext == ".scene") {
        SceneDesc desc;
        loadOk = loadScene(currentMapFile, desc);
        if (loadOk) {
            gDesc.name        = desc.name;
            gDesc.cameraPos   = desc.cameraPos;
            gDesc.cameraYaw   = desc.cameraYaw;
            gDesc.cameraPitch = desc.cameraPitch;

            struct SceneMeshRuntime {
                SceneMeshDesc desc;
                LoadedMesh    mesh;
            };
            std::vector<SceneMeshRuntime> sceneMeshes;

            auto findMesh = [&](const std::string& id) -> SceneMeshRuntime* {
                for (auto& m : sceneMeshes)
                    if (m.desc.id == id) return &m;
                return nullptr;
            };

            for (const auto& md : desc.meshes) {
                SceneMeshRuntime m{};
                m.desc = md;
                m.mesh = loadMeshDetailed(md.path, vertices, indices, md.color);
                sceneMeshes.push_back(std::move(m));
            }

            auto partGroupId = [](int baseGroup, int partIdx) {
                return (baseGroup >= 0) ? baseGroup * 1024 + partIdx : -1;
            };

            auto appendSceneObject = [&](const SceneMeshRuntime& mesh,
                                         const MeshRange& range,
                                         const glm::mat4& model,
                                         int instanceGroupId) {
                DrawObject obj{};
                obj.indexStart            = range.indexStart;
                obj.indexCount            = range.indexCount;
                obj.push.model            = model;
                obj.push.baseColor        = glm::vec4(1.0f, 1.0f, 1.0f, mesh.desc.alpha);
                obj.push.shininess        = mesh.desc.shininess;
                obj.push.specularStrength = mesh.desc.specular;
                obj.push.reflectStrength  = mesh.desc.reflect;
                obj.push.textureIndex     = -1.0f;
                obj.push.emissive         = glm::vec4(0.0f);
                obj.textureIndex          = -1;
                obj.instanceGroupId       = instanceGroupId;
                computeBoundSphere(obj, range.bboxMin, range.bboxMax);
                drawObjects.push_back(obj);
            };

            if (desc.hasFloor) {
                MeshRange floor = addFloorMesh(vertices, indices, desc.floorSize, desc.floorDivs,
                                               {0.48f, 0.52f, 0.50f});
                DrawObject obj{};
                obj.indexStart            = floor.indexStart;
                obj.indexCount            = floor.indexCount;
                obj.push.model            = glm::mat4(1.0f);
                obj.push.baseColor        = glm::vec4(1.0f);
                obj.push.shininess        = 24.0f;
                obj.push.specularStrength = 0.15f;
                obj.push.reflectStrength  = 0.0f;
                obj.push.textureIndex     = -1.0f;
                obj.push.emissive         = glm::vec4(0.0f);
                obj.textureIndex          = -1;
                obj.twoSided              = true;
                obj.skipOcclusion         = true;
                computeBoundSphere(obj, floor.bboxMin, floor.bboxMax);
                drawObjects.push_back(obj);
            }

            for (const auto& gd : desc.grids) {
                const SceneMeshRuntime* mesh = findMesh(gd.meshId);
                if (!mesh) continue;
                int half = gd.cells / 2;
                int c0 = 0, c1 = gd.cells - 1, r0 = 0, r1 = gd.cells - 1;
                if (gd.hasRange) {
                    c0 = gd.ranC0; c1 = gd.ranC1;
                    r0 = gd.ranR0; r1 = gd.ranR1;
                }

                for (int row = r0; row <= r1; ++row) {
                    for (int col = c0; col <= c1; ++col) {
                        if (gd.evenOnly && (col % 2 != 0 || row % 2 != 0)) continue;
                        if (gd.hasExclude && col >= gd.exC0 && col <= gd.exC1 &&
                            row >= gd.exR0 && row <= gd.exR1) continue;
                        if (gd.hasHollow && col >= gd.holC0 && col <= gd.holC1 &&
                            row >= gd.holR0 && row <= gd.holR1) continue;

                        glm::vec3 pos((col - half) * gd.cellSize, 0.0f,
                                      (row - half) * gd.cellSize);
                        for (int pi = 0; pi < (int)mesh->mesh.parts.size(); ++pi) {
                            const MeshRange& part = mesh->mesh.parts[pi];
                            glm::mat4 model = makeSTLModel(mesh->mesh.whole, gd.scale, pos);
                            appendSceneObject(*mesh, part, model, partGroupId(gd.instanceGroup, pi));
                        }
                    }
                }
            }

            for (const auto& od : desc.objects) {
                const SceneMeshRuntime* mesh = findMesh(od.meshId);
                if (!mesh) continue;
                for (int pi = 0; pi < (int)mesh->mesh.parts.size(); ++pi) {
                    const MeshRange& part = mesh->mesh.parts[pi];
                    glm::mat4 model = makeSTLModel(mesh->mesh.whole, od.scale, od.pos);
                    appendSceneObject(*mesh, part, model, partGroupId(od.instanceGroup, pi));
                }
            }
        }
    } else if (ext == ".obj") {
        loadOk = loadOBJScene(currentMapFile, vertices, indices, drawObjects, gDesc, sceneTextures);
    } else if (ext == ".blend") {
        std::string glbPath;
        if (convertBlendToGlb(currentMapFile, glbPath))
            loadOk = loadGLTFScene(glbPath, vertices, indices, drawObjects, gDesc, sceneTextures);
    } else {
        // .glb / .gltf (기본)
        loadOk = loadGLTFScene(currentMapFile, vertices, indices, drawObjects, gDesc, sceneTextures);
    }
    if (!loadOk)
        throw std::runtime_error("Failed to load scene: " + currentMapFile);
    lastLoadTiming.sceneParseMs = elapsed(t_parse);

    sceneLights     = gDesc.lights; // KHR_lights_punctual 조명 저장
    sceneLightDirty = true; // UBO 갱신 필요

    camera.position = gDesc.cameraPos;
    camera.yaw      = gDesc.cameraYaw;
    camera.pitch    = gDesc.cameraPitch;
    camera.syncTarget(); // 시네마틱 보간 타겟을 새 씬 위치에 다시 맞춤
    refreshFrontFaceHints(camera);

    // 오클루전 쿼리용 단위 박스 프록시를 만든다.
    occBBoxMesh = addBox(vertices, indices, 1.f, 1.f, 1.f, {0.f, 0.f, 0.f});

    // 오클루전 쿼리 결과와 안정화 상태를 초기화한다.
    resetOcclusionState();

    rebuildInstancingGroups();

    // 스트레스 배율 기준 복사본 저장 (씬 로드마다 리셋)
    baseDrawObjects = drawObjects;
    stressLevel     = 0;

    // 씬 전체 바운딩 구 계산 — 방향광 섀도맵 ortho 절두체를 씬에 맞춘다.
    if (!drawObjects.empty()) {
        glm::vec3 bmin( std::numeric_limits<float>::max());
        glm::vec3 bmax(-std::numeric_limits<float>::max());
        for (const DrawObject& o : drawObjects) {
            bmin = glm::min(bmin, o.boundCenter - glm::vec3(o.boundRadius));
            bmax = glm::max(bmax, o.boundCenter + glm::vec3(o.boundRadius));
        }
        sceneBoundsCenter = (bmin + bmax) * 0.5f;
        sceneBoundsRadius = glm::max(glm::length(bmax - sceneBoundsCenter), 1.0f);
    } else {
        sceneBoundsCenter = glm::vec3(0.0f);
        sceneBoundsRadius = 50.0f;
    }

    std::cout << "Scene built: "
              << drawObjects.size() << " draw objects, "
              << vertices.size()    << " vertices, "
              << indices.size() / 3 << " triangles\n";
}

void VulkanApp::rebuildInstancingGroups() {
    instGroupDefs.clear();
    std::map<std::tuple<int, uint32_t, uint32_t, int, bool, bool>, int> groupIdxMap;
    for (int i = 0; i < (int)drawObjects.size(); ++i) {
        const DrawObject& obj = drawObjects[i];
        int gid = obj.instanceGroupId;
        if (gid < 0) continue;
        auto key = std::make_tuple(gid, obj.indexStart, obj.indexCount,
                                   obj.textureIndex, obj.twoSided, obj.reverseFrontFace);
        auto it = groupIdxMap.find(key);
        if (it == groupIdxMap.end()) {
            groupIdxMap[key] = (int)instGroupDefs.size();
            InstGroupDef def;
            def.groupId    = gid;
            def.indexStart = obj.indexStart;
            def.indexCount = obj.indexCount;
            def.push       = obj.push;
            def.members.push_back(i);
            instGroupDefs.push_back(std::move(def));
        } else {
            instGroupDefs[it->second].members.push_back(i);
        }
    }
}

void VulkanApp::reloadScene() {
    using Clock = std::chrono::steady_clock;
    auto t_total = Clock::now();
    auto elapsed = [](Clock::time_point t0) {
        return std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
    };

    lastLoadTiming = {};
    lastLoadTiming.mapName = currentMapFile;

    std::string prevMapFile = currentMapFile;

    // 1단계: 새 CPU 씬 데이터를 먼저 구성한다.
    // 파일 파싱 중에는 기존 GPU 버퍼를 유지하고 CPU 배열만 새로 채운다.
    vertices.clear();
    indices.clear();
    drawObjects.clear();
    occBBoxMesh = {};

    bool buildOk = false;
    try {
        buildScene();
        buildOk = true;
    } catch (const std::exception& e) {
        std::cerr << "[reloadScene] Failed to load '" << currentMapFile
                  << "': " << e.what() << "\n";

        if (currentMapFile != prevMapFile) {
            std::cerr << "[reloadScene] Reverting to: " << prevMapFile << "\n";
            currentMapFile = prevMapFile;
            for (int i = 0; i < (int)availableMaps.size(); ++i)
                if (availableMaps[i] == prevMapFile) { currentMapIndex = i; break; }
            vertices.clear(); indices.clear(); drawObjects.clear();
            try {
                buildScene();
                buildOk = true;
            } catch (...) {
                std::cerr << "[reloadScene] Fallback scene also failed. Running empty.\n";
            }
        }
    }
    (void)buildOk;

    // 2단계: GPU 작업을 비운 뒤 이전 GPU 버퍼를 제거한다.
    // 진행 중인 프레임이 이전 버퍼를 참조할 수 있으므로 GPU idle을 기다린다.
    // 느린 CPU 파싱은 이미 끝났기 때문에 여기서는 남은 프레임 작업만 비운다.
    vkDeviceWaitIdle(device);

    vkDestroyBuffer(device, indexBuffer,  nullptr);
    vkFreeMemory(device,    indexBufferMemory,  nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device,    vertexBufferMemory, nullptr);
    indexBuffer        = VK_NULL_HANDLE;  indexBufferMemory  = VK_NULL_HANDLE;
    vertexBuffer       = VK_NULL_HANDLE;  vertexBufferMemory = VK_NULL_HANDLE;

    if (occlusionQueryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);
        occlusionQueryPool = VK_NULL_HANDLE;
        occQueryCount      = 0;
    }
    occWarmupFrames = 0;

    // 3단계: 새 씬 데이터를 한 번의 배치 전송으로 업로드한다.
    // createSceneBuffers()는 버텍스와 인덱스 복사를 하나의 커맨드 버퍼로 기록한다.
    // 별도 queue idle 두 번 대신 하나의 fence만 기다린다.
    auto t_upload = Clock::now();
    if (!vertices.empty() && !indices.empty())
        createSceneBuffers();
    else {
        if (!vertices.empty()) createVertexBuffer();
        if (!indices.empty())  createIndexBuffer();
    }
    lastLoadTiming.uploadMs = elapsed(t_upload);

    if (!drawObjects.empty()) createOcclusionQueryPool();
    resetOcclusionState();

    createTextureResources();

    lastLoadTiming.totalMs = elapsed(t_total);
    lastLoadTiming.valid   = true;

    printf("[Load] %-32s  total %6.1f ms  parse %6.1f ms  upload %5.1f ms\n",
           currentMapFile.c_str(),
           lastLoadTiming.totalMs,
           lastLoadTiming.sceneParseMs,
           lastLoadTiming.uploadMs);
}



void VulkanApp::run() {
    // maps/ 디렉터리에서 렌더링 가능한 씬 파일을 모두 찾는다.
    availableMaps = findMapFiles("maps");
    if (availableMaps.empty()) {
        // 탐색에 실패하면 기본 맵 경로를 그대로 사용한다.
        if (currentMapFile.empty())
            currentMapFile = "maps/test_city.glb";
        availableMaps.push_back(currentMapFile);
    }

    // currentMapFile과 일치하는 항목으로 현재 맵 인덱스를 맞춘다.
    // 비어 있거나 목록에 없으면 가벼운 기본 맵(test_city)을 우선 사용한다.
    auto inOrganizedMapDir = [](const std::string& p) {
        std::string lower = p;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return lower.find("/map/") != std::string::npos ||
               lower.find("\\map\\") != std::string::npos;
    };
    bool hasOrganizedMaps = false;
    for (const auto& m : availableMaps) {
        if (inOrganizedMapDir(m)) {
            hasOrganizedMaps = true;
            break;
        }
    }

    int selectedIndex = -1;
    if (!hasOrganizedMaps || currentMapFile != "maps/test_city.glb") {
        for (int i = 0; i < (int)availableMaps.size(); ++i) {
            if (availableMaps[i] == currentMapFile) {
                selectedIndex = i;
                break;
            }
        }
    }
    if (selectedIndex < 0) {
        for (int i = 0; i < (int)availableMaps.size(); ++i) {
            if (availableMaps[i] == "maps/test_city.glb") {
                selectedIndex = i;
                break;
            }
        }
    }
    if (selectedIndex < 0)
        selectedIndex = 0;
    if (!availableMaps.empty()) {
        currentMapIndex = selectedIndex;
        currentMapFile  = availableMaps[currentMapIndex];
    }

    std::cout << "Found " << availableMaps.size() << " map(s):\n";
    for (auto& m : availableMaps) std::cout << "  " << m << "\n";

    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

// 윈도우와 입력 콜백
void VulkanApp::initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Debug Scene", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanApp::framebufferResizeCallback(GLFWwindow* w, int, int) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    app->framebufferResized = true;
}

void VulkanApp::cursorPosCallback(GLFWwindow* w, double xpos, double ypos) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    if (app->firstMouse) {
        app->lastMouseX = xpos;
        app->lastMouseY = ypos;
        app->firstMouse = false;
        return;
    }
    float dx = static_cast<float>(xpos - app->lastMouseX);
    float dy = static_cast<float>(ypos - app->lastMouseY);
    app->lastMouseX = xpos;
    app->lastMouseY = ypos;

    if (app->ghostMode) {
        // ghost 모드에서 가운데 버튼을 누르면 배치 카메라를 회전한다.
        bool midHeld = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        if (midHeld)
            app->camera.processMouseDelta(dx, dy);
        else
            app->observerCamera.processMouseDelta(dx, dy);
    } else {
        // 리플레이 중에는 카메라를 재생 데이터가 제어한다.
        if (!app->isReplaying)
            app->camera.processMouseDelta(dx, dy);
    }
}

void VulkanApp::mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    // 필요하면 클릭 포커스 처리를 여기에 추가할 수 있다.
}

void VulkanApp::scrollCallback(GLFWwindow* w, double /*xoffset*/, double yoffset) {
    auto app = reinterpret_cast<VulkanApp*>(glfwGetWindowUserPointer(w));
    // 휠 한 칸마다 방향광을 Y축 기준 12도 회전한다.
    app->lightYaw += static_cast<float>(yoffset) * 12.0f;
}

void VulkanApp::toggleFullscreen() {
    if (!isFullscreen) {
        // 창 모드 위치와 크기를 저장한다.
        glfwGetWindowPos(window,  &windowedX, &windowedY);
        glfwGetWindowSize(window, &windowedW, &windowedH);
        // 주 모니터의 보더리스 전체화면으로 전환한다.
        GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0,
                             mode->width, mode->height, mode->refreshRate);
        isFullscreen = true;
    } else {
        // 저장해 둔 창 모드로 복귀한다.
        glfwSetWindowMonitor(window, nullptr,
                             windowedX, windowedY, windowedW, windowedH, 0);
        isFullscreen = false;
    }
    // GLFW resize 콜백이 스왑체인 재생성을 유도한다.
}

// Vulkan 초기화 순서
void VulkanApp::initVulkan() {
    createInstance();
    if (kEnableValidation) setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createShadowResources(); // 섀도맵은 descriptor set이 binding 3으로 참조하므로 그 전에 만든다.
    createShadowPipeline();
    createCommandPool();
    {
        using Clock = std::chrono::steady_clock;
        auto t_total = Clock::now();
        if (currentMapFile.empty()) {
            if (!availableMaps.empty()) {
                if (currentMapIndex < 0 || currentMapIndex >= (int)availableMaps.size())
                    currentMapIndex = 0;
                currentMapFile = availableMaps[currentMapIndex];
            } else {
                currentMapFile = "maps/test_city.glb";
            }
        }
        lastLoadTiming = {};
        lastLoadTiming.mapName = currentMapFile;
        buildScene();
        auto t_upload = Clock::now();
        createSceneBuffers(); // 하나의 커맨드 버퍼와 펜스로 일괄 업로드
        lastLoadTiming.uploadMs = std::chrono::duration<float, std::milli>(Clock::now() - t_upload).count();
        lastLoadTiming.totalMs  = std::chrono::duration<float, std::milli>(Clock::now() - t_total).count();
        lastLoadTiming.valid    = true;
        printf("[Load] %-32s  total %6.1f ms  parse %6.1f ms  upload %5.1f ms\n",
               currentMapFile.c_str(),
               lastLoadTiming.totalMs, lastLoadTiming.sceneParseMs,
               lastLoadTiming.uploadMs);
    }
    createUniformBuffers();
    createDefaultTextureAndSampler();
    createDescriptorPool();
    createDescriptorSets();
    createTextureResources();
    createCommandBuffers();
    createSyncObjects();
    createInstanceResources();
    createOcclusionQueryPool();
    createGizmoPipeline();
    createGizmoBuffers();
    createDeferredResources();
    initImGui();
    perfStats.init();
}

// Vulkan 인스턴스 생성
void VulkanApp::createInstance() {
    if (kEnableValidation) {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> layers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, layers.data());
        for (auto* name : kValidationLayers) {
            bool found = false;
            for (auto& l : layers) if (strcmp(l.layerName, name) == 0) { found = true; break; }
            if (!found) throw std::runtime_error(std::string("Validation layer not found: ") + name);
        }
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName   = "VulkanDebugScene";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName        = "None";
    appInfo.apiVersion         = VK_API_VERSION_1_2;

    uint32_t glfwExtCount;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char*> extensions(glfwExts, glfwExts + glfwExtCount);
    if (kEnableValidation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo        = &appInfo;
    ci.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
    ci.ppEnabledExtensionNames = extensions.data();
    if (kEnableValidation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }
    if (vkCreateInstance(&ci, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("vkCreateInstance failed");
}

void VulkanApp::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT ci{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debugCallback;
    CreateDebugMessenger(instance, &ci, &debugMessenger);
}

void VulkanApp::createSurface() {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        throw std::runtime_error("glfwCreateWindowSurface failed");
}

// 물리/논리 디바이스 선택
QueueFamilyIndices VulkanApp::findQueueFamilies(VkPhysicalDevice dev) {
    uint32_t count;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, props.data());

    QueueFamilyIndices idx;
    for (uint32_t i = 0; i < count; ++i) {
        if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) idx.graphics = i;
        VkBool32 presentSupport;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &presentSupport);
        if (presentSupport) idx.present = i;
        if (idx.isComplete()) break;
    }
    return idx;
}

SwapChainSupportDetails VulkanApp::querySwapChainSupport(VkPhysicalDevice dev) {
    SwapChainSupportDetails d;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, surface, &d.capabilities);
    uint32_t count;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, nullptr);
    if (count) { d.formats.resize(count); vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &count, d.formats.data()); }
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, nullptr);
    if (count) { d.presentModes.resize(count); vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &count, d.presentModes.data()); }
    return d;
}

bool VulkanApp::isDeviceSuitable(VkPhysicalDevice dev) {
    auto idx = findQueueFamilies(dev);
    if (!idx.isComplete()) return false;

    uint32_t extCount;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());
    for (auto* req : kDeviceExtensions) {
        bool found = false;
        for (auto& e : exts) if (strcmp(e.extensionName, req) == 0) { found = true; break; }
        if (!found) return false;
    }

    auto sc = querySwapChainSupport(dev);
    return !sc.formats.empty() && !sc.presentModes.empty();
}

void VulkanApp::pickPhysicalDevice() {
    uint32_t count;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (!count) throw std::runtime_error("No Vulkan GPUs found");
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    // 가능하면 외장 GPU를 우선 선택한다.
    for (auto dev : devs) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(dev, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && isDeviceSuitable(dev)) {
            physDevice = dev; return;
        }
    }
    for (auto dev : devs) if (isDeviceSuitable(dev)) { physDevice = dev; return; }
    throw std::runtime_error("No suitable GPU found");
}

void VulkanApp::createLogicalDevice() {
    auto idx = findQueueFamilies(physDevice);
    std::set<uint32_t> unique = {idx.graphics.value(), idx.present.value()};
    std::vector<VkDeviceQueueCreateInfo> queueCIs;
    float priority = 1.0f;
    for (uint32_t qf : unique) {
        VkDeviceQueueCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        ci.queueFamilyIndex = qf; ci.queueCount = 1; ci.pQueuePriorities = &priority;
        queueCIs.push_back(ci);
    }

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy                       = VK_TRUE;
    features.shaderSampledImageArrayDynamicIndexing  = VK_TRUE; // textures[] 동적 인덱싱

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount    = static_cast<uint32_t>(queueCIs.size());
    ci.pQueueCreateInfos       = queueCIs.data();
    ci.enabledExtensionCount   = static_cast<uint32_t>(kDeviceExtensions.size());
    ci.ppEnabledExtensionNames = kDeviceExtensions.data();
    ci.pEnabledFeatures        = &features;
    if (kEnableValidation) {
        ci.enabledLayerCount   = static_cast<uint32_t>(kValidationLayers.size());
        ci.ppEnabledLayerNames = kValidationLayers.data();
    }
    if (vkCreateDevice(physDevice, &ci, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDevice failed");

    vkGetDeviceQueue(device, idx.graphics.value(), 0, &graphicsQueue);
    vkGetDeviceQueue(device, idx.present.value(),  0, &presentQueue);
}

// 스왑체인 생성
void VulkanApp::createSwapChain() {
    auto sc = querySwapChainSupport(physDevice);

    // 스왑체인 색상 포맷 선택
    VkSurfaceFormatKHR fmt = sc.formats[0];
    for (auto& f : sc.formats)
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            { fmt = f; break; }

    // mailbox present mode를 우선 사용하고 없으면 FIFO를 쓴다.
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    for (auto m : sc.presentModes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) { mode = m; break; }

    // 프레임버퍼 크기 결정
    VkExtent2D extent;
    if (sc.capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        extent = sc.capabilities.currentExtent;
    } else {
        int w, h;
        glfwGetFramebufferSize(window, &w, &h);
        extent = {std::clamp((uint32_t)w, sc.capabilities.minImageExtent.width,  sc.capabilities.maxImageExtent.width),
                  std::clamp((uint32_t)h, sc.capabilities.minImageExtent.height, sc.capabilities.maxImageExtent.height)};
    }

    uint32_t imgCount = sc.capabilities.minImageCount + 1;
    if (sc.capabilities.maxImageCount > 0) imgCount = std::min(imgCount, sc.capabilities.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface            = surface;
    ci.minImageCount      = imgCount;
    ci.imageFormat        = fmt.format;
    ci.imageColorSpace    = fmt.colorSpace;
    ci.imageExtent        = extent;
    ci.imageArrayLayers   = 1;
    ci.imageUsage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    auto idx = findQueueFamilies(physDevice);
    uint32_t queueFamilies[] = {idx.graphics.value(), idx.present.value()};
    if (idx.graphics != idx.present) {
        ci.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices   = queueFamilies;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    ci.preTransform   = sc.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode    = mode;
    ci.clipped        = VK_TRUE;

    if (vkCreateSwapchainKHR(device, &ci, nullptr, &swapchain) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSwapchainKHR failed");

    uint32_t n;
    vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
    scImages.resize(n);
    vkGetSwapchainImagesKHR(device, swapchain, &n, scImages.data());
    scFormat = fmt.format;
    scExtent = extent;
}

void VulkanApp::createImageViews() {
    scImageViews.resize(scImages.size());
    for (size_t i = 0; i < scImages.size(); ++i)
        scImageViews[i] = createImageView(scImages[i], scFormat, VK_IMAGE_ASPECT_COLOR_BIT);
}

// 기본 렌더 패스
void VulkanApp::createRenderPass() {
    VkAttachmentDescription color{};
    color.format         = scFormat;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format         = findDepthFormat();
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference depthRef{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.srcAccessMask = 0;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array<VkAttachmentDescription, 2> attachments = {color, depth};
    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments    = attachments.data();
    ci.subpassCount    = 1;
    ci.pSubpasses      = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies   = &dep;

    if (vkCreateRenderPass(device, &ci, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("vkCreateRenderPass failed");
}

// 디스크립터 세트 레이아웃
void VulkanApp::createDescriptorSetLayout() {
    // set 0 binding 0은 카메라 UBO다.
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding         = 0;
    uboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.descriptorCount = 1;
    uboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 1은 씬 텍스처 배열이다.
    // 사용하지 않는 슬롯은 1×1 흰색 기본 텍스처로 채워서 모든 슬롯이 항상 유효하다.
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding         = 1;
    samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = MAX_TEXTURES;
    samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 2는 SceneLightUBO다.
    VkDescriptorSetLayoutBinding lightBinding{};
    lightBinding.binding         = 2;
    lightBinding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    lightBinding.descriptorCount = 1;
    lightBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    // set 0 binding 3은 방향광 섀도맵 샘플러다(forward·deferred 양쪽이 공유).
    VkDescriptorSetLayoutBinding shadowBinding{};
    shadowBinding.binding         = 3;
    shadowBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    shadowBinding.descriptorCount = 1;
    shadowBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutBinding bindings[] = {uboBinding, samplerBinding, lightBinding, shadowBinding};

    VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 4;
    ci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &descSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");

    // set 1 binding 0은 인스턴스 모델 행렬 SSBO다.
    VkDescriptorSetLayoutBinding ssboBinding{};
    ssboBinding.binding         = 0;
    ssboBinding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    ssboBinding.descriptorCount = 1;
    ssboBinding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    ci.bindingCount = 1;
    ci.pBindings = &ssboBinding;
    if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &instanceDescSetLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorSetLayout (instance SSBO) failed");
}

// 그래픽 파이프라인 생성
std::vector<char> VulkanApp::readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file: " + path);
    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    return buf;
}

VkShaderModule VulkanApp::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m;
    if (vkCreateShaderModule(device, &ci, nullptr, &m) != VK_SUCCESS)
        throw std::runtime_error("vkCreateShaderModule failed");
    return m;
}

void VulkanApp::createGraphicsPipeline() {
    // SPIR-V 셰이더 모듈을 읽고 생성한다.
    auto vertCode     = readFile("shaders/spv/scene.vert.spv");
    auto fragCode     = readFile("shaders/spv/scene.frag.spv");
    auto vertInstCode = readFile("shaders/spv/scene_instanced.vert.spv");

    VkShaderModule vertM     = createShaderModule(vertCode);
    VkShaderModule fragM     = createShaderModule(fragCode);
    VkShaderModule vertInstM = createShaderModule(vertInstCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertM;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragM;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vertStage, fragStage};

    // 고정 기능 파이프라인 상태를 설정한다.
    auto binding  = Vertex::getBindingDesc();
    auto attrs    = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    std::array<VkVertexInputBindingDescription, 2> instBindings{};
    instBindings[0] = binding;
    instBindings[1].binding   = 1;
    instBindings[1].stride    = sizeof(glm::mat4);
    instBindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    std::array<VkVertexInputAttributeDescription, 8> instAttrs{};
    for (size_t i = 0; i < attrs.size(); ++i)
        instAttrs[i] = attrs[i];
    for (uint32_t col = 0; col < 4; ++col) {
        instAttrs[4 + col].location = 4 + col;
        instAttrs[4 + col].binding  = 1;
        instAttrs[4 + col].format   = VK_FORMAT_R32G32B32A32_SFLOAT;
        instAttrs[4 + col].offset   = sizeof(glm::vec4) * col;
    }

    VkPipelineVertexInputStateCreateInfo instVertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    instVertInput.vertexBindingDescriptionCount   = static_cast<uint32_t>(instBindings.size());
    instVertInput.pVertexBindingDescriptions      = instBindings.data();
    instVertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(instAttrs.size());
    instVertInput.pVertexAttributeDescriptions    = instAttrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // 뷰포트와 시저는 프레임마다 동적으로 설정한다.
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState blendAttach{};
    blendAttach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttach.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &blendAttach;

    // 모든 파이프라인이 같은 push constant 범위를 공유한다.
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(PushConstants);

    // 일반/인스턴싱 파이프라인 레이아웃을 만든다.
    // 일반 렌더링은 set 0의 카메라/조명/텍스처 디스크립터만 사용한다.
    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount         = 1;
    layoutCI.pSetLayouts            = &descSetLayout;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout failed");

    // 인스턴싱 렌더링은 set 0에 더해 set 1의 인스턴스 행렬 SSBO를 사용한다.
    VkDescriptorSetLayout instLayouts[] = { descSetLayout, instanceDescSetLayout };
    layoutCI.setLayoutCount = 2;
    layoutCI.pSetLayouts    = instLayouts;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &instancePipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout (instanced) failed");

    // 공통 그래픽 파이프라인 생성 정보를 준비한다.
    VkGraphicsPipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeCI.stageCount          = 2;
    pipeCI.pStages             = stages;
    pipeCI.pVertexInputState   = &vertInput;
    pipeCI.pInputAssemblyState = &inputAssembly;
    pipeCI.pViewportState      = &viewportState;
    pipeCI.pRasterizationState = &raster;
    pipeCI.pMultisampleState   = &msaa;
    pipeCI.pDepthStencilState  = &depthStencil;
    pipeCI.pColorBlendState    = &blend;
    pipeCI.pDynamicState       = &dynState;
    pipeCI.renderPass          = renderPass;
    pipeCI.subpass             = 0;

    // 일반 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_BACK_BIT;
    pipeCI.layout   = pipelineLayout;
    stages[0].module = vertM;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines failed");

    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineFlippedCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (flipped cull) failed");
    raster.frontFace = VK_FRONT_FACE_CLOCKWISE;

    // 일반 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_NONE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineNoCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (no-cull) failed");

    // 인스턴싱 파이프라인을 생성한다.
    raster.cullMode  = VK_CULL_MODE_BACK_BIT;
    pipeCI.layout    = instancePipelineLayout;
    pipeCI.pVertexInputState = &instVertInput;
    stages[0].module = vertInstM;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineInst) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (instanced) failed");

    // 인스턴싱 파이프라인을 생성한다.
    raster.cullMode = VK_CULL_MODE_NONE;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineInstNoCull) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (instanced no-cull) failed");
    pipeCI.pVertexInputState = &vertInput;

    // 투명 오브젝트용 알파 블렌딩 파이프라인을 만든다.
    // 일반 버텍스/프래그먼트 셰이더를 그대로 재사용한다.
    // 투명 오브젝트는 모든 불투명 오브젝트 뒤에 뒤에서 앞으로 그린다.
    {
        blendAttach.blendEnable         = VK_TRUE;
        blendAttach.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttach.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttach.colorBlendOp        = VK_BLEND_OP_ADD;
        blendAttach.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttach.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttach.alphaBlendOp        = VK_BLEND_OP_ADD;

        depthStencil.depthWriteEnable = VK_FALSE; // 투명 오브젝트는 깊이를 쓰지 않는다.
        raster.cullMode               = VK_CULL_MODE_NONE; // 유리처럼 양면이 보여야 하는 재질을 위해 컬링하지 않는다.

        pipeCI.layout    = pipelineLayout;
        stages[0].module = vertM;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineAlpha) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (alpha) failed");

        // 이후 파이프라인 생성에 영향을 주지 않도록 상태를 되돌린다.
        blendAttach.blendEnable       = VK_FALSE;
        depthStencil.depthWriteEnable = VK_TRUE;
    }

    // 바운딩 박스 오클루전 쿼리 전용 파이프라인을 만든다.
    {
        blendAttach.colorWriteMask    = 0; // 색상 버퍼에는 쓰지 않는다.
        blendAttach.blendEnable       = VK_FALSE;
        depthStencil.depthWriteEnable = VK_FALSE; // 실제 깊이 버퍼를 오염시키지 않는다.
        depthStencil.depthTestEnable  = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
        raster.cullMode               = VK_CULL_MODE_NONE;
        pipeCI.layout    = pipelineLayout;
        stages[0].module = vertM;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &graphicsPipelineQueryOnly) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (query-only) failed");
        // 상태 복원
        blendAttach.colorWriteMask    = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;
    }

    vkDestroyShaderModule(device, vertM,     nullptr);
    vkDestroyShaderModule(device, fragM,     nullptr);
    vkDestroyShaderModule(device, vertInstM, nullptr);

    // 디퍼드 렌더링 파이프라인을 만든다.

    // G-Buffer 렌더패스는 컬러 attachment들과 깊이 attachment를 사용한다.
    {
        VkFormat depthFmt = findDepthFormat();
        VkAttachmentDescription gbAtts[GBUFFER_COLOR_ATTACHMENTS + 1] = {};
        gbAtts[0].format         = VK_FORMAT_R8G8B8A8_UNORM;
        gbAtts[0].samples        = VK_SAMPLE_COUNT_1_BIT;
        gbAtts[0].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbAtts[0].storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
        gbAtts[0].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        gbAtts[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[0].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        gbAtts[0].finalLayout    = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        gbAtts[1]        = gbAtts[0];
        gbAtts[1].format = VK_FORMAT_R16G16B16A16_SFLOAT;
        gbAtts[2]        = gbAtts[1];
        gbAtts[3]        = gbAtts[1];
        gbAtts[4].format         = depthFmt;
        gbAtts[4].samples        = VK_SAMPLE_COUNT_1_BIT;
        gbAtts[4].loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        gbAtts[4].storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[4].stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        gbAtts[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        gbAtts[4].initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        gbAtts[4].finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        VkAttachmentReference cRefs[GBUFFER_COLOR_ATTACHMENTS] = {
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        };
        VkAttachmentReference dRef = {GBUFFER_COLOR_ATTACHMENTS, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount    = GBUFFER_COLOR_ATTACHMENTS;
        sp.pColorAttachments       = cRefs;
        sp.pDepthStencilAttachment = &dRef;
        VkSubpassDependency dep{};
        dep.srcSubpass    = VK_SUBPASS_EXTERNAL; dep.dstSubpass = 0;
        dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.srcAccessMask = 0;
        dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpCI.attachmentCount = GBUFFER_COLOR_ATTACHMENTS + 1; rpCI.pAttachments = gbAtts;
        rpCI.subpassCount    = 1; rpCI.pSubpasses   = &sp;
        rpCI.dependencyCount = 1; rpCI.pDependencies = &dep;
        if (vkCreateRenderPass(device, &rpCI, nullptr, &gbufRenderPass) != VK_SUCCESS)
            throw std::runtime_error("vkCreateRenderPass (G-Buffer) failed");
    }

    // 지오메트리 패스용 파이프라인을 만든다.
    {
        auto vertCode2    = readFile("shaders/spv/scene.vert.spv");
        auto gbufFragCode = readFile("shaders/spv/gbuffer.frag.spv");
        VkShaderModule vertM2    = createShaderModule(vertCode2);
        VkShaderModule gbufFragM = createShaderModule(gbufFragCode);
        VkPipelineShaderStageCreateInfo gbStages[2]{};
        gbStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        gbStages[0].module = vertM2; gbStages[0].pName = "main";
        gbStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        gbStages[1].module = gbufFragM; gbStages[1].pName = "main";
        VkPipelineColorBlendAttachmentState gbAtt{};
        gbAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        gbAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendAttachmentState gbAtts3[GBUFFER_COLOR_ATTACHMENTS] = {gbAtt, gbAtt, gbAtt, gbAtt};
        VkPipelineColorBlendStateCreateInfo gbBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        gbBlend.attachmentCount = GBUFFER_COLOR_ATTACHMENTS; gbBlend.pAttachments = gbAtts3;
        VkPipelineDepthStencilStateCreateInfo gbDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        gbDs.depthTestEnable  = VK_TRUE;
        gbDs.depthWriteEnable = VK_TRUE;
        gbDs.depthCompareOp   = VK_COMPARE_OP_LESS;
        VkGraphicsPipelineCreateInfo gbCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        gbCI.stageCount = 2; gbCI.pStages = gbStages;
        gbCI.pVertexInputState   = &vertInput;
        gbCI.pInputAssemblyState = &inputAssembly;
        gbCI.pViewportState      = &viewportState;
        gbCI.pRasterizationState = &raster;
        gbCI.pMultisampleState   = &msaa;
        gbCI.pDepthStencilState  = &gbDs;
        gbCI.pColorBlendState    = &gbBlend;
        gbCI.pDynamicState       = &dynState;
        gbCI.layout              = pipelineLayout;
        gbCI.renderPass          = gbufRenderPass;
        gbCI.subpass             = 0;
        raster.cullMode  = VK_CULL_MODE_BACK_BIT;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gbCI, nullptr, &gbufPipeline) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (gbuf) failed");
        raster.cullMode = VK_CULL_MODE_NONE;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &gbCI, nullptr, &gbufPipelineNoCull) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (gbuf no-cull) failed");
        vkDestroyShaderModule(device, gbufFragM, nullptr);
        vkDestroyShaderModule(device, vertM2,    nullptr);
    }

    // 디퍼드 조명 패스가 읽을 G-Buffer 샘플러 레이아웃을 만든다.
    {
        VkDescriptorSetLayoutBinding binds[GBUFFER_COLOR_ATTACHMENTS] = {};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            binds[i].binding         = (uint32_t)i;
            binds[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ci.bindingCount = GBUFFER_COLOR_ATTACHMENTS; ci.pBindings = binds;
        if (vkCreateDescriptorSetLayout(device, &ci, nullptr, &deferredDescSetLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorSetLayout (deferred) failed");
    }

    // 전체화면 조명 패스는 씬 UBO와 G-Buffer 디스크립터를 함께 사용한다.
    {
        VkDescriptorSetLayout sets[] = {descSetLayout, deferredDescSetLayout};
        VkPipelineLayoutCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        ci.setLayoutCount = 2; ci.pSetLayouts = sets;
        if (vkCreatePipelineLayout(device, &ci, nullptr, &deferredLightLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreatePipelineLayout (deferred light) failed");
    }

    // 버텍스 버퍼 없이 전체화면 삼각형을 그린다.
    {
        auto dLVCode = readFile("shaders/spv/deferred_light.vert.spv");
        auto dLFCode = readFile("shaders/spv/deferred_light.frag.spv");
        VkShaderModule dLVM = createShaderModule(dLVCode);
        VkShaderModule dLFM = createShaderModule(dLFCode);
        VkPipelineShaderStageCreateInfo dStages[2]{};
        dStages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_VERTEX_BIT,   dLVM, "main", nullptr};
        dStages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_FRAGMENT_BIT, dLFM, "main", nullptr};
        VkPipelineVertexInputStateCreateInfo emptyVI{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState dAtt{};
        dAtt.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                              VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        dAtt.blendEnable = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo dBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        dBlend.attachmentCount = 1; dBlend.pAttachments = &dAtt;
        VkPipelineDepthStencilStateCreateInfo dDs{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        dDs.depthTestEnable  = VK_FALSE;
        dDs.depthWriteEnable = VK_FALSE;
        raster.cullMode  = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        VkGraphicsPipelineCreateInfo dCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        dCI.stageCount = 2; dCI.pStages = dStages;
        dCI.pVertexInputState   = &emptyVI;
        dCI.pInputAssemblyState = &inputAssembly;
        dCI.pViewportState      = &viewportState;
        dCI.pRasterizationState = &raster;
        dCI.pMultisampleState   = &msaa;
        dCI.pDepthStencilState  = &dDs;
        dCI.pColorBlendState    = &dBlend;
        dCI.pDynamicState       = &dynState;
        dCI.layout              = deferredLightLayout;
        dCI.renderPass          = renderPass;
        dCI.subpass             = 0;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &dCI, nullptr, &deferredLightPipeline) != VK_SUCCESS)
            throw std::runtime_error("vkCreateGraphicsPipelines (deferred light) failed");
        vkDestroyShaderModule(device, dLVM, nullptr);
        vkDestroyShaderModule(device, dLFM, nullptr);
    }

}

// 깊이 이미지
VkFormat VulkanApp::findSupportedFormat(const std::vector<VkFormat>& candidates,
                                         VkImageTiling tiling, VkFormatFeatureFlags features) {
    for (auto fmt : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physDevice, fmt, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR  && (props.linearTilingFeatures  & features) == features) return fmt;
        if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) return fmt;
    }
    throw std::runtime_error("No supported depth format");
}

VkFormat VulkanApp::findDepthFormat() {
    return findSupportedFormat(
        {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void VulkanApp::createDepthResources() {
    auto fmt = findDepthFormat();
    createImage(scExtent.width, scExtent.height, fmt, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(depthImage, fmt, VK_IMAGE_ASPECT_DEPTH_BIT);
}

// 프레임버퍼
void VulkanApp::createFramebuffers() {
    framebuffers.resize(scImageViews.size());
    for (size_t i = 0; i < scImageViews.size(); ++i) {
        std::array<VkImageView, 2> attachments = {scImageViews[i], depthImageView};
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = renderPass;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments    = attachments.data();
        ci.width           = scExtent.width;
        ci.height          = scExtent.height;
        ci.layers          = 1;
        if (vkCreateFramebuffer(device, &ci, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer failed");
    }
}

// 커맨드 풀과 커맨드 버퍼
void VulkanApp::createCommandPool() {
    auto idx = findQueueFamilies(physDevice);
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = idx.graphics.value();
    if (vkCreateCommandPool(device, &ci, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateCommandPool failed");
}

void VulkanApp::createCommandBuffers() {
    commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());
    if (vkAllocateCommandBuffers(device, &ai, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateCommandBuffers failed");
}

// 버퍼 헬퍼
uint32_t VulkanApp::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i)
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable memory type");
}

void VulkanApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &ci, nullptr, &buf) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");
    vkBindBufferMemory(device, buf, mem, 0);
}


void VulkanApp::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    VkBufferCopy copy{0, 0, size};
    vkCmdCopyBuffer(cmd, src, dst, 1, &copy);
    vkEndCommandBuffer(cmd);

    // 전체 큐를 멈추지 않도록 fence로 복사 완료만 기다린다.
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fci, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);
}

// 버텍스/인덱스 버퍼
void VulkanApp::createVertexBuffer() {
    VkDeviceSize size = sizeof(Vertex) * vertices.size();
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* data;
    vkMapMemory(device, stagingMem, 0, size, 0, &data);
    memcpy(data, vertices.data(), size);
    vkUnmapMemory(device, stagingMem);
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);
    copyBuffer(staging, vertexBuffer, size);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
}

void VulkanApp::createIndexBuffer() {
    VkDeviceSize size = sizeof(uint32_t) * indices.size();
    VkBuffer staging; VkDeviceMemory stagingMem;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);
    void* data;
    vkMapMemory(device, stagingMem, 0, size, 0, &data);
    memcpy(data, indices.data(), size);
    vkUnmapMemory(device, stagingMem);
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
    copyBuffer(staging, indexBuffer, size);
    vkDestroyBuffer(device, staging, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);
}

// 버텍스와 인덱스 버퍼를 하나의 커맨드 버퍼/펜스로 함께 업로드한다.
// 씬 재로드 때 버텍스/인덱스 업로드가 큐 제출 두 번으로 쪼개지는 것을 피한다.
void VulkanApp::createSceneBuffers() {
    VkDeviceSize vSize = sizeof(Vertex)   * vertices.size();
    VkDeviceSize iSize = sizeof(uint32_t) * indices.size();

    // 버텍스 스테이징 버퍼를 준비한다.
    VkBuffer vStaging; VkDeviceMemory vStagingMem;
    createBuffer(vSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 vStaging, vStagingMem);
    void* vData;
    vkMapMemory(device, vStagingMem, 0, vSize, 0, &vData);
    memcpy(vData, vertices.data(), vSize);
    vkUnmapMemory(device, vStagingMem);
    createBuffer(vSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vertexBuffer, vertexBufferMemory);

    // 인덱스 스테이징 버퍼를 준비한다.
    VkBuffer iStaging; VkDeviceMemory iStagingMem;
    createBuffer(iSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 iStaging, iStagingMem);
    void* iData;
    vkMapMemory(device, iStagingMem, 0, iSize, 0, &iData);
    memcpy(iData, indices.data(), iSize);
    vkUnmapMemory(device, iStagingMem);
    createBuffer(iSize,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 indexBuffer, indexBufferMemory);

    // 하나의 커맨드 버퍼에 복사를 모두 기록한다.
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = commandPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(device, &ai, &cmd);

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);
    VkBufferCopy vCopy{0, 0, vSize};
    vkCmdCopyBuffer(cmd, vStaging, vertexBuffer, 1, &vCopy);
    VkBufferCopy iCopy{0, 0, iSize};
    vkCmdCopyBuffer(cmd, iStaging, indexBuffer,  1, &iCopy);
    vkEndCommandBuffer(cmd);

    // 한 번 제출하고 하나의 fence만 기다린다.
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(device, &fci, nullptr, &fence);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(graphicsQueue, 1, &si, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, commandPool, 1, &cmd);

    vkDestroyBuffer(device, vStaging, nullptr); vkFreeMemory(device, vStagingMem, nullptr);
    vkDestroyBuffer(device, iStaging, nullptr); vkFreeMemory(device, iStagingMem, nullptr);
}


// 프레임별 UBO
void VulkanApp::createUniformBuffers() {
    VkDeviceSize camSize   = sizeof(CameraUBO);
    VkDeviceSize lightSize = sizeof(SceneLightUBO);

    uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBufferMemories.resize(MAX_FRAMES_IN_FLIGHT);
    cullUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOs.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOMemories.resize(MAX_FRAMES_IN_FLIGHT);
    sceneLightUBOMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(camSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     uniformBuffers[i], uniformBufferMemories[i]);
        vkMapMemory(device, uniformBufferMemories[i], 0, camSize, 0, &uniformBuffersMapped[i]);

        createBuffer(camSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     cullUniformBuffers[i], cullUniformBufferMemories[i]);
        vkMapMemory(device, cullUniformBufferMemories[i], 0, camSize, 0, &cullUniformBuffersMapped[i]);

        createBuffer(lightSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     sceneLightUBOs[i], sceneLightUBOMemories[i]);
        vkMapMemory(device, sceneLightUBOMemories[i], 0, lightSize, 0, &sceneLightUBOMapped[i]);
        // 초기값: 조명 없음, fallback 모드
        memset(sceneLightUBOMapped[i], 0, lightSize);
    }
}

// 디스크립터 풀과 세트
void VulkanApp::createDescriptorPool() {
    // 렌더 카메라와 컬링 카메라가 각각 프레임 수만큼 디스크립터 세트를 가진다.
    // 각 세트는 카메라 UBO, 조명 UBO, 텍스처 샘플러 배열을 포함한다.
    uint32_t setCount = 2u * (uint32_t)MAX_FRAMES_IN_FLIGHT;
    VkDescriptorPoolSize poolSizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
         setCount * 2}, // binding 0=카메라, binding 2=씬 조명
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         (MAX_TEXTURES + 1) * setCount}, // binding 1=텍스처 배열, binding 3=섀도맵
    };
    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets       = setCount;
    ci.poolSizeCount = 2;
    ci.pPoolSizes    = poolSizes;
    if (vkCreateDescriptorPool(device, &ci, nullptr, &descPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool failed");
}

void VulkanApp::createDescriptorSets() {
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descSetLayout);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = descPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    descSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, descSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets failed");

    cullDescSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, cullDescSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (cull) failed");

    // 텍스처 배열 (binding 1): 모든 슬롯을 기본 1×1 흰색 텍스처로 채움.
    // 실제 씬 텍스처는 createTextureResources() 에서 갱신된다.
    std::vector<VkDescriptorImageInfo> defaultImgInfos(MAX_TEXTURES);
    for (uint32_t s = 0; s < MAX_TEXTURES; ++s) {
        defaultImgInfos[s].sampler     = texSampler;
        defaultImgInfos[s].imageView   = defaultTexView;
        defaultImgInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        // binding 0: 카메라 UBO
        VkDescriptorBufferInfo bufInfo{uniformBuffers[i], 0, sizeof(CameraUBO)};
        VkWriteDescriptorSet uboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        uboWrite.dstSet          = descSets[i];
        uboWrite.dstBinding      = 0;
        uboWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboWrite.descriptorCount = 1;
        uboWrite.pBufferInfo     = &bufInfo;

        // binding 1: 텍스처 배열(기본 흰색으로 초기화)
        VkWriteDescriptorSet texWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        texWrite.dstSet          = descSets[i];
        texWrite.dstBinding      = 1;
        texWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        texWrite.descriptorCount = MAX_TEXTURES;
        texWrite.pImageInfo      = defaultImgInfos.data();

        // binding 2: 씬 조명 UBO
        VkDescriptorBufferInfo lightBufInfo{sceneLightUBOs[i], 0, sizeof(SceneLightUBO)};
        VkWriteDescriptorSet lightWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        lightWrite.dstSet          = descSets[i];
        lightWrite.dstBinding      = 2;
        lightWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightWrite.descriptorCount = 1;
        lightWrite.pBufferInfo     = &lightBufInfo;

        // binding 3: 방향광 섀도맵 (프레임 슬롯별 깊이 이미지)
        VkDescriptorImageInfo shadowInfo{shadowSampler, shadowImageViews[i],
                                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet shadowWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        shadowWrite.dstSet          = descSets[i];
        shadowWrite.dstBinding      = 3;
        shadowWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        shadowWrite.descriptorCount = 1;
        shadowWrite.pImageInfo      = &shadowInfo;

        VkWriteDescriptorSet writes[] = {uboWrite, texWrite, lightWrite, shadowWrite};
        vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);

        // 컬링용 세트도 같은 레이아웃을 쓰므로 텍스처 배열까지 유효하게 채운다.
        VkDescriptorBufferInfo cullBufInfo{cullUniformBuffers[i], 0, sizeof(CameraUBO)};
        VkWriteDescriptorSet cullUboWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullUboWrite.dstSet          = cullDescSets[i];
        cullUboWrite.dstBinding      = 0;
        cullUboWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cullUboWrite.descriptorCount = 1;
        cullUboWrite.pBufferInfo     = &cullBufInfo;

        VkWriteDescriptorSet cullTexWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullTexWrite.dstSet          = cullDescSets[i];
        cullTexWrite.dstBinding      = 1;
        cullTexWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cullTexWrite.descriptorCount = MAX_TEXTURES;
        cullTexWrite.pImageInfo      = defaultImgInfos.data();

        VkDescriptorBufferInfo cullLightBufInfo{sceneLightUBOs[i], 0, sizeof(SceneLightUBO)};
        VkWriteDescriptorSet cullLightWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullLightWrite.dstSet          = cullDescSets[i];
        cullLightWrite.dstBinding      = 2;
        cullLightWrite.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        cullLightWrite.descriptorCount = 1;
        cullLightWrite.pBufferInfo     = &cullLightBufInfo;

        VkDescriptorImageInfo cullShadowInfo{shadowSampler, shadowImageViews[i],
                                             VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet cullShadowWrite{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        cullShadowWrite.dstSet          = cullDescSets[i];
        cullShadowWrite.dstBinding      = 3;
        cullShadowWrite.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cullShadowWrite.descriptorCount = 1;
        cullShadowWrite.pImageInfo      = &cullShadowInfo;

        VkWriteDescriptorSet cullWrites[] = {cullUboWrite, cullTexWrite, cullLightWrite, cullShadowWrite};
        vkUpdateDescriptorSets(device, 4, cullWrites, 0, nullptr);
    }
}

// 프레임 동기화 객체
void VulkanApp::createSyncObjects() {
    imageAvailableSems.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &sci, nullptr, &imageAvailableSems[i]);
        vkCreateFence(device, &fci, nullptr, &inFlightFences[i]);
    }
    recreatePresentSemaphores();
}

void VulkanApp::recreatePresentSemaphores() {
    for (VkSemaphore sem : renderFinishedSems)
        if (sem != VK_NULL_HANDLE)
            vkDestroySemaphore(device, sem, nullptr);

    renderFinishedSems.assign(scImages.size(), VK_NULL_HANDLE);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (size_t i = 0; i < renderFinishedSems.size(); ++i)
        vkCreateSemaphore(device, &sci, nullptr, &renderFinishedSems[i]);
}

// GPU 인스턴싱 리소스
void VulkanApp::createInstanceResources() {
    const VkDeviceSize ssboSize = MAX_INSTANCES * sizeof(glm::mat4);

    // 프레임 슬롯마다 host-visible SSBO를 하나씩 둔다.
    instanceSSBOs.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOMemories.resize(MAX_FRAMES_IN_FLIGHT);
    instanceStagingBuffers.resize(MAX_FRAMES_IN_FLIGHT);
    instanceStagingMemories.resize(MAX_FRAMES_IN_FLIGHT);
    instanceSSBOMapped.resize(MAX_FRAMES_IN_FLIGHT);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(ssboSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     instanceSSBOs[i], instanceSSBOMemories[i]);
        createBuffer(ssboSize,
                     VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     instanceStagingBuffers[i], instanceStagingMemories[i]);
        vkMapMemory(device, instanceStagingMemories[i], 0, ssboSize, 0, &instanceSSBOMapped[i]);
    }

    // 프레임별 SSBO descriptor set용 풀을 만든다.
    VkDescriptorPoolSize ssboPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)MAX_FRAMES_IN_FLIGHT};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = MAX_FRAMES_IN_FLIGHT;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &ssboPoolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &instanceDescPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool (instance) failed");

    // 인스턴싱 파이프라인 set 1을 프레임별로 할당한다.
    std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, instanceDescSetLayout);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = instanceDescPool;
    ai.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    ai.pSetLayouts        = layouts.data();
    instanceDescSets.resize(MAX_FRAMES_IN_FLIGHT);
    if (vkAllocateDescriptorSets(device, &ai, instanceDescSets.data()) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (instance) failed");

    // 각 descriptor set이 자기 프레임의 SSBO를 가리키게 한다.
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorBufferInfo bufInfo{instanceSSBOs[i], 0, ssboSize};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet          = instanceDescSets[i];
        write.dstBinding      = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.descriptorCount = 1;
        write.pBufferInfo     = &bufInfo;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
    }
}

// 오클루전 쿼리 풀
void VulkanApp::createOcclusionQueryPool() {
    // 프레임 슬롯마다 draw object 수만큼 오클루전 쿼리를 확보한다.
    occQueryCount = static_cast<uint32_t>(drawObjects.size());
    VkQueryPoolCreateInfo ci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    ci.queryType  = VK_QUERY_TYPE_OCCLUSION;
    ci.queryCount = occQueryCount * MAX_FRAMES_IN_FLIGHT;
    if (vkCreateQueryPool(device, &ci, nullptr, &occlusionQueryPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateQueryPool failed");
}

void VulkanApp::resetOcclusionState() {
    const size_t objectCount = drawObjects.size();
    occResults.assign(objectCount, 1u);
    occQueryBuf.assign(objectCount * 2, 0u);
    occVisible.assign(objectCount, 1u);
    occZeroStreak.assign(objectCount, 0u);
    occWarmupFrames = MAX_FRAMES_IN_FLIGHT + OCC_ZERO_FRAMES_TO_HIDE;
}

bool VulkanApp::isOcclusionHidden(int objectIndex) const {
    if (objectIndex < 0) return false;
    const size_t idx = static_cast<size_t>(objectIndex);
    return idx < occVisible.size() && occVisible[idx] == 0;
}

// 기즈모 라인 파이프라인
void VulkanApp::createGizmoPipeline() {
    auto vertCode = readFile("shaders/spv/gizmo.vert.spv");
    auto fragCode = readFile("shaders/spv/gizmo.frag.spv");
    VkShaderModule vertM = createShaderModule(vertCode);
    VkShaderModule fragM = createShaderModule(fragCode);

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertM; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragM; stages[1].pName = "main";

    // 기즈모도 동일한 버텍스 포맷(pos+normal+color)을 사용한다.
    auto binding = Vertex::getBindingDesc();
    auto attrs   = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; // LINE_LIST 토폴로지

    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1;
    vs.scissorCount  = 1;

    VkDynamicState gizmoDynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo gizmoDyn{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    gizmoDyn.dynamicStateCount = 2;
    gizmoDyn.pDynamicStates    = gizmoDynStates;

    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode    = VK_CULL_MODE_NONE;
    raster.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // 깊이 테스트를 꺼서 기즈모를 항상 위에 그린다.
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable  = VK_FALSE;
    ds.depthWriteEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT|VK_COLOR_COMPONENT_G_BIT|
                        VK_COLOR_COMPONENT_B_BIT|VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1; blend.pAttachments = &ba;

    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount          = 2;
    ci.pStages             = stages;
    ci.pVertexInputState   = &vertInput;
    ci.pInputAssemblyState = &ia;
    ci.pViewportState      = &vs;
    ci.pRasterizationState = &raster;
    ci.pMultisampleState   = &msaa;
    ci.pDepthStencilState  = &ds;
    ci.pColorBlendState    = &blend;
    ci.pDynamicState       = &gizmoDyn;
    ci.layout              = pipelineLayout;
    ci.renderPass          = renderPass;
    ci.subpass             = 0;

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &gizmoPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (gizmo) failed");

    vkDestroyShaderModule(device, vertM, nullptr);
    vkDestroyShaderModule(device, fragM, nullptr);
}

// 프레임별 기즈모 버텍스 버퍼
void VulkanApp::createGizmoBuffers() {
    const VkDeviceSize sz = GIZMO_MAX_VERTS * sizeof(Vertex);
    gizmoVBs.resize(MAX_FRAMES_IN_FLIGHT);
    gizmoVBMemories.resize(MAX_FRAMES_IN_FLIGHT);
    gizmoVBMapped.resize(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        createBuffer(sz,
                     VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     gizmoVBs[i], gizmoVBMemories[i]);
        vkMapMemory(device, gizmoVBMemories[i], 0, sz, 0, &gizmoVBMapped[i]);
    }
}

// 배치 카메라 기즈모 라인 생성
void VulkanApp::buildGizmoGeometry(uint32_t frame) {
    std::vector<Vertex> verts;
    verts.reserve(GIZMO_MAX_VERTS);

    auto addLine = [&](glm::vec3 a, glm::vec3 b, glm::vec3 col) {
        verts.push_back({a, {0,1,0}, col});
        verts.push_back({b, {0,1,0}, col});
    };

    glm::vec3 pos   = camera.position;
    glm::vec3 fwd   = camera.getCameraFront();
    glm::vec3 right = camera.getRight();
    glm::vec3 up    = glm::normalize(glm::cross(right, fwd));

    // 카메라 본체를 방향이 있는 박스로 그린다.
    const float hw = 0.22f, hh = 0.14f, hd = 0.16f;
    glm::vec3 bodyCol = {1.0f, 0.55f, 0.05f}; // 주황색

    // 뒤/앞 면의 8개 꼭짓점을 계산한다.
    glm::vec3 c[8] = {
        pos + right*(-hw) + up*(-hh) + fwd*(-hd), // 0 BLB
        pos + right*(+hw) + up*(-hh) + fwd*(-hd), // 1 BRB
        pos + right*(+hw) + up*(+hh) + fwd*(-hd), // 2 BRT
        pos + right*(-hw) + up*(+hh) + fwd*(-hd), // 3 BLT
        pos + right*(-hw) + up*(-hh) + fwd*(+hd), // 4 FLB
        pos + right*(+hw) + up*(-hh) + fwd*(+hd), // 5 FRB
        pos + right*(+hw) + up*(+hh) + fwd*(+hd), // 6 FRT
        pos + right*(-hw) + up*(+hh) + fwd*(+hd), // 7 FLT
    };
    // 박스의 12개 모서리를 연결한다.
    int edges[12][2] = {
        {0,1},{1,2},{2,3},{3,0}, // 뒤쪽 면
        {4,5},{5,6},{6,7},{7,4}, // 앞쪽 면
        {0,4},{1,5},{2,6},{3,7} // 앞뒤 연결 모서리
    };
    for (auto& e : edges) addLine(c[e[0]], c[e[1]], bodyCol);

    // 위쪽 방향을 청록색 선으로 표시한다.
    glm::vec3 topMid = (c[3] + c[2]) * 0.5f;
    addLine(topMid, topMid + up * 0.18f, {0.4f, 0.85f, 1.0f});

    // 렌즈 방향을 앞쪽 꼭짓점에서 뻗는 선으로 표시한다.
    glm::vec3 lensColor = {1.0f, 0.9f, 0.2f}; // 노란색
    glm::vec3 lensTip   = pos + fwd * (hd + 0.18f);
    addLine(c[4], lensTip, lensColor);
    addLine(c[5], lensTip, lensColor);
    addLine(c[6], lensTip, lensColor);
    addLine(c[7], lensTip, lensColor);

    // 카메라 시야 절두체를 선으로 표시한다.
    float fovY   = glm::radians(60.f);
    float aspect = scExtent.width / (float)scExtent.height;
    glm::vec3 frustCol = {0.85f, 1.0f, 0.3f}; // 연두색

    auto makeRect = [&](float dist) {
        float hY = dist * std::tan(fovY * 0.5f);
        float hX = hY * aspect;
        glm::vec3 center = pos + fwd * dist;
        return std::array<glm::vec3,4>{
            center + right*(-hX) + up*(-hY),
            center + right*(+hX) + up*(-hY),
            center + right*(+hX) + up*(+hY),
            center + right*(-hX) + up*(+hY),
        };
    };

    auto nearR = makeRect(0.5f);
    auto farR  = makeRect(7.0f);

    // 카메라 위치에서 원평면 네 꼭짓점으로 선을 뻗는다.
    for (int i = 0; i < 4; ++i)
        addLine(pos, farR[i], frustCol);
    // near plane 사각형
    for (int i = 0; i < 4; ++i)
        addLine(nearR[i], nearR[(i+1)%4], frustCol);
    // far plane 사각형
    for (int i = 0; i < 4; ++i)
        addLine(farR[i], farR[(i+1)%4], frustCol);
    // near/far plane 연결선
    for (int i = 0; i < 4; ++i)
        addLine(nearR[i], farR[i], frustCol);

    // LOD/원거리 거리 링 (수평 원, 32 세그먼트)
    constexpr int   RING_SEGS = 32;
    constexpr float TWO_PI    = 6.28318530718f;
    auto addRing = [&](float radius, float yHeight, glm::vec3 col) {
        glm::vec3 center = glm::vec3(pos.x, yHeight, pos.z);
        for (int i = 0; i < RING_SEGS; ++i) {
            float a0 = TWO_PI * i       / RING_SEGS;
            float a1 = TWO_PI * (i + 1) / RING_SEGS;
            glm::vec3 p0 = center + glm::vec3(std::cos(a0)*radius, 0.f, std::sin(a0)*radius);
            glm::vec3 p1 = center + glm::vec3(std::cos(a1)*radius, 0.f, std::sin(a1)*radius);
            addLine(p0, p1, col);
        }
    };
    addRing(18.f, pos.y, {1.0f, 0.60f, 0.05f}); // 주황 = LOD1 전환 거리 (18 m)
    addRing(36.f, pos.y, {1.0f, 0.20f, 0.05f}); // 빨강 = LOD2 전환 거리 (36 m)
    addRing(viewDistMax, pos.y, {0.2f, 0.8f, 1.0f}); // 하늘색 = 원거리 컬링 한계

    gizmoVertCount = static_cast<uint32_t>(verts.size());
    memcpy(gizmoVBMapped[frame], verts.data(), verts.size() * sizeof(Vertex));
}

// 정적 렌더링 헬퍼
Frustum VulkanApp::extractFrustum(const glm::mat4& vp) {
    // Vulkan 클립 공간 규칙에 맞춰 평면을 추출한다.
    // -w <= x <= w, -w <= y <= w, 0 <= z <= w
    // GLM은 열 우선 저장이므로 명시적으로 행 벡터를 꺼낸다.
    auto row = [&](int r) {
        return glm::vec4(vp[0][r], vp[1][r], vp[2][r], vp[3][r]);
    };
    const glm::vec4 r0 = row(0);
    const glm::vec4 r1 = row(1);
    const glm::vec4 r2 = row(2);
    const glm::vec4 r3 = row(3);

    Frustum f;
    f.planes[0] = r3 + r0; // 왼쪽
    f.planes[1] = r3 - r0; // 오른쪽
    f.planes[2] = r3 + r1; // 아래
    f.planes[3] = r3 - r1; // 위
    f.planes[4] = r2;      // 가까운 평면
    f.planes[5] = r3 - r2; // 먼 평면
    // 평면 법선을 정규화한다.
    for (auto& p : f.planes) {
        float len = glm::length(glm::vec3(p));
        if (len > 0.f) p /= len;
    }
    return f;
}

bool VulkanApp::sphereInFrustum(const Frustum& f, glm::vec3 c, float r) {
    constexpr float kCullEpsilon = 1e-4f;
    for (const auto& p : f.planes)
        if (glm::dot(glm::vec3(p), c) + p.w < -r - kCullEpsilon)
            return false;
    return true;
}

void VulkanApp::computeBoundSphere(DrawObject& obj, glm::vec3 bmin, glm::vec3 bmax) {
    // AABB의 8개 꼭짓점을 모델 행렬로 월드 공간에 옮긴다.
    const glm::mat4& M = obj.push.model;
    glm::vec3 corners[8];
    int k = 0;
    for (float x : {bmin.x, bmax.x})
        for (float y : {bmin.y, bmax.y})
            for (float z : {bmin.z, bmax.z})
                corners[k++] = glm::vec3(M * glm::vec4(x, y, z, 1.f));

    // 바운딩 구 중심은 변환된 꼭짓점들의 평균으로 잡는다.
    glm::vec3 center{};
    for (auto& c : corners) center += c;
    center /= 8.f;

    // 반지름은 중심에서 가장 먼 꼭짓점까지의 거리다.
    float radius = 0.f;
    for (auto& c : corners)
        radius = std::max(radius, glm::distance(c, center));

    obj.boundCenter = center;
    obj.boundRadius = radius;
}

// 이미지와 이미지 뷰 헬퍼
void VulkanApp::createImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageTiling tiling,
                             VkImageUsageFlags usage, VkMemoryPropertyFlags props,
                             VkImage& img, VkDeviceMemory& mem) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType     = VK_IMAGE_TYPE_2D;
    ci.extent        = {w, h, 1};
    ci.mipLevels     = 1;
    ci.arrayLayers   = 1;
    ci.format        = fmt;
    ci.tiling        = tiling;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ci.usage         = usage;
    ci.samples       = VK_SAMPLE_COUNT_1_BIT;
    ci.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateImage(device, &ci, nullptr, &img) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImage failed");

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, img, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    if (vkAllocateMemory(device, &ai, nullptr, &mem) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory (image) failed");
    vkBindImageMemory(device, img, mem, 0);
}

VkImageView VulkanApp::createImageView(VkImage img, VkFormat fmt, VkImageAspectFlags aspect) {
    VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    ci.image                           = img;
    ci.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    ci.format                          = fmt;
    ci.subresourceRange.aspectMask     = aspect;
    ci.subresourceRange.baseMipLevel   = 0;
    ci.subresourceRange.levelCount     = 1;
    ci.subresourceRange.baseArrayLayer = 0;
    ci.subresourceRange.layerCount     = 1;
    VkImageView view;
    if (vkCreateImageView(device, &ci, nullptr, &view) != VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView failed");
    return view;
}

// 메인 루프
// 최적화 토글 입력 처리

// 텍스처 업로드용 일회성 커맨드 버퍼 헬퍼
static VkCommandBuffer beginSingleCmd(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd; vkAllocateCommandBuffers(device, &ai, &cmd);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    return cmd;
}
static void endSingleCmd(VkDevice dev, VkCommandPool pool, VkQueue q, VkCommandBuffer cmd) {
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(q);
    vkFreeCommandBuffers(dev, pool, 1, &cmd);
}
static void transitionImageLayout(VkCommandBuffer cmd, VkImage image,
                                   VkImageLayout oldL, VkImageLayout newL)
{
    VkImageMemoryBarrier bar{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    bar.oldLayout = oldL; bar.newLayout = newL;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.image = image;
    bar.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkPipelineStageFlags src, dst;
    if (oldL == VK_IMAGE_LAYOUT_UNDEFINED) {
        bar.srcAccessMask = 0;
        bar.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        src = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT; dst = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else {
        bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        src = VK_PIPELINE_STAGE_TRANSFER_BIT; dst = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    vkCmdPipelineBarrier(cmd, src, dst, 0, 0, nullptr, 0, nullptr, 1, &bar);
}

void VulkanApp::createDefaultTextureAndSampler() {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_TRUE; sci.maxAnisotropy = 4.0f;
    sci.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    if (vkCreateSampler(device, &sci, nullptr, &texSampler) != VK_SUCCESS)
        throw std::runtime_error("vkCreateSampler failed");

    const uint8_t white[4] = {255,255,255,255};
    createImage(1, 1, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, defaultTexImage, defaultTexMemory);
    VkBuffer sb; VkDeviceMemory sm;
    createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, sb, sm);
    void* mp; vkMapMemory(device, sm, 0, 4, 0, &mp); memcpy(mp, white, 4); vkUnmapMemory(device, sm);
    VkCommandBuffer cmd = beginSingleCmd(device, commandPool);
    transitionImageLayout(cmd, defaultTexImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy reg{}; reg.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1}; reg.imageExtent={1,1,1};
    vkCmdCopyBufferToImage(cmd, sb, defaultTexImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
    transitionImageLayout(cmd, defaultTexImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSingleCmd(device, commandPool, graphicsQueue, cmd);
    vkDestroyBuffer(device, sb, nullptr); vkFreeMemory(device, sm, nullptr);
    defaultTexView = createImageView(defaultTexImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
}

void VulkanApp::cleanupTextureResources() {
    for (auto& v : texViews)    vkDestroyImageView(device, v, nullptr);
    for (auto& i : texImages)   vkDestroyImage(device, i, nullptr);
    for (auto& mm : texMemories) vkFreeMemory(device, mm, nullptr);
    texViews.clear(); texImages.clear(); texMemories.clear();
}

void VulkanApp::createTextureResources() {
    cleanupTextureResources();
    if (!sceneTextures.empty()) {
        std::vector<VkBuffer>       sbs(sceneTextures.size(), VK_NULL_HANDLE);
        std::vector<VkDeviceMemory> sms(sceneTextures.size(), VK_NULL_HANDLE);
        texImages.resize(sceneTextures.size(),   VK_NULL_HANDLE);
        texMemories.resize(sceneTextures.size(), VK_NULL_HANDLE);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            const auto& td = sceneTextures[i];
            VkDeviceSize sz = (VkDeviceSize)td.width * td.height * 4;
            createImage((uint32_t)td.width,(uint32_t)td.height,VK_FORMAT_R8G8B8A8_SRGB,
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, texImages[i], texMemories[i]);
            createBuffer(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         sbs[i], sms[i]);
            void* p; vkMapMemory(device, sms[i], 0, sz, 0, &p);
            memcpy(p, td.pixels.data(), (size_t)sz); vkUnmapMemory(device, sms[i]);
        }
        VkCommandBuffer cmd = beginSingleCmd(device, commandPool);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            transitionImageLayout(cmd, texImages[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{}; r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};
            r.imageExtent = {(uint32_t)sceneTextures[i].width,(uint32_t)sceneTextures[i].height,1};
            vkCmdCopyBufferToImage(cmd, sbs[i], texImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            transitionImageLayout(cmd, texImages[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        endSingleCmd(device, commandPool, graphicsQueue, cmd);
        for (size_t i = 0; i < sceneTextures.size(); ++i) {
            vkDestroyBuffer(device, sbs[i], nullptr); vkFreeMemory(device, sms[i], nullptr);
        }
        texViews.resize(sceneTextures.size());
        for (size_t i = 0; i < sceneTextures.size(); ++i)
            texViews[i] = createImageView(texImages[i], VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    // descriptor set의 텍스처 배열을 새 이미지 뷰로 갱신한다.
    std::vector<VkDescriptorImageInfo> imgInfos(MAX_TEXTURES);
    for (uint32_t s = 0; s < MAX_TEXTURES; ++s) {
        imgInfos[s].sampler     = texSampler;
        imgInfos[s].imageView   = (s < texViews.size()) ? texViews[s] : defaultTexView;
        imgInfos[s].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        auto doWrite = [&](VkDescriptorSet dst) {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = dst; w.dstBinding = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w.descriptorCount = MAX_TEXTURES; w.pImageInfo = imgInfos.data();
            vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
        };
        doWrite(descSets[f]); doWrite(cullDescSets[f]);
    }
}

void VulkanApp::handleOptKeys() {
    // 키를 누르는 순간에만 토글되도록 이전 상태를 기억한다.
    static bool prevKeys[10] = {};
    struct { int key; bool* flag; } binds[] = {
        { GLFW_KEY_1, &optFlags.frustumCulling   },
        { GLFW_KEY_2, &optFlags.lod              },
        { GLFW_KEY_3, &optFlags.instancing       },
        { GLFW_KEY_4, &optFlags.backfaceCulling  },
        { GLFW_KEY_5, &optFlags.depthSort        },
        { GLFW_KEY_6, &optFlags.occlusionCulling },
        { GLFW_KEY_7, &optFlags.viewDistCulling  },
        { GLFW_KEY_8, &optFlags.smallCulling     },
        { GLFW_KEY_9, &optFlags.deferredShading  },
    };
    for (int i = 0; i < 9; ++i) {
        bool pressed = (glfwGetKey(window, binds[i].key) == GLFW_PRESS);
        if (pressed && !prevKeys[i]) {
            *binds[i].flag = !*binds[i].flag;
            if (binds[i].key == GLFW_KEY_6) {
                if (*binds[i].flag) // 방금 켜진 경우에만 워밍업
                    resetOcclusionState();
            }
        }
        prevKeys[i] = pressed;
    }

    // B 키로 바닥 밝기 모드를 토글한다.
    static bool prevB = false;
    bool bPressed = (glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS);
    if (bPressed && !prevB)
        darkFloor = !darkFloor;
    prevB = bPressed;

    // F 키로 far plane을 200과 5000 사이에서 전환한다.
    static bool prevF = false;
    bool fPressed = (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
    if (fPressed && !prevF)
        extendedFarPlane = !extendedFarPlane;
    prevF = fPressed;

    // L 키로 씬 조명 전체를 켜고 끈다.
    static bool prevL = false;
    bool lPressed = (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
    if (lPressed && !prevL) {
        sceneLightsOn = !sceneLightsOn;
        sceneLightDirty = true;
    }
    prevL = lPressed;

    // N 키로 환경광을 켜고 끈다.
    static bool prevN = false;
    bool nPressed = (glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS);
    if (nPressed && !prevN) {
        ambientOn = !ambientOn;
        sceneLightDirty = true;
    }
    prevN = nPressed;

    // V 키로 발광 재질을 켜고 끈다.
    static bool prevV = false;
    bool vPressed = (glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS);
    if (vPressed && !prevV) {
        emissiveOn = !emissiveOn;
        sceneLightDirty = true;
    }
    prevV = vPressed;

    // 0 key: 최적화 전체 ON/OFF 토글
    static bool prev0 = false;
    bool k0Pressed = (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS);
    if (k0Pressed && !prev0) {
        bool anyOn = optFlags.frustumCulling || optFlags.lod || optFlags.instancing
                  || optFlags.backfaceCulling || optFlags.depthSort || optFlags.occlusionCulling
                  || optFlags.viewDistCulling || optFlags.smallCulling || optFlags.deferredShading;
        bool val = !anyOn;
        optFlags.frustumCulling  = val;
        optFlags.lod             = val;
        optFlags.instancing      = val;
        optFlags.backfaceCulling = val;
        optFlags.depthSort       = val;
        optFlags.occlusionCulling = val;
        optFlags.viewDistCulling = val;
        optFlags.smallCulling    = val;
        optFlags.deferredShading = val;
        if (val) {
            resetOcclusionState();
        }
    }
    prev0 = k0Pressed;
}

void VulkanApp::mainLoop() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        float now = static_cast<float>(glfwGetTime());
        float dt  = now - lastFrameTime;
        lastFrameTime = now;

        // ESC를 누르면 앱을 종료한다.
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        // F11로 전체화면을 토글한다.
        {
            static bool prevF11 = false;
            bool f11 = (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS);
            if (f11 && !prevF11) toggleFullscreen();
            prevF11 = f11;
        }

        // Tab으로 다음 맵을 로드한다.
        {
            static bool prevTab = false;
            bool tab = (glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS);
            if (tab && !prevTab && availableMaps.size() > 1) {
                int nextIndex  = (currentMapIndex + 1) % (int)availableMaps.size();
                currentMapFile = availableMaps[nextIndex];
                currentMapIndex = nextIndex;
                reloadScene(); // 내부에서 오류를 처리하고 실패 시 이전 씬으로 되돌린다.
                if (optFlags.occlusionCulling) resetOcclusionState();
            }
            prevTab = tab;
        }

        // G 키로 ghost 관찰자 모드를 토글한다.
        {
            bool gPressed = (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS);
            if (gPressed && !prevGhostKey) {
                ghostMode = !ghostMode;
                if (ghostMode) {
                    // ghost 카메라를 배치 카메라 뒤쪽과 위쪽에 초기화한다.
                    glm::vec3 back = -camera.getCameraFront();
                    observerCamera.position = camera.position
                                           + back * 5.0f
                                           + glm::vec3(0.f, 2.0f, 0.f);
                    // ghost 카메라가 배치 카메라를 바라보게 한다.
                    glm::vec3 toPlaced = camera.position - observerCamera.position;
                    float hDist = glm::length(glm::vec2(toPlaced.x, toPlaced.z));
                    observerCamera.yaw   = glm::degrees(std::atan2f(toPlaced.z, toPlaced.x));
                    observerCamera.pitch = glm::degrees(std::atan2f(toPlaced.y, hDist));
                    observerCamera.normalSpeed    = camera.normalSpeed;
                    observerCamera.fastSpeed      = camera.fastSpeed;
                    observerCamera.mouseSensitivity = camera.mouseSensitivity;
                    observerCamera.cinematic      = camera.cinematic; // 시네마틱 모드를 같이 따라감
                    observerCamera.syncTarget();
                }
            }
            prevGhostKey = gPressed;
        }

        // C 키로 시네마틱 카메라(부드러운 이동/회전) 모드를 토글한다.
        {
            static bool prevC = false;
            bool cPressed = (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS);
            if (cPressed && !prevC) {
                bool newMode = !camera.cinematic;
                camera.cinematic         = newMode;
                observerCamera.cinematic = newMode;
                // 토글 순간 잔여 속도/타겟 보간이 카메라를 끌고가지 않도록 동기화
                camera.syncTarget();
                observerCamera.syncTarget();
            }
            prevC = cPressed;
        }

        // R/P 키 + 리플레이/녹화 로직: 자동 벤치마크 중에는 전부 건너뜀
        if (!autoBenchActive) {
            // R 키로 리플레이 녹화를 시작하거나 종료한다.
            static bool prevR = false;
            bool r = (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS);
            if (r && !prevR && !isReplaying) {
                if (!isRecording) startRecording();
                else              stopRecording();
            }
            prevR = r;

            // P 키로 리플레이 재생을 시작하거나 종료한다.
            static bool prevP = false;
            bool p = (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS);
            if (p && !prevP) {
                if (!isReplaying) startReplay();
                else              stopReplay();
            }
            prevP = p;
        }

        // 녹화 중에는 일정 간격으로 카메라 상태를 저장한다.
        if (isRecording) {
            float elapsed = now - recordStartTime;
            if (recordedFrames.empty() || elapsed - recordedFrames.back().time >= 0.014f) {
                recordedFrames.push_back({elapsed, camera.position,
                                          camera.yaw, camera.pitch});
            }
        }

        // 리플레이 중에는 시간에 맞춰 카메라 상태를 보간한다.
        bool prevIsReplaying = isReplaying;
        if (isReplaying && !replayFrames.empty()) {
            float elapsed = now - replayStartTime;

            // 현재 재생 시간보다 늦지 않은 마지막 프레임까지 인덱스를 진행한다.
            while (replayFrameIdx + 1 < (int)replayFrames.size() &&
                   replayFrames[replayFrameIdx + 1].time <= elapsed)
                ++replayFrameIdx;

            if (replayFrameIdx + 1 < (int)replayFrames.size()) {
                // 인접한 두 리플레이 프레임 사이를 선형 보간한다.
                const auto& f0 = replayFrames[replayFrameIdx];
                const auto& f1 = replayFrames[replayFrameIdx + 1];
                float span = f1.time - f0.time;
                float t    = (span > 0.f) ? glm::clamp((elapsed - f0.time) / span, 0.f, 1.f) : 1.f;
                camera.position = glm::mix(f0.pos,   f1.pos,   t);
                camera.yaw      = f0.yaw   + t * (f1.yaw   - f0.yaw);
                camera.pitch    = f0.pitch + t * (f1.pitch - f0.pitch);
                camera.syncTarget(); // 리플레이가 카메라를 점프시켜도 보간이 따라가지 않도록
            } else {
                // 마지막 프레임에 도달하면 리플레이를 종료한다.
                const auto& last = replayFrames.back();
                camera.position = last.pos;
                camera.yaw      = last.yaw;
                camera.pitch    = last.pitch;
                camera.syncTarget();
                stopReplay();
            }
        }

        // 자동 벤치마크: 리플레이 종료 감지 -> 다음 실험/반복으로 진행
        if (autoBenchActive && prevIsReplaying && !isReplaying)
            onAutoBenchRunEnd();

        // 카메라 입력 처리
        if (ghostMode) {
            observerCamera.processKeyboard(window, dt);
            // 리플레이 중이 아닐 때만 방향키로 배치 카메라를 움직인다.
            if (!isReplaying) camera.processArrowKeys(window, dt);
        } else {
            // 리플레이 중에는 카메라를 재생 데이터가 제어한다.
            if (!isReplaying) camera.processKeyboard(window, dt);
        }

        // M 키는 리플레이가 있으면 자동 벤치마크, 없으면 5초 벤치마크를 시작한다.
        if (!autoBenchActive) {
            static bool prevM = false;
            bool m = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            if (m && !prevM) {
                if (!findReplayFiles(replayDir).empty()) {
                    startAutoBenchmark();
                } else {
                    if (!benchmarkActive) startBenchmark();
                    else                  finishBenchmark();
                }
            }
            prevM = m;
        } else {
            // 자동 벤치마크 진행 중: M 키로 중단
            static bool prevM = false;
            bool m = (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS);
            if (m && !prevM) {
                printf("[AutoBench] Aborted by user.\n");
                if (isReplaying) stopReplay();
                autoBenchActive = false;
                optFlags        = autoBenchSavedFlags;
                ghostMode       = autoBenchSavedGhost;
            }
            prevM = m;
        }

        // T 키로 스트레스 배율을 순환한다.
        if (!autoBenchActive) {
            static bool prevT = false;
            bool tk = (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS);
            if (tk && !prevT) {
                stressLevel = (stressLevel + 1) % 5;
                applyStress();
            }
            prevT = tk;
        }

        // 자동 벤치마크 중에는 숫자 키(1~9) 토글 비활성화
        if (!autoBenchActive) handleOptKeys();

        perfStats.update(dt);
        const bool validBenchFrame = dt > 0.0f && dt < 0.25f;
        const float frameFps = (perfStats.frameTimeMs > 0.0f)
            ? 1000.0f / perfStats.frameTimeMs
            : 0.0f;

        // 프레임 시간 히스토리 갱신 (고정 링버퍼)
        frameTimeHistBuf[frameTimeHistIdx] = dt * 1000.f;
        frameTimeHistIdx = (frameTimeHistIdx + 1) % FRAME_HISTORY_SIZE;
        if (frameTimeHistCount < FRAME_HISTORY_SIZE) ++frameTimeHistCount;

        // 수동 벤치마크 샘플 수집 (5초 단순 측정)
        if (benchmarkActive) {
            if (validBenchFrame) {
                benchmarkElapsed += dt;
                benchmarkSamples.push_back({
                    frameFps,
                    perfStats.frameTimeMs,
                    perfStats.cpuPercent,
                    perfStats.gpuPercent,
                    renderedCount,
                    culledCount
                });
            }
            if (benchmarkElapsed >= benchmarkDuration)
                finishBenchmark();
        }

        // 자동 벤치마크 샘플 수집 (리플레이 재생 중)
        if (autoBenchActive && isReplaying &&
            autoBenchExpIdx < (int)autoBenchExps.size()) {
            if (autoBenchSkipFrames > 0) {
                --autoBenchSkipFrames;
            } else if (validBenchFrame) {
                autoBenchExps[autoBenchExpIdx].current.push_back({
                    frameFps,
                    perfStats.frameTimeMs,
                    perfStats.cpuPercent,
                    perfStats.gpuPercent,
                    renderedCount,
                    culledCount
                });
            }
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawStatsOverlay();
        ImGui::Render();

        drawFrame();
    }
    vkDeviceWaitIdle(device);
}

// 프레임 갱신과 커맨드 기록
glm::mat4 VulkanApp::computeLightMatrix() const {
    // 그림자를 드리울 방향광을 고른다.
    // GLTF 조명이 활성화돼 있으면 첫 방향광을, 아니면 셰이더 기본 태양(lightYaw 반영)을 쓴다.
    glm::vec3 dir(0.0f);
    bool haveDir = false;
    if (sceneLightsOn && !sceneLights.empty()) {
        for (const SceneLight& sl : sceneLights) {
            if (sl.type == SceneLight::Directional && sl.enabled) {
                dir = glm::normalize(sl.direction);
                haveDir = true;
                break;
            }
        }
    }
    if (!haveDir) {
        glm::vec3 baseDir = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
        glm::mat4 rotY    = glm::rotate(glm::mat4(1.f), glm::radians(lightYaw), glm::vec3(0, 1, 0));
        dir = glm::normalize(glm::vec3(rotY * glm::vec4(baseDir, 0.f)));
    }

    // 씬 바운딩 구를 감싸는 ortho 절두체를 라이트 방향으로 정렬한다.
    const glm::vec3 center = sceneBoundsCenter;
    const float     r      = sceneBoundsRadius;
    const glm::vec3 eye    = center - dir * (r * 2.0f); // 광원을 구 밖(방향 반대쪽)에 둔다.
    const glm::vec3 up     = (std::abs(dir.y) > 0.99f) ? glm::vec3(0, 0, 1) : glm::vec3(0, 1, 0);

    glm::mat4 view = glm::lookAt(eye, center, up);
    // GLM_FORCE_DEPTH_ZERO_TO_ONE 가 정의돼 ortho 가 0..1 깊이(Vulkan 규칙)를 만든다.
    glm::mat4 proj = glm::ortho(-r, r, -r, r, 0.1f, r * 4.0f);
    return proj * view;
}

void VulkanApp::updateUniformBuffer(uint32_t frameIndex) {
    // ghost 모드에서는 observerCamera, 일반 모드에서는 배치 카메라로 렌더링한다.
    const Camera& renderCam = ghostMode ? observerCamera : camera;
    const Camera& cullCam   = camera;

    // 이번 프레임의 라이트 뷰·투영을 한 번 계산해 섀도 패스(recordCommandBuffer)에서 재사용한다.
    currentLightVP = computeLightMatrix();

    auto writeCameraUbo = [&](void* dst, const Camera& cam) {
        CameraUBO ubo;
        ubo.view      = cam.getViewMatrix();
        ubo.proj      = glm::perspective(glm::radians(60.0f),
                                         scExtent.width / (float)scExtent.height,
                                         0.1f, getFarPlane());
        ubo.proj[1][1] *= -1; // Vulkan NDC 규칙에 맞춰 Y축을 뒤집는다.
        ubo.cameraPos  = glm::vec4(cam.position, darkFloor ? 1.0f : 0.0f);

        glm::vec3 baseDir = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.3f));
        glm::mat4 rotY    = glm::rotate(glm::mat4(1.f), glm::radians(lightYaw), glm::vec3(0,1,0));
        ubo.lightDir      = rotY * glm::vec4(baseDir, 0.f);
        ubo.lightVP       = currentLightVP;

        memcpy(dst, &ubo, sizeof(ubo));
    };

    writeCameraUbo(uniformBuffersMapped[frameIndex], renderCam);
    writeCameraUbo(cullUniformBuffersMapped[frameIndex], cullCam);

    // SceneLightUBO 갱신 (dirty flag: 변경 시에만 memcpy)
    if (sceneLightDirty) {
        SceneLightUBO lightUbo{};
        lightUbo.useSceneLights = (!sceneLights.empty() && sceneLightsOn) ? 1 : 0;
        lightUbo.ambientOn      = ambientOn  ? 1 : 0;
        lightUbo.emissiveOn     = emissiveOn ? 1 : 0;

        int gpuIdx = 0;
        for (const SceneLight& sl : sceneLights) {
            if (gpuIdx >= MAX_SCENE_LIGHTS) break;
            GpuSceneLight& gl = lightUbo.lights[gpuIdx++];
            if (sl.type == SceneLight::Directional) {
                gl.posRange  = glm::vec4(sl.direction, 0.f);
                gl.dirType   = glm::vec4(sl.direction, 1.f);
            } else {
                gl.posRange  = glm::vec4(sl.position, sl.range);
                gl.dirType   = glm::vec4(sl.direction, (sl.type == SceneLight::Spot) ? 2.f : 0.f);
            }
            gl.colorEnab = glm::vec4(sl.color * sl.intensity,
                                     (sceneLightsOn && sl.enabled) ? 1.f : 0.f);
        }
        lightUbo.numLights = gpuIdx;
        // 더블 버퍼 모두 업데이트
        for (int fi = 0; fi < MAX_FRAMES_IN_FLIGHT; ++fi)
            memcpy(sceneLightUBOMapped[fi], &lightUbo, sizeof(lightUbo));
        sceneLightDirty = false;
    }
}

void VulkanApp::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &begin);

    const bool occlusionEnabled = optFlags.occlusionCulling
                               && !optFlags.deferredShading
                               && occlusionQueryPool != VK_NULL_HANDLE
                               && occQueryCount > 0;
    const Camera& renderCam = ghostMode ? observerCamera : camera;
    const Camera& cullCam   = camera;

    if (occlusionEnabled) {
        uint32_t base = currentFrame * occQueryCount;
        vkCmdResetQueryPool(cmd, occlusionQueryPool, base, occQueryCount);
    }

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color        = {{0.53f, 0.68f, 0.85f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rpBegin.renderPass        = renderPass;
    rpBegin.framebuffer       = framebuffers[imageIndex];
    rpBegin.renderArea.extent = scExtent;
    rpBegin.clearValueCount   = static_cast<uint32_t>(clearValues.size());
    rpBegin.pClearValues      = clearValues.data();

    VkViewport vp{0.f, 0.f, (float)scExtent.width, (float)scExtent.height, 0.f, 1.f};
    VkRect2D   sc{{0, 0}, scExtent};
    VkBuffer     vbs[]  = {vertexBuffer};
    VkDeviceSize offs[] = {0};

    // 방향광 섀도맵: 라이트 시점에서 불투명 캐스터의 깊이를 먼저 기록한다.
    // forward·deferred 양쪽 모두 set0 binding3로 이 결과를 샘플링한다.
    if (shadowPipeline != VK_NULL_HANDLE && !drawObjects.empty()) {
        VkClearValue shadowClear{};
        shadowClear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo spBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        spBegin.renderPass        = shadowRenderPass;
        spBegin.framebuffer       = shadowFramebuffers[currentFrame];
        spBegin.renderArea.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE};
        spBegin.clearValueCount   = 1;
        spBegin.pClearValues      = &shadowClear;

        vkCmdBeginRenderPass(cmd, &spBegin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport sVp{0.f, 0.f, (float)SHADOW_MAP_SIZE, (float)SHADOW_MAP_SIZE, 0.f, 1.f};
        VkRect2D   sSc{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};
        vkCmdSetViewport(cmd, 0, 1, &sVp);
        vkCmdSetScissor(cmd, 0, 1, &sSc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        for (const DrawObject& obj : drawObjects) {
            if (obj.push.baseColor.w < 0.999f) continue; // 투명체는 캐스터에서 제외한다.
            glm::mat4 lightMVP = currentLightVP * obj.push.model;
            vkCmdPushConstants(cmd, shadowPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &lightMVP);
            vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
        }
        vkCmdEndRenderPass(cmd);
    }

    Frustum frustum{};
    if (optFlags.frustumCulling || optFlags.occlusionCulling) {
        glm::mat4 view = cullCam.getViewMatrix();
        glm::mat4 proj = glm::perspective(glm::radians(60.f),
                                          scExtent.width / (float)scExtent.height,
                                          0.1f, getFarPlane());
        proj[1][1] *= -1;
        frustum = extractFrustum(proj * view);
    }

    // 프레임마다 힙 할당하지 않도록 frameOrder를 재사용한다.
    frameOrder.resize(drawObjects.size());
    std::iota(frameOrder.begin(), frameOrder.end(), 0);
    if (optFlags.depthSort || optFlags.occlusionCulling) {
        glm::vec3 camPos = cullCam.position;
        // sqrt 없이 제곱 거리로 같은 정렬 순서를 얻는다.
        std::sort(frameOrder.begin(), frameOrder.end(), [&](int a, int b) {
            glm::vec3 da = drawObjects[a].boundCenter - camPos;
            glm::vec3 db = drawObjects[b].boundCenter - camPos;
            return glm::dot(da, da) < glm::dot(db, db);
        });
    }
    const std::vector<int>& order = frameOrder;
    frameInstancedMask.clear();

    // 통합 컬링 체크: frustum + viewDist + smallObj 를 한 번에 수행
    // 3개 lambda 캡처·호출 오버헤드 제거
    constexpr float kTanHalfFov = 0.57735f; // tan(30°)
    const bool doFrustum  = optFlags.frustumCulling;
    const bool doViewDist = optFlags.viewDistCulling;
    const bool doSmall    = optFlags.smallCulling;
    const glm::vec3 cullPos = cullCam.position;
    const float halfH = scExtent.height * 0.5f;

    auto isCulled = [&](const DrawObject& obj) -> bool {
        if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
            return true;
        if (doViewDist) {
            float dist = glm::length(cullPos - obj.boundCenter) - obj.boundRadius;
            if (dist > viewDistMax) return true;
        }
        if (doSmall) {
            float dist = glm::length(cullPos - obj.boundCenter);
            if (dist >= 1e-4f && obj.boundRadius / dist * halfH / kTanHalfFov < smallCullPx)
                return true;
        }
        return false;
    };

    // Ghost 모드에서 컬링 이유별 색상을 반환한다. 컬링되지 않으면 alpha=0이다.
    auto ghostCullColor = [&](const DrawObject& obj) -> glm::vec4 {
        if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
            return {1.0f, 0.15f, 0.15f, 0.13f};
        if (doViewDist) {
            float dist = glm::length(cullPos - obj.boundCenter) - obj.boundRadius;
            if (dist > viewDistMax) return {1.0f, 0.55f, 0.05f, 0.13f};
        }
        if (doSmall) {
            float dist = glm::length(cullPos - obj.boundCenter);
            if (dist >= 1e-4f && obj.boundRadius / dist * halfH / kTanHalfFov < smallCullPx)
                return {1.0f, 0.95f, 0.05f, 0.13f};
        }
        return {0.f, 0.f, 0.f, 0.f};
    };

    auto beginScenePass = [&](VkDescriptorSet camSet, VkPipeline pipe) {
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &camSet, 0, nullptr);
    };

    auto selectLod = [&](const DrawObject& obj,
                         uint32_t& idxStart,
                         uint32_t& idxCount,
                         glm::mat4& model,
                         int* outLodLevel = nullptr) -> bool {
        idxStart = obj.indexStart;
        idxCount = obj.indexCount;
        model    = obj.push.model;
        int level = 0;

        if (optFlags.lod && obj.numLods > 0) {
            // 큰 오브젝트가 바로 앞에 있어도 중심점이 멀다는 이유로 LOD가 바뀌지 않도록
            // 바운딩 구 표면까지의 거리로 전환 임계값을 비교한다.
            float dist = glm::length(cullCam.position - obj.boundCenter) - obj.boundRadius;
            dist = std::max(0.0f, dist);
            if (dist > obj.lodDist[1] && obj.numLods >= 2) {
                if (obj.lods[1].count == 0) return false;
                idxStart = obj.lods[1].start;
                idxCount = obj.lods[1].count;
                model    = obj.lods[1].model;
                level    = 2;
            } else if (dist > obj.lodDist[0]) {
                idxStart = obj.lods[0].start;
                idxCount = obj.lods[0].count;
                model    = obj.lods[0].model;
                level    = 1;
            }
        }
        if (outLodLevel) *outLodLevel = level;
        return true;
    };

    auto issueOcclusionQueries = [&](VkDescriptorSet camSet) {
        if (!occlusionEnabled) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineQueryOnly);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &camSet, 0, nullptr);

        for (int idx : order) {
            auto& obj = drawObjects[idx];
            if (obj.skipOcclusion) continue;
            if (obj.push.baseColor.w < 0.999f) continue;
            if (optFlags.frustumCulling &&
                !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius)) continue;

            // 바운딩 구 프록시: 단위 큐브를 boundCenter 로 이동, boundRadius 로 스케일
            PushConstants proxyPc = obj.push;
            proxyPc.model = glm::translate(glm::mat4(1.f), obj.boundCenter)
                          * glm::scale(glm::mat4(1.f), glm::vec3(obj.boundRadius));
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &proxyPc);

            uint32_t queryIndex = currentFrame * occQueryCount + (uint32_t)idx;
            vkCmdBeginQuery(cmd, occlusionQueryPool, queryIndex, 0);
            vkCmdDrawIndexed(cmd, occBBoxMesh.indexCount, 1, occBBoxMesh.indexStart, 0, 0);
            vkCmdEndQuery(cmd, occlusionQueryPool, queryIndex);
        }
    };

    if (ghostMode && occlusionEnabled) {
        beginScenePass(cullDescSets[currentFrame], graphicsPipelineNoCull);
        VkPipeline boundGhostPipe = VK_NULL_HANDLE;

        for (int idx : order) {
            auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue;
            if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius))
                continue;

            uint32_t idxStart = obj.indexStart;
            uint32_t idxCount = obj.indexCount;
            glm::mat4 model   = obj.push.model;
            if (!selectLod(obj, idxStart, idxCount, model)) continue;

            PushConstants pc = obj.push;
            pc.model = model;
            VkPipeline pipe = (!optFlags.backfaceCulling || obj.twoSided)
                ? graphicsPipelineNoCull
                : (obj.reverseFrontFace ? graphicsPipelineFlippedCull : graphicsPipeline);
            if (pipe != boundGhostPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                boundGhostPipe = pipe;
            }
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &pc);
            vkCmdDrawIndexed(cmd, idxCount, 1, idxStart, 0, 0);
        }

        issueOcclusionQueries(cullDescSets[currentFrame]);
        vkCmdEndRenderPass(cmd);
    }

    VkPipeline boundOpaquePipe = VK_NULL_HANDLE; // forward/deferred 분기에서 공유하는 현재 파이프라인 캐시
    // 디퍼드 렌더링 파이프라인을 만든다.
    if (optFlags.deferredShading && gbufRenderPass != VK_NULL_HANDLE) {
        std::array<VkClearValue, GBUFFER_COLOR_ATTACHMENTS + 1> gbClear{};
        gbClear[0].color = {{0,0,0,0}};
        gbClear[1].color = {{0,0,0,0}};
        gbClear[2].color = {{0,0,0,0}};
        gbClear[3].color = {{0,0,0,0}};
        gbClear[4].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo gbRpBegin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        gbRpBegin.renderPass        = gbufRenderPass;
        gbRpBegin.framebuffer       = gbufFramebuffers[currentFrame];
        gbRpBegin.renderArea.extent = scExtent;
        gbRpBegin.clearValueCount   = static_cast<uint32_t>(gbClear.size());
        gbRpBegin.pClearValues      = gbClear.data();

        vkCmdBeginRenderPass(cmd, &gbRpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

        VkPipeline gbCurPipe = VK_NULL_HANDLE;
        VkPipeline gbDefault = optFlags.backfaceCulling ? gbufPipeline : gbufPipelineNoCull;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gbDefault);
        gbCurPipe = gbDefault;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

        for (int idx : order) {
            const auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue; // 투명 오브젝트는 이 패스에서 제외한다.
            if (isCulled(obj)) continue;

            uint32_t  gbIdxStart = obj.indexStart;
            uint32_t  gbIdxCount = obj.indexCount;
            glm::mat4 gbModel    = obj.push.model;
            if (!selectLod(obj, gbIdxStart, gbIdxCount, gbModel)) continue;

            PushConstants gbPc = obj.push;
            gbPc.model = gbModel;

            VkPipeline wantPipe = (!optFlags.backfaceCulling || obj.twoSided || obj.reverseFrontFace)
                                  ? gbufPipelineNoCull : gbufPipeline;
            if (wantPipe != gbCurPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, wantPipe);
                gbCurPipe = wantPipe;
            }
            vkCmdPushConstants(cmd, pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &gbPc);
            vkCmdDrawIndexed(cmd, gbIdxCount, 1, gbIdxStart, 0, 0);
        }
        vkCmdEndRenderPass(cmd);

        // G-Buffer 컬러 attachment를 조명 패스에서 샘플링 가능한 레이아웃으로 전환한다.
        std::array<VkImageMemoryBarrier, GBUFFER_COLOR_ATTACHMENTS> gbBarriers{};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            gbBarriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            gbBarriers[i].oldLayout           = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            gbBarriers[i].newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            gbBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            gbBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            gbBarriers[i].image               = gbufImages[i][currentFrame];
            gbBarriers[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            gbBarriers[i].srcAccessMask       = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            gbBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        }
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, nullptr, 0, nullptr,
            GBUFFER_COLOR_ATTACHMENTS, gbBarriers.data());
    }

    // 메인 렌더패스: 디퍼드 조명 삼각형 또는 forward 씬을 그린다.
    if (optFlags.deferredShading && gbufRenderPass != VK_NULL_HANDLE) {
        // 메인 렌더 패스를 시작하고 전체화면 조명 삼각형을 그린다.
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdSetViewport(cmd, 0, 1, &vp);
        vkCmdSetScissor(cmd, 0, 1, &sc);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferredLightPipeline);
        VkDescriptorSet dSets[] = {descSets[currentFrame], deferredDescSets[currentFrame]};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                deferredLightLayout, 0, 2, dSets, 0, nullptr);
        vkCmdDraw(cmd, 3, 1, 0, 0); // 버텍스 버퍼 없이 전체화면 삼각형을 그린다.

        // 후속 투명/ghost/기즈모 패스를 위해 버텍스 버퍼를 다시 바인딩한다.
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindIndexBuffer(cmd, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineNoCull);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

        renderedCount       = (int)drawObjects.size();
        culledCount         = 0;
        instancedGroupCount = 0;
     } else {
    renderedCount       = 0;
    culledCount         = 0;
    instancedGroupCount = 0;
    frameInstMats.clear();
    frameInstObjectIndices.clear();
    frameInstDrawEntries.clear();
    if (optFlags.instancing)
        frameInstancedMask.assign(drawObjects.size(), 0u);

    if (optFlags.instancing && !instGroupDefs.empty()) {
        // 미리 만든 인스턴스 그룹 멤버십을 사용한다.
        // 매 프레임 힙 할당을 줄이기 위해 frameInstMats를 재사용한다.
        frameInstDrawEntries.reserve(instGroupDefs.size());

        bool instanceBufferFull = false;
        for (uint32_t di = 0; di < (uint32_t)instGroupDefs.size() && !instanceBufferFull; ++di) {
            const InstGroupDef& grp = instGroupDefs[di];
            uint32_t matrixOffset = (uint32_t)frameInstMats.size();
            uint32_t objectOffset = (uint32_t)frameInstObjectIndices.size();
            uint32_t count = 0;
            uint32_t groupCulled = 0;

            for (int idx : grp.members) {
                auto& obj = drawObjects[idx];
                if (obj.instanceGroupId < 0 || obj.twoSided || obj.reverseFrontFace) continue;
                if (obj.push.baseColor.w < 0.999f) continue;
                if (isCulled(obj)) {
                    groupCulled++;
                    continue;
                }
                if (occlusionEnabled && !obj.skipOcclusion && isOcclusionHidden(idx)) {
                    groupCulled++;
                    continue;
                }
                if ((uint32_t)frameInstMats.size() >= (uint32_t)MAX_INSTANCES) {
                    instanceBufferFull = true;
                    break;
                }
                frameInstMats.push_back(obj.push.model);
                frameInstObjectIndices.push_back(idx);
                ++count;
            }

            if (count >= (uint32_t)MIN_INSTANCES_PER_DRAW) {
                frameInstDrawEntries.push_back({matrixOffset, count, di});
                culledCount += (int)groupCulled;
                for (uint32_t oi = objectOffset; oi < (uint32_t)frameInstObjectIndices.size(); ++oi)
                    frameInstancedMask[frameInstObjectIndices[oi]] = 1u;
            } else {
                frameInstMats.resize(matrixOffset);
                frameInstObjectIndices.resize(objectOffset);
            }
        }

        if (!frameInstMats.empty()) {
            const VkDeviceSize instBytes = frameInstMats.size() * sizeof(glm::mat4);
            memcpy(instanceSSBOMapped[currentFrame],
                   frameInstMats.data(),
                   static_cast<size_t>(instBytes));

            VkBufferCopy instCopy{0, 0, instBytes};
            vkCmdCopyBuffer(cmd, instanceStagingBuffers[currentFrame],
                            instanceSSBOs[currentFrame], 1, &instCopy);
            VkBufferMemoryBarrier instBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            instBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            instBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
            instBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            instBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            instBarrier.buffer = instanceSSBOs[currentFrame];
            instBarrier.offset = 0;
            instBarrier.size = instBytes;
            vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                                 0, 0, nullptr, 1, &instBarrier, 0, nullptr);
        }
    }

    beginScenePass(descSets[currentFrame], graphicsPipelineNoCull);

    if (!frameInstMats.empty()) {
        VkBuffer instVbs[] = { vertexBuffer, instanceSSBOs[currentFrame] };
        VkDeviceSize instOffs[] = { 0, 0 };
        vkCmdBindVertexBuffers(cmd, 0, 2, instVbs, instOffs);

        VkPipeline instPipe = optFlags.backfaceCulling
                            ? graphicsPipelineInst : graphicsPipelineInstNoCull;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, instPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                instancePipelineLayout, 0, 1,
                                &descSets[currentFrame], 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                instancePipelineLayout, 1, 1,
                                &instanceDescSets[currentFrame], 0, nullptr);

        for (const InstDrawEntry& e : frameInstDrawEntries) {
            const InstGroupDef& grp = instGroupDefs[e.groupDefIndex];
            vkCmdPushConstants(cmd, instancePipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(PushConstants), &grp.push);
            vkCmdDrawIndexed(cmd, grp.indexCount, e.instanceCount,
                             grp.indexStart, 0, e.matrixOffset);
            renderedCount++;
            instancedGroupCount++;
        }

        // 남은 개별 오브젝트 렌더링을 위해 일반 파이프라인으로 되돌린다.
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offs);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineNoCull);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1,
                                &descSets[currentFrame], 0, nullptr);
        boundOpaquePipe = VK_NULL_HANDLE;
    }

    for (int idx : order) {
        auto& obj = drawObjects[idx];
        if (optFlags.instancing && idx < (int)frameInstancedMask.size() && frameInstancedMask[idx])
            continue;
        if (obj.push.baseColor.w < 0.999f) continue;

        if (isCulled(obj)) {
            culledCount++;
            continue;
        }

        if (occlusionEnabled && !obj.skipOcclusion && isOcclusionHidden(idx)) {
            culledCount++;
            continue;
        }

        uint32_t  idxStart = obj.indexStart;
        uint32_t  idxCount = obj.indexCount;
        glm::mat4 model    = obj.push.model;
        int lodLevel = 0;
        if (!selectLod(obj, idxStart, idxCount, model, &lodLevel)) {
            culledCount++;
            continue;
        }

        PushConstants pc = obj.push;
        pc.model = model;

        // Ghost 모드에서는 LOD 레벨에 따라 색상을 구분한다(LOD1=주황, LOD2=빨강).
        if (ghostMode && optFlags.lod && lodLevel > 0) {
            pc.textureIndex = -1.0f;
            pc.baseColor = (lodLevel == 1)
                ? glm::vec4(1.0f, 0.60f, 0.05f, 1.0f) // 주황 = LOD1
                : glm::vec4(1.0f, 0.20f, 0.05f, 1.0f); // 빨강 = LOD2
        }

        VkPipeline pipe = (!optFlags.backfaceCulling || obj.twoSided)
            ? graphicsPipelineNoCull
            : (obj.reverseFrontFace ? graphicsPipelineFlippedCull : graphicsPipeline);
        if (pipe != boundOpaquePipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            boundOpaquePipe = pipe;
        }
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdDrawIndexed(cmd, idxCount, 1, idxStart, 0, 0);
        renderedCount++;
    }

    } // forward 전용 렌더링 블록 끝

    // Ghost 모드에서는 컬링된 오브젝트를 반투명 컬러 오버레이로 표시한다.
    // frustum 컬링 -> 빨강 / 원거리 컬링 -> 주황 / 소형 컬링 -> 노랑
    if (ghostMode && (optFlags.frustumCulling || optFlags.viewDistCulling || optFlags.smallCulling)) {
        // 알파 블렌딩이 맞도록 뒤에서 앞으로 정렬한다.
        std::vector<int> culledOverlay;
        for (int idx : order) {
            const auto& obj = drawObjects[idx];
            if (obj.push.baseColor.w < 0.999f) continue;
            if (optFlags.instancing && idx < (int)frameInstancedMask.size() && frameInstancedMask[idx])
                continue;
            glm::vec4 col = ghostCullColor(obj);
            if (col.a > 0.f) culledOverlay.push_back(idx);
        }
        if (!culledOverlay.empty()) {
            glm::vec3 obsPos = observerCamera.position;
            std::sort(culledOverlay.begin(), culledOverlay.end(), [&](int a, int b) {
                return glm::distance(drawObjects[a].boundCenter, obsPos)
                     > glm::distance(drawObjects[b].boundCenter, obsPos);
            });
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineAlpha);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);
            boundOpaquePipe = VK_NULL_HANDLE;
            for (int idx : culledOverlay) {
                const auto& obj = drawObjects[idx];
                PushConstants pc = obj.push;
                pc.baseColor     = ghostCullColor(obj);
                pc.textureIndex  = -1.0f;
                vkCmdPushConstants(cmd, pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(PushConstants), &pc);
                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
            }
        }
    }

    if (!ghostMode && occlusionEnabled)
        issueOcclusionQueries(descSets[currentFrame]);

    {
        std::vector<int> transOrder;
        for (int i = 0; i < (int)drawObjects.size(); ++i) {
            if (drawObjects[i].push.baseColor.w < 0.999f)
                transOrder.push_back(i);
        }
        if (!transOrder.empty()) {
            glm::vec3 camPos = renderCam.position;
            std::sort(transOrder.begin(), transOrder.end(), [&](int a, int b) {
                float da = glm::distance(drawObjects[a].boundCenter, camPos);
                float db = glm::distance(drawObjects[b].boundCenter, camPos);
                return da > db;
            });

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineAlpha);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);

            for (int idx : transOrder) {
                auto& obj = drawObjects[idx];
                if (doFrustum && !sphereInFrustum(frustum, obj.boundCenter, obj.boundRadius)) {
                    culledCount++;
                    continue;
                }

                vkCmdPushConstants(cmd, pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(PushConstants), &obj.push);
                vkCmdDrawIndexed(cmd, obj.indexCount, 1, obj.indexStart, 0, 0);
            }
        }
    }

    if (ghostMode && gizmoVertCount > 0) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, gizmoPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                pipelineLayout, 0, 1, &descSets[currentFrame], 0, nullptr);
        VkBuffer     gVbs[]  = { gizmoVBs[currentFrame] };
        VkDeviceSize gOffs[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, gVbs, gOffs);
        vkCmdDraw(cmd, gizmoVertCount, 1, 0, 0);
    }

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

void VulkanApp::drawFrame() {
    vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
                                             imageAvailableSems[currentFrame], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) { recreateSwapChain(); return; }

    vkResetFences(device, 1, &inFlightFences[currentFrame]);
    vkResetCommandBuffer(commandBuffers[currentFrame], 0);

    // 현재 프레임 슬롯의 이전 오클루전 쿼리 결과를 읽는다.
    // 위의 fence 대기로 해당 GPU 작업이 끝났음을 보장한다.
    if (optFlags.occlusionCulling
        && occlusionQueryPool != VK_NULL_HANDLE
        && occQueryCount > 0
        && !occResults.empty()) {
        if (occVisible.size() != drawObjects.size() ||
            occZeroStreak.size() != drawObjects.size() ||
            occQueryBuf.size() != drawObjects.size() * 2) {
            resetOcclusionState();
        }
        if (occWarmupFrames > 0) {
            --occWarmupFrames; // GPU 쿼리가 아직 초기화되지 않은 warmup 프레임은 읽지 않는다.
        } else {
            uint32_t base  = currentFrame * occQueryCount;
            uint32_t count = std::min(occQueryCount, (uint32_t)occResults.size());
            std::fill(occQueryBuf.begin(), occQueryBuf.end(), 0);
            VkResult queryResult = vkGetQueryPoolResults(
                device, occlusionQueryPool, base, count,
                occQueryBuf.size() * sizeof(uint64_t), occQueryBuf.data(),
                2 * sizeof(uint64_t),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
            (void)queryResult;
            for (uint32_t i = 0; i < count; ++i) {
                if (occQueryBuf[i * 2 + 1] == 0) continue;

                const uint64_t samples = occQueryBuf[i * 2];
                occResults[i] = samples;

                const bool eligible =
                    i < drawObjects.size() &&
                    !drawObjects[i].skipOcclusion &&
                    drawObjects[i].push.baseColor.w >= 0.999f;
                if (!eligible || samples > 0) {
                    occVisible[i] = 1u;
                    occZeroStreak[i] = 0u;
                    continue;
                }

                const int nextStreak = std::min<int>(
                    static_cast<int>(occZeroStreak[i]) + 1,
                    OCC_ZERO_FRAMES_TO_HIDE);
                occZeroStreak[i] = static_cast<uint8_t>(nextStreak);
                if (nextStreak >= OCC_ZERO_FRAMES_TO_HIDE)
                    occVisible[i] = 0u;
            }
        }
    }

    updateUniformBuffer(currentFrame);
    if (ghostMode) buildGizmoGeometry(currentFrame); // 커맨드 기록 전에 기즈모 버퍼를 먼저 업로드한다.
    recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

    VkSemaphore renderFinished = renderFinishedSems[imageIndex];
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &imageAvailableSems[currentFrame];
    si.pWaitDstStageMask    = &waitStage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &commandBuffers[currentFrame];
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &renderFinished;
    vkQueueSubmit(graphicsQueue, 1, &si, inFlightFences[currentFrame]);

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &renderFinished;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &swapchain;
    pi.pImageIndices      = &imageIndex;
    result = vkQueuePresentKHR(presentQueue, &pi);
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        framebufferResized = false;
        recreateSwapChain();
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

// 스왑체인 재생성

// 디퍼드 렌더링 리소스
void VulkanApp::createDeferredResources() {
    // G-Buffer 샘플러는 스왑체인 재생성 사이에도 재사용한다.
    if (gbufSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (vkCreateSampler(device, &si, nullptr, &gbufSampler) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSampler (gbuf) failed");
    }

    // 현재 스왑체인 크기에 맞는 G-Buffer 포맷을 준비한다.
    VkFormat depthFmt = findDepthFormat();
    VkFormat fmts[GBUFFER_COLOR_ATTACHMENTS] = {
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_FORMAT_R16G16B16A16_SFLOAT,
    };

    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        // G-Buffer 컬러 이미지를 프레임 슬롯별로 생성한다.
        for (int c = 0; c < GBUFFER_COLOR_ATTACHMENTS; ++c) {
            createImage(scExtent.width, scExtent.height, fmts[c],
                        VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        gbufImages[c][f], gbufMemories[c][f]);
            gbufViews[c][f] = createImageView(gbufImages[c][f], fmts[c],
                                              VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // 지오메트리 패스 전용 깊이 이미지를 만든다.
        createImage(scExtent.width, scExtent.height, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    gbufDepthImages[f], gbufDepthMemories[f]);
        gbufDepthViews[f] = createImageView(gbufDepthImages[f], depthFmt,
                                            VK_IMAGE_ASPECT_DEPTH_BIT);

        // 컬러 attachment들과 깊이 1개를 묶어 G-Buffer 프레임버퍼를 만든다.
        VkImageView atts[GBUFFER_COLOR_ATTACHMENTS + 1] = {
            gbufViews[0][f], gbufViews[1][f], gbufViews[2][f], gbufViews[3][f], gbufDepthViews[f]
        };
        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = gbufRenderPass;
        fbCI.attachmentCount = GBUFFER_COLOR_ATTACHMENTS + 1;
        fbCI.pAttachments    = atts;
        fbCI.width           = scExtent.width;
        fbCI.height          = scExtent.height;
        fbCI.layers          = 1;
        if (vkCreateFramebuffer(device, &fbCI, nullptr, &gbufFramebuffers[f]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer (G-Buffer) failed");
    }

    // 디퍼드 조명 패스용 디스크립터 풀과 세트를 만든다.
    if (deferredDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, deferredDescPool, nullptr);
        deferredDescPool = VK_NULL_HANDLE;
    }
    VkDescriptorPoolSize poolSz{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                GBUFFER_COLOR_ATTACHMENTS * MAX_FRAMES_IN_FLIGHT};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.maxSets       = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSz;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &deferredDescPool) != VK_SUCCESS)
        throw std::runtime_error("vkCreateDescriptorPool (deferred) failed");

    VkDescriptorSetLayout layouts[2] = {deferredDescSetLayout, deferredDescSetLayout};
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool     = deferredDescPool;
    allocInfo.descriptorSetCount = (uint32_t)MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts        = layouts;
    if (vkAllocateDescriptorSets(device, &allocInfo, deferredDescSets) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateDescriptorSets (deferred) failed");

    // 각 프레임 슬롯의 G-Buffer 이미지 뷰를 디스크립터에 연결한다.
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        VkDescriptorImageInfo imgInfos[GBUFFER_COLOR_ATTACHMENTS] = {
            {gbufSampler, gbufViews[0][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[1][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[2][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {gbufSampler, gbufViews[3][f], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        };
        VkWriteDescriptorSet writes[GBUFFER_COLOR_ATTACHMENTS] = {};
        for (int i = 0; i < GBUFFER_COLOR_ATTACHMENTS; i++) {
            writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet          = deferredDescSets[f];
            writes[i].dstBinding      = (uint32_t)i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo      = &imgInfos[i];
        }
        vkUpdateDescriptorSets(device, GBUFFER_COLOR_ATTACHMENTS, writes, 0, nullptr);
    }
}

void VulkanApp::destroyDeferredResources() {
    if (deferredDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, deferredDescPool, nullptr);
        deferredDescPool = VK_NULL_HANDLE;
    }
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        if (gbufFramebuffers[f] != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device, gbufFramebuffers[f], nullptr);
            gbufFramebuffers[f] = VK_NULL_HANDLE;
        }
        for (int c = 0; c < GBUFFER_COLOR_ATTACHMENTS; ++c) {
            vkDestroyImageView(device, gbufViews[c][f],    nullptr);
            vkDestroyImage    (device, gbufImages[c][f],   nullptr);
            vkFreeMemory      (device, gbufMemories[c][f], nullptr);
            gbufViews[c][f]    = VK_NULL_HANDLE;
            gbufImages[c][f]   = VK_NULL_HANDLE;
            gbufMemories[c][f] = VK_NULL_HANDLE;
        }
        vkDestroyImageView(device, gbufDepthViews[f],    nullptr);
        vkDestroyImage    (device, gbufDepthImages[f],   nullptr);
        vkFreeMemory      (device, gbufDepthMemories[f], nullptr);
        gbufDepthViews[f]    = VK_NULL_HANDLE;
        gbufDepthImages[f]   = VK_NULL_HANDLE;
        gbufDepthMemories[f] = VK_NULL_HANDLE;
    }
}

void VulkanApp::createShadowResources() {
    VkFormat depthFmt = findDepthFormat();

    // 경계 밖(섀도맵 밖)은 항상 밝게(깊이 1.0) 처리하도록 clamp-to-border 샘플러를 만든다.
    // 비교는 셰이더에서 직접 수행하므로 compareEnable은 끈다.
    {
        VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        si.magFilter    = VK_FILTER_NEAREST;
        si.minFilter    = VK_FILTER_NEAREST;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
        si.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE; // 깊이 1.0 = 그림자 없음
        if (vkCreateSampler(device, &si, nullptr, &shadowSampler) != VK_SUCCESS)
            throw std::runtime_error("vkCreateSampler (shadow) failed");
    }

    // 깊이 전용 렌더패스: 컬러 attachment 없이 깊이만 기록하고 샘플링 가능 레이아웃으로 끝낸다.
    {
        VkAttachmentDescription depth{};
        depth.format         = depthFmt;
        depth.samples        = VK_SAMPLE_COUNT_1_BIT;
        depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE; // 메인 패스에서 샘플링하므로 보존한다.
        depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
        depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depthRef{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
        VkSubpassDescription sp{};
        sp.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp.colorAttachmentCount    = 0;
        sp.pDepthStencilAttachment = &depthRef;

        // 이전 프레임의 섀도맵 샘플링이 끝난 뒤 깊이 쓰기를 시작하고,
        // 깊이 쓰기가 끝난 뒤 메인 패스의 프래그먼트 샘플링이 시작되도록 동기화한다.
        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass    = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass    = 0;
        deps[0].srcStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        deps[0].dstStageMask  = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].srcSubpass    = 0;
        deps[1].dstSubpass    = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask  = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask  = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rpCI{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        rpCI.attachmentCount = 1;
        rpCI.pAttachments    = &depth;
        rpCI.subpassCount    = 1;
        rpCI.pSubpasses      = &sp;
        rpCI.dependencyCount = 2;
        rpCI.pDependencies   = deps;
        if (vkCreateRenderPass(device, &rpCI, nullptr, &shadowRenderPass) != VK_SUCCESS)
            throw std::runtime_error("vkCreateRenderPass (shadow) failed");
    }

    // 프레임 슬롯별 섀도맵 깊이 이미지·뷰·프레임버퍼를 만든다.
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        createImage(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, depthFmt,
                    VK_IMAGE_TILING_OPTIMAL,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    shadowImages[f], shadowImageMemories[f]);
        shadowImageViews[f] = createImageView(shadowImages[f], depthFmt, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkFramebufferCreateInfo fbCI{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        fbCI.renderPass      = shadowRenderPass;
        fbCI.attachmentCount = 1;
        fbCI.pAttachments    = &shadowImageViews[f];
        fbCI.width           = SHADOW_MAP_SIZE;
        fbCI.height          = SHADOW_MAP_SIZE;
        fbCI.layers          = 1;
        if (vkCreateFramebuffer(device, &fbCI, nullptr, &shadowFramebuffers[f]) != VK_SUCCESS)
            throw std::runtime_error("vkCreateFramebuffer (shadow) failed");
    }
}

void VulkanApp::createShadowPipeline() {
    auto vertCode = readFile("shaders/spv/shadow.vert.spv");
    VkShaderModule vertM = createShaderModule(vertCode);

    VkPipelineShaderStageCreateInfo vertStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertM;
    vertStage.pName  = "main";

    // push constant = lightMVP (라이트 뷰·투영 × 모델)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo layoutCI{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutCI.setLayoutCount         = 0;
    layoutCI.pushConstantRangeCount = 1;
    layoutCI.pPushConstantRanges    = &pcRange;
    if (vkCreatePipelineLayout(device, &layoutCI, nullptr, &shadowPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("vkCreatePipelineLayout (shadow) failed");

    auto binding = Vertex::getBindingDesc();
    auto attrs   = Vertex::getAttrDescs();
    VkPipelineVertexInputStateCreateInfo vertInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vertInput.vertexBindingDescriptionCount   = 1;
    vertInput.pVertexBindingDescriptions      = &binding;
    vertInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
    vertInput.pVertexAttributeDescriptions    = attrs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkDynamicState dynStates[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dynStates;

    // 섀도 액네(self-shadowing)를 줄이기 위해 경사 비례 깊이 바이어스를 적용한다.
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode             = VK_POLYGON_MODE_FILL;
    raster.cullMode                = VK_CULL_MODE_NONE; // 양면 모두 깊이를 남겨 누락을 막는다.
    raster.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth               = 1.0f;
    raster.depthBiasEnable         = VK_TRUE;
    raster.depthBiasConstantFactor = 1.25f;
    raster.depthBiasSlopeFactor    = 1.75f;

    VkPipelineMultisampleStateCreateInfo msaa{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable  = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;

    // 컬러 attachment가 없으므로 블렌드 상태도 비운다.
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 0;
    blend.pAttachments    = nullptr;

    // 깊이 전용 패스는 프래그먼트 셰이더가 필요 없다.
    VkGraphicsPipelineCreateInfo pipeCI{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeCI.stageCount          = 1;
    pipeCI.pStages             = &vertStage;
    pipeCI.pVertexInputState   = &vertInput;
    pipeCI.pInputAssemblyState = &inputAssembly;
    pipeCI.pViewportState      = &viewportState;
    pipeCI.pRasterizationState = &raster;
    pipeCI.pMultisampleState   = &msaa;
    pipeCI.pDepthStencilState  = &depthStencil;
    pipeCI.pColorBlendState    = &blend;
    pipeCI.pDynamicState       = &dynState;
    pipeCI.layout              = shadowPipelineLayout;
    pipeCI.renderPass          = shadowRenderPass;
    pipeCI.subpass             = 0;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeCI, nullptr, &shadowPipeline) != VK_SUCCESS)
        throw std::runtime_error("vkCreateGraphicsPipelines (shadow) failed");

    vkDestroyShaderModule(device, vertM, nullptr);
}

void VulkanApp::cleanupSwapChain() {
    destroyDeferredResources();
    vkDestroyImageView(device, depthImageView, nullptr);
    vkDestroyImage(device, depthImage, nullptr);
    vkFreeMemory(device, depthImageMemory, nullptr);
    for (auto fb : framebuffers) vkDestroyFramebuffer(device, fb, nullptr);
    for (auto iv : scImageViews) vkDestroyImageView(device, iv, nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
}

void VulkanApp::recreateSwapChain() {
    int w = 0, h = 0;
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    recreatePresentSemaphores();
    createDepthResources();
    createFramebuffers();
    createDeferredResources();
}

// 리플레이와 정리
// 리소스 정리
// 리플레이 시스템

void VulkanApp::startRecording() {
    if (isReplaying) stopReplay();
    recordedFrames.clear();
    recordStartTime = static_cast<float>(glfwGetTime());
    isRecording     = true;
    std::cout << "[Replay] Recording started\n";
}

void VulkanApp::stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    if (recordedFrames.empty()) { std::cout << "[Replay] Nothing recorded\n"; return; }
    std::string path = nextReplayPath(replayDir);
    saveReplay(path, recordedFrames);
    lastSavedReplay = path;
}

void VulkanApp::startReplay(const std::string& path) {
    if (isRecording) stopRecording();
    std::string target = path;
    if (target.empty()) {
        auto files = findReplayFiles(replayDir);
        if (files.empty()) { std::cerr << "[Replay] No replay files in " << replayDir << "\n"; return; }
        target = files.back();
    }
    replayFrames.clear();
    if (!loadReplay(target, replayFrames) || replayFrames.empty()) return;
    replayFrameIdx  = 0;
    replayStartTime = static_cast<float>(glfwGetTime());
    isReplaying     = true;
    std::cout << "[Replay] Playback started: " << target << "\n";
}

void VulkanApp::stopReplay() {
    isReplaying = false;
    std::cout << "[Replay] Playback stopped\n";
}

void VulkanApp::cleanup() {
    perfStats.cleanup();
    cleanupImGui();
    cleanupSwapChain();

    // 기즈모 버텍스 버퍼
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (i < (int)gizmoVBs.size()) {
            vkDestroyBuffer(device, gizmoVBs[i], nullptr);
            vkFreeMemory(device, gizmoVBMemories[i], nullptr);
        }
    }
    vkDestroyPipeline(device, gizmoPipeline, nullptr);

    // 오클루전 쿼리 풀
    if (occlusionQueryPool != VK_NULL_HANDLE)
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);

    // 인스턴스 SSBO와 디스크립터 풀
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (i < (int)instanceStagingBuffers.size()) {
            if (i < (int)instanceSSBOMapped.size() && instanceSSBOMapped[i])
                vkUnmapMemory(device, instanceStagingMemories[i]);
            vkDestroyBuffer(device, instanceStagingBuffers[i], nullptr);
            vkFreeMemory(device, instanceStagingMemories[i], nullptr);
        }
        if (i < (int)instanceSSBOs.size()) {
            vkDestroyBuffer(device, instanceSSBOs[i], nullptr);
            vkFreeMemory(device, instanceSSBOMemories[i], nullptr);
        }
    }
    if (instanceDescPool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, instanceDescPool, nullptr);

    // 프레임별 UBO
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBufferMemories[i], nullptr);
        vkDestroyBuffer(device, cullUniformBuffers[i], nullptr);
        vkFreeMemory(device, cullUniformBufferMemories[i], nullptr);
        if (i < (int)sceneLightUBOs.size()) {
            vkDestroyBuffer(device, sceneLightUBOs[i], nullptr);
            vkFreeMemory(device, sceneLightUBOMemories[i], nullptr);
        }
    }
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, instanceDescSetLayout, nullptr);

    // 씬 텍스처(씬 재로드마다 갱신)
    cleanupTextureResources();
    // 기본 1×1 흰색 텍스처와 샘플러(앱 생명주기 동안 유지)
    vkDestroyImageView(device, defaultTexView, nullptr);
    vkDestroyImage(device, defaultTexImage, nullptr);
    vkFreeMemory(device, defaultTexMemory, nullptr);
    vkDestroySampler(device, texSampler, nullptr);

    vkDestroyBuffer(device, indexBuffer, nullptr);
    vkFreeMemory(device, indexBufferMemory, nullptr);
    vkDestroyBuffer(device, vertexBuffer, nullptr);
    vkFreeMemory(device, vertexBufferMemory, nullptr);

    for (VkSemaphore sem : imageAvailableSems)
        vkDestroySemaphore(device, sem, nullptr);
    for (VkSemaphore sem : renderFinishedSems)
        vkDestroySemaphore(device, sem, nullptr);
    for (VkFence fence : inFlightFences)
        vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, commandPool, nullptr);

    // 렌더링 파이프라인
    vkDestroyPipeline(device, graphicsPipelineFlippedCull, nullptr);
    vkDestroyPipeline(device, graphicsPipeline,           nullptr);
    vkDestroyPipeline(device, graphicsPipelineNoCull,     nullptr);
    vkDestroyPipeline(device, graphicsPipelineAlpha,      nullptr);
    vkDestroyPipeline(device, graphicsPipelineQueryOnly, nullptr);
    vkDestroyPipeline(device, graphicsPipelineInst,       nullptr);
    vkDestroyPipeline(device, graphicsPipelineInstNoCull, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout,         nullptr);
    vkDestroyPipelineLayout(device, instancePipelineLayout, nullptr);
    // 디퍼드 렌더링 영구 리소스
    vkDestroyPipeline(device, gbufPipeline,          nullptr);
    vkDestroyPipeline(device, gbufPipelineNoCull,    nullptr);
    vkDestroyPipeline(device, deferredLightPipeline, nullptr);
    vkDestroyPipelineLayout(device, deferredLightLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, deferredDescSetLayout, nullptr);
    if (gbufRenderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, gbufRenderPass, nullptr);
    if (gbufSampler    != VK_NULL_HANDLE)
        vkDestroySampler(device, gbufSampler, nullptr);

    // 방향광 섀도맵 리소스
    vkDestroyPipeline(device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(device, shadowPipelineLayout, nullptr);
    for (int f = 0; f < MAX_FRAMES_IN_FLIGHT; ++f) {
        if (shadowFramebuffers[f] != VK_NULL_HANDLE)
            vkDestroyFramebuffer(device, shadowFramebuffers[f], nullptr);
        if (shadowImageViews[f] != VK_NULL_HANDLE)
            vkDestroyImageView(device, shadowImageViews[f], nullptr);
        if (shadowImages[f] != VK_NULL_HANDLE)
            vkDestroyImage(device, shadowImages[f], nullptr);
        if (shadowImageMemories[f] != VK_NULL_HANDLE)
            vkFreeMemory(device, shadowImageMemories[f], nullptr);
    }
    if (shadowRenderPass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, shadowRenderPass, nullptr);
    if (shadowSampler != VK_NULL_HANDLE)
        vkDestroySampler(device, shadowSampler, nullptr);

    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyDevice(device, nullptr);
    if (kEnableValidation) DestroyDebugMessenger(instance, debugMessenger);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
    glfwDestroyWindow(window);
    glfwTerminate();
}

// ImGui 초기화와 정리
void VulkanApp::initImGui() {
    // ImGui가 사용할 descriptor pool을 만든다.
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
    VkDescriptorPoolCreateInfo poolCI{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolCI.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCI.maxSets       = 1;
    poolCI.poolSizeCount = 1;
    poolCI.pPoolSizes    = &poolSize;
    if (vkCreateDescriptorPool(device, &poolCI, nullptr, &imguiDescPool) != VK_SUCCESS)
        throw std::runtime_error("ImGui descriptor pool creation failed");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // imgui.ini 저장을 비활성화한다.

    // 약간 투명한 다크 스타일을 적용한다.
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding    = 6.0f;
    style.FrameRounding     = 4.0f;
    style.WindowBorderSize  = 0.0f;
    style.Alpha             = 0.92f;

    ImGui_ImplGlfw_InitForVulkan(window, true);

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion     = VK_API_VERSION_1_2;
    initInfo.Instance       = instance;
    initInfo.PhysicalDevice = physDevice;
    initInfo.Device         = device;
    initInfo.QueueFamily    = findQueueFamilies(physDevice).graphics.value();
    initInfo.Queue          = graphicsQueue;
    initInfo.DescriptorPool = imguiDescPool;
    initInfo.MinImageCount  = 2;
    initInfo.ImageCount     = static_cast<uint32_t>(scImages.size());
    // ImGui 1.92 이상 규칙에 맞춰 렌더 패스와 MSAA 값을 지정한다.
    initInfo.PipelineInfoMain.RenderPass   = renderPass;
    initInfo.PipelineInfoMain.MSAASamples  = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&initInfo);
}

void VulkanApp::cleanupImGui() {
    vkDeviceWaitIdle(device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    if (imguiDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, imguiDescPool, nullptr);
        imguiDescPool = VK_NULL_HANDLE;
    }
}

// 성능 오버레이
// 벤치마크 CSV 로거
void VulkanApp::startBenchmark() {
    benchmarkSamples.clear();
    benchmarkElapsed = 0.f;
    benchmarkActive  = true;
    printf("[Benchmark] Started (%.0f s)\n", benchmarkDuration);
}

void VulkanApp::finishBenchmark() {
    benchmarkActive = false;
    if (benchmarkSamples.empty()) return;

    int   n      = (int)benchmarkSamples.size();
    float sumFps = 0, sumFt = 0, sumCpu = 0, sumGpu = 0;
    float minFt  = 1e9f, maxFt = 0.f;
    int   sumDc  = 0,    sumCulled = 0;
    for (auto& s : benchmarkSamples) {
        sumFps    += s.fps;
        sumFt     += s.frameTimeMs;
        sumCpu    += s.cpuPercent;
        sumGpu    += s.gpuPercent;
        sumDc     += s.drawCalls;
        sumCulled += s.culled;
        minFt = std::min(minFt, s.frameTimeMs);
        maxFt = std::max(maxFt, s.frameTimeMs);
    }
    float avgFt  = sumFt / n;
    float avgFps = (avgFt > 0.0f) ? 1000.0f / avgFt : 0.0f;

    std::filesystem::create_directories("results");

    auto       tp = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    char ts[20]   = {};
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

    // 최적화 플래그 문자열 (9비트)
    char flags[12] = {};
    std::snprintf(flags, sizeof(flags), "%d%d%d%d%d%d%d%d%d",
        (int)optFlags.frustumCulling,  (int)optFlags.lod,
        (int)optFlags.instancing,      (int)optFlags.backfaceCulling,
        (int)optFlags.depthSort,       (int)optFlags.occlusionCulling,
        (int)optFlags.viewDistCulling, (int)optFlags.smallCulling,
        (int)optFlags.deferredShading);

    std::string path = std::string("results/bench_") + flags
                     + "_stress" + std::to_string(stressLevel)
                     + "_" + ts + ".csv";

    std::ofstream csv(path);
    csv << "frame,fps,frametime_ms,cpu_pct,gpu_pct,draw_calls,culled\n";
    for (int i = 0; i < n; ++i) {
        auto& s = benchmarkSamples[i];
        csv << i << "," << s.fps << "," << s.frameTimeMs << ","
            << s.cpuPercent << "," << s.gpuPercent << ","
            << s.drawCalls  << "," << s.culled << "\n";
    }
    csv << "\n# summary: avg_fps,avg_ft_ms,min_ft_ms,max_ft_ms,avg_cpu_pct,avg_gpu_pct,avg_dc,avg_culled\n";
    csv << "summary,"
        << avgFps      << "," << avgFt        << ","
        << minFt        << "," << maxFt        << ","
        << (sumCpu / n) << "," << (sumGpu / n) << ","
        << (sumDc  / n) << "," << (sumCulled / n) << "\n";
    csv << "# flags(FC,LOD,Inst,BFC,DS,OC,VDC,SC,Def): " << flags << "\n";
    csv << "# stress_level: " << stressLevel
        << "  objects: " << (int)drawObjects.size() << "\n";

    printf("[Benchmark] Saved: %s  (avg FPS %.1f  avg ft %.2f ms)\n",
           path.c_str(), avgFps, avgFt);
}

// 스트레스 씬 복제
void VulkanApp::applyStress() {
    if (occlusionQueryPool != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
        vkDestroyQueryPool(device, occlusionQueryPool, nullptr);
        occlusionQueryPool = VK_NULL_HANDLE;
        occQueryCount      = 0;
    }

    if (stressLevel == 0) {
        drawObjects = baseDrawObjects;
    } else {
        const int   extra   = (1 << stressLevel) - 1; // 2x=1, 4x=3, 8x=7, 16x=15
        const float SPACING = 30.0f;
        drawObjects = baseDrawObjects;
        int row = 1, col = 0;
        for (int c = 0; c < extra; ++c) {
            glm::vec3 offset((float)col * SPACING, 0.f, (float)row * SPACING);
            for (const auto& obj : baseDrawObjects) {
                DrawObject copy      = obj;
                copy.push.model      = glm::translate(glm::mat4(1.f), offset) * obj.push.model;
                copy.boundCenter     = obj.boundCenter + offset;
                copy.skipOcclusion   = true;
                drawObjects.push_back(copy);
            }
            if (++col >= 4) { col = 0; ++row; }
        }
    }

    rebuildInstancingGroups();
    resetOcclusionState();
    if (!drawObjects.empty()) createOcclusionQueryPool();

    printf("[Stress] Level %d -> %dx  (%d objects)\n",
           stressLevel, 1 << stressLevel, (int)drawObjects.size());
}

// 리플레이 기반 자동 벤치마크
void VulkanApp::startAutoBenchmark() {
    // 리플레이 파일 필요
    auto files = findReplayFiles(replayDir);
    if (files.empty()) {
        printf("[AutoBench] No replay found. Record a replay with [R] first.\n");
        return;
    }
    autoBenchReplayPath = files.back(); // 가장 최근 리플레이

    // 현재 상태 저장 & 불필요한 기능 비활성화 (순수 렌더 성능만 측정)
    autoBenchSavedFlags = optFlags;
    autoBenchSavedGhost = ghostMode;
    ghostMode = false; // ghost 모드 OFF (기즈모·오버레이 컬링 시각화 제거)

    // 10 실험 정의
    // 모든 실험은 baseline(all OFF) 기준으로 하나씩 최적화 기법을 추가
    OptFlags off{};
    off.frustumCulling = off.lod = off.instancing = off.backfaceCulling =
    off.depthSort = off.occlusionCulling = off.viewDistCulling =
    off.smallCulling = off.deferredShading = false;

    autoBenchExps.clear();
    auto addExp = [&](const std::string& name, OptFlags f) {
        AutoBenchExp e; e.name = name; e.flags = f; autoBenchExps.push_back(std::move(e));
    };

    OptFlags f;
    addExp("0.Baseline (all OFF)", off);

    f = off; f.frustumCulling   = true;  addExp("1.Frustum Culling",   f);
    f = off; f.lod               = true;  addExp("2.LOD",               f);
    f = off; f.instancing        = true;  addExp("3.GPU Instancing",    f);
    f = off; f.backfaceCulling   = true;  addExp("4.Backface Culling",  f);
    f = off; f.depthSort         = true;  addExp("5.Depth Sort",        f);
    f = off; f.occlusionCulling  = true;  addExp("6.Occlusion Culling", f);
    f = off; f.viewDistCulling   = true;  addExp("7.View Dist Cull",    f);
    f = off; f.smallCulling      = true;  addExp("8.Small Obj Cull",    f);
    f = off; f.deferredShading   = true;  addExp("9.Deferred Shading",  f);

    autoBenchExpIdx = 0;
    autoBenchRunIdx = 0;
    autoBenchActive = true;
    printf("[AutoBench] Starting: %d experiments x %d runs each  replay=%s\n",
           AUTO_BENCH_TOTAL, AUTO_BENCH_RUNS, autoBenchReplayPath.c_str());
    startAutoBenchRun();
}

void VulkanApp::startAutoBenchRun() {
    auto& exp = autoBenchExps[autoBenchExpIdx];
    optFlags  = exp.flags;
    exp.current.clear();
    autoBenchSkipFrames = 3;
    startReplay(autoBenchReplayPath);
    printf("[AutoBench] Exp %d/%d '%s'  run %d/%d\n",
           autoBenchExpIdx + 1, AUTO_BENCH_TOTAL,
           exp.name.c_str(),
           autoBenchRunIdx + 1, AUTO_BENCH_RUNS);
}

void VulkanApp::onAutoBenchRunEnd() {
    auto& exp = autoBenchExps[autoBenchExpIdx];
    if (!exp.current.empty()) {
        int   n      = (int)exp.current.size();
        float sumFps = 0, sumFt = 0, sumCpu = 0, sumGpu = 0;
        float minFt  = 1e9f, maxFt = 0.f;
        int   sumDc  = 0, sumCulled = 0;
        for (auto& s : exp.current) {
            sumFps    += s.fps;
            sumFt     += s.frameTimeMs;
            sumCpu    += s.cpuPercent;
            sumGpu    += s.gpuPercent;
            sumDc     += s.drawCalls;
            sumCulled += s.culled;
            minFt = std::min(minFt, s.frameTimeMs);
            maxFt = std::max(maxFt, s.frameTimeMs);
        }
        float avgFt  = sumFt / n;
        float avgFps = (avgFt > 0.0f) ? 1000.0f / avgFt : 0.0f;
        AutoBenchRunResult r{};
        r.avgFps    = avgFps;
        r.avgFtMs   = avgFt;
        r.minFtMs   = minFt;
        r.maxFtMs   = maxFt;
        r.avgCpu    = sumCpu / n;
        r.avgGpu    = sumGpu / n;
        r.avgDc     = sumDc  / n;
        r.avgCulled = sumCulled / n;
        exp.runs.push_back(r);
        exp.current.clear();
    }

    ++autoBenchRunIdx;
    if (autoBenchRunIdx < AUTO_BENCH_RUNS) {
        startAutoBenchRun();
        return;
    }
    // 다음 실험
    autoBenchRunIdx = 0;
    ++autoBenchExpIdx;
    if (autoBenchExpIdx < AUTO_BENCH_TOTAL) {
        startAutoBenchRun();
        return;
    }
    finishAutoBenchmark();
}

void VulkanApp::finishAutoBenchmark() {
    autoBenchActive = false;
    optFlags        = autoBenchSavedFlags;
    ghostMode       = autoBenchSavedGhost;

    std::filesystem::create_directories("results");

    auto       tp = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(tp);
    char ts[20]   = {};
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));

    std::string path = std::string("results/autobench_") + ts + ".csv";
    std::ofstream csv(path);

    // 헤더
    csv << "experiment,run,avg_fps,avg_ft_ms,min_ft_ms,max_ft_ms,"
           "avg_cpu_pct,avg_gpu_pct,avg_dc,avg_culled\n";

    for (auto& exp : autoBenchExps) {
        for (int r = 0; r < (int)exp.runs.size(); ++r) {
            auto& res = exp.runs[r];
            csv << exp.name << "," << (r + 1) << ","
                << res.avgFps    << "," << res.avgFtMs   << ","
                << res.minFtMs   << "," << res.maxFtMs   << ","
                << res.avgCpu    << "," << res.avgGpu    << ","
                << res.avgDc     << "," << res.avgCulled << "\n";
        }
    }

    // 실험별 평균 요약 + baseline 대비 개선률
    csv << "\n# --- 실험별 평균 요약 (5회 평균) ---\n";
    csv << "experiment,avg_fps,avg_ft_ms,fps_vs_baseline_pct\n";

    float baseAvgFps = 0.f, baseAvgFt = 0.f;
    if (!autoBenchExps.empty() && !autoBenchExps[0].runs.empty()) {
        for (auto& r : autoBenchExps[0].runs) { baseAvgFps += r.avgFps; baseAvgFt += r.avgFtMs; }
        baseAvgFps /= (float)autoBenchExps[0].runs.size();
        baseAvgFt  /= (float)autoBenchExps[0].runs.size();
    }

    for (auto& exp : autoBenchExps) {
        if (exp.runs.empty()) continue;
        float fps = 0.f, ft = 0.f;
        for (auto& r : exp.runs) { fps += r.avgFps; ft += r.avgFtMs; }
        fps /= (float)exp.runs.size();
        ft  /= (float)exp.runs.size();
        float improvement = (baseAvgFps > 0.f) ? (fps - baseAvgFps) / baseAvgFps * 100.f : 0.f;
        csv << exp.name << "," << fps << "," << ft << "," << improvement << "\n";
    }

    csv << "# replay: " << autoBenchReplayPath << "\n";
    csv << "# stress_level: " << stressLevel
        << "  objects: " << (int)drawObjects.size() << "\n";

    printf("[AutoBench] Done! Saved: %s\n", path.c_str());
    printf("[AutoBench] Baseline avg FPS: %.1f  ft: %.2f ms\n", baseAvgFps, baseAvgFt);
}

void VulkanApp::drawStatsOverlay() {
    const float PAD = 10.0f;
    ImGuiIO& io = ImGui::GetIO();

    // 오버레이 창을 좌상단에 고정한다.
    ImGui::SetNextWindowPos({PAD, PAD}, ImGuiCond_Always);
    ImGui::SetNextWindowSize({200, 0}, ImGuiCond_Always); // 내용에 맞춰 높이를 자동으로 둔다.
    ImGui::SetNextWindowBgAlpha(0.78f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoDecoration |
        ImGuiWindowFlags_NoMove       |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoFocusOnAppearing |
        ImGuiWindowFlags_NoNav;
        // NoInputs 제거: 조명 체크박스 클릭 허용

    ImGui::SetNextWindowSize({220, 0}, ImGuiCond_Always);

    if (ImGui::Begin("##stats", nullptr, flags)) {
        // 자동 벤치마크 중: 최소 오버레이 (진행 상황만 표시)
        if (autoBenchActive) {
            ImGui::TextColored({1.0f, 0.9f, 0.2f, 1.0f}, "AUTO-BENCHMARK");
            ImGui::Separator();
            int totalRuns = AUTO_BENCH_TOTAL * AUTO_BENCH_RUNS;
            int doneRuns  = autoBenchExpIdx * AUTO_BENCH_RUNS + autoBenchRunIdx;
            char pbuf[48];
            std::snprintf(pbuf, sizeof(pbuf), "%d / %d", doneRuns, totalRuns);
            ImGui::ProgressBar((float)doneRuns / totalRuns, ImVec2(-1, 0), pbuf);
            if (autoBenchExpIdx < (int)autoBenchExps.size()) {
                ImGui::Text("Exp %d: %s", autoBenchExpIdx + 1,
                            autoBenchExps[autoBenchExpIdx].name.c_str());
                ImGui::Text("Run %d/%d", autoBenchRunIdx + 1, AUTO_BENCH_RUNS);
            }
            ImGui::TextColored({0.6f,0.6f,0.6f,1.f}, "[M] Abort");
            ImGui::TextColored(
                (perfStats.fps >= 60.f) ? ImVec4{0.4f,1.0f,0.4f,1.0f}
                                        : ImVec4{1.0f,0.3f,0.3f,1.0f},
                "FPS %.1f", perfStats.fps);
            ImGui::End();
            return; // 나머지 UI 전부 스킵 -> 오버헤드 최소화
        }
        // 성능 지표
        ImGui::TextColored({0.85f, 0.85f, 1.0f, 1.0f}, "Performance");
        ImGui::Separator();

        ImVec4 fpsColor = (perfStats.fps >= 60.f) ? ImVec4{0.4f,1.0f,0.4f,1.0f}
                        : (perfStats.fps >= 30.f) ? ImVec4{1.0f,0.9f,0.2f,1.0f}
                                                  : ImVec4{1.0f,0.3f,0.3f,1.0f};
        ImGui::TextColored(fpsColor, "FPS       %6.1f", perfStats.fps);
        ImGui::Text(        "Frame     %5.2f ms",       perfStats.frameTimeMs);

        // 프레임 시간 히스토리 그래프 (고정 링버퍼 -> 선형 배열로 언롤)
        if (frameTimeHistCount > 0) {
            float hist[FRAME_HISTORY_SIZE];
            int n = frameTimeHistCount;
            int oldest = (frameTimeHistIdx - n + FRAME_HISTORY_SIZE) % FRAME_HISTORY_SIZE;
            for (int i = 0; i < n; ++i)
                hist[i] = frameTimeHistBuf[(oldest + i) % FRAME_HISTORY_SIZE];
            float ftMin = hist[0], ftMax = hist[0], ftSum = 0.f;
            for (int i = 0; i < n; ++i) {
                ftMin = std::min(ftMin, hist[i]);
                ftMax = std::max(ftMax, hist[i]);
                ftSum += hist[i];
            }
            float ftAvg = ftSum / (float)n;
            char overlay[32];
            std::snprintf(overlay, sizeof(overlay), "avg %.1f ms", ftAvg);
            ImGui::PlotLines("##ft", hist, n, 0,
                             overlay, 0.f, std::max(ftMax * 1.2f, 33.f), {-1, 40});
            ImGui::Text("min %5.2f  max %5.2f ms", ftMin, ftMax);
        }

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();

        ImVec4 cpuColor = (perfStats.cpuPercent < 50.f) ? ImVec4{0.4f,1.0f,0.4f,1.0f}
                        : (perfStats.cpuPercent < 80.f) ? ImVec4{1.0f,0.9f,0.2f,1.0f}
                                                        : ImVec4{1.0f,0.3f,0.3f,1.0f};
        ImGui::TextColored(cpuColor, "CPU sys   %5.1f %%", perfStats.cpuPercent);
        ImGui::Text(        "CPU proc  %5.1f %%",         perfStats.processCpuPercent);
        ImGui::Text(        "RAM     %7.1f MB",           perfStats.ramMB);
        if (perfStats.gpuPercent >= 0.f) {
            ImVec4 gpuColor = (perfStats.gpuPercent < 50.f) ? ImVec4{0.4f,1.0f,0.4f,1.0f}
                            : (perfStats.gpuPercent < 80.f) ? ImVec4{1.0f,0.9f,0.2f,1.0f}
                                                            : ImVec4{1.0f,0.3f,0.3f,1.0f};
            ImGui::TextColored(gpuColor, "GPU       %5.1f %%", perfStats.gpuPercent);
        } else {
            ImGui::TextDisabled("GPU          N/A");
        }

        // 드로우콜 통계
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::Text("Draw calls  %3d / %3d",
                    renderedCount, (int)drawObjects.size());
        ImGui::Text("Culled      %3d", culledCount);
        if (optFlags.instancing && instancedGroupCount > 0)
            ImGui::TextColored({0.4f,1.0f,0.7f,1.0f},
                               "Inst groups %3d", instancedGroupCount);

        // 씬 로드 시간 표시
        if (lastLoadTiming.valid) {
            ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
            ImGui::TextColored({0.85f,0.85f,1.0f,1.0f}, "Last scene load");
            ImGui::Separator();
            ImGui::Text("Total     %6.1f ms", lastLoadTiming.totalMs);
            ImGui::Text("Parse     %6.1f ms", lastLoadTiming.sceneParseMs);
            ImGui::Text("Upload    %6.1f ms", lastLoadTiming.uploadMs);
        }


        // 최적화 토글 표시
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextColored({0.85f, 0.85f, 1.0f, 1.0f}, "Optimizations");
        ImGui::Separator();

        auto row = [](const char* key, const char* label, bool on) {
            ImVec4 col = on ? ImVec4{0.4f,1.0f,0.4f,1.0f} : ImVec4{0.6f,0.6f,0.6f,1.0f};
            ImGui::TextColored({0.9f,0.8f,0.4f,1.0f}, "[%s]", key);
            ImGui::SameLine();
            ImGui::TextColored(col, "%-22s %s", label, on ? "ON " : "OFF");
        };

        row("1", "Frustum Culling",   optFlags.frustumCulling);
        row("2", "LOD",               optFlags.lod);
        row("3", "GPU Instancing",    optFlags.instancing);
        row("4", "Backface Culling",  optFlags.backfaceCulling);
        row("5", "Front-Back Sort",   optFlags.depthSort);
        row("6", "Occlusion Culling", optFlags.occlusionCulling);
        row("7", "View Dist Cull",    optFlags.viewDistCulling);
        row("8", "Small Obj Cull",    optFlags.smallCulling);
        row("9", "Deferred Shading",  optFlags.deferredShading);
        row("0", "All Optimizations", optFlags.frustumCulling && optFlags.lod &&
            optFlags.instancing && optFlags.backfaceCulling && optFlags.depthSort &&
            optFlags.occlusionCulling && optFlags.viewDistCulling &&
            optFlags.smallCulling && optFlags.deferredShading);
        row("B", "Dark Floor",        darkFloor);
        row("F", "Far Plane 5000",    extendedFarPlane);

        // 조명 토글 표시
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextColored({0.85f, 0.85f, 1.0f, 1.0f}, "Lighting");
        ImGui::Separator();
        row("L", "Scene Lights",      sceneLightsOn);
        row("N", "Ambient",           ambientOn);
        row("V", "Emissive",          emissiveOn);
        if (!sceneLights.empty()) {
            ImGui::Spacing();
            ImGui::TextDisabled("GLTF Lights (%d)", (int)sceneLights.size());
            for (int li = 0; li < (int)sceneLights.size() && li < MAX_SCENE_LIGHTS; ++li) {
                ImGui::TextColored(
                    sceneLights[li].enabled ? ImVec4{0.4f,1.f,0.4f,1.f} : ImVec4{0.5f,0.5f,0.5f,1.f},
                    "  %s %s", sceneLights[li].enabled ? "ON " : "OFF",
                    sceneLights[li].name.c_str());
            }
        }

        // 벤치마크 표시
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextColored({0.85f, 0.85f, 1.0f, 1.0f}, "Benchmark");
        ImGui::Separator();

        if (autoBenchActive) {
            // 자동 벤치마크 진행 상황
            int totalRuns  = AUTO_BENCH_TOTAL * AUTO_BENCH_RUNS;
            int doneRuns   = autoBenchExpIdx * AUTO_BENCH_RUNS + autoBenchRunIdx;
            float progAll  = (totalRuns > 0) ? (float)doneRuns / totalRuns : 0.f;

            char pbuf[48];
            std::snprintf(pbuf, sizeof(pbuf), "%d / %d runs", doneRuns, totalRuns);
            ImGui::ProgressBar(progAll, ImVec2(-1, 0), pbuf);

            if (autoBenchExpIdx < (int)autoBenchExps.size()) {
                auto& exp = autoBenchExps[autoBenchExpIdx];
                ImGui::TextColored({1.0f,0.9f,0.2f,1.0f},
                    "Exp %d/%d  Run %d/%d",
                    autoBenchExpIdx + 1, AUTO_BENCH_TOTAL,
                    autoBenchRunIdx + 1, AUTO_BENCH_RUNS);
                ImGui::TextColored({0.7f,0.9f,1.0f,1.0f}, "%s", exp.name.c_str());
                ImGui::Text("Samples: %d", (int)exp.current.size());
            }
            ImGui::TextDisabled("[M] Abort");
        } else if (benchmarkActive) {
            float prog = glm::clamp(benchmarkElapsed / benchmarkDuration, 0.f, 1.f);
            char  pbuf[32];
            std::snprintf(pbuf, sizeof(pbuf), "%.1f / %.0f s", benchmarkElapsed, benchmarkDuration);
            ImGui::ProgressBar(prog, ImVec2(-1, 0), pbuf);
            ImGui::TextColored({1.0f,0.9f,0.2f,1.0f}, "%d samples", (int)benchmarkSamples.size());
            ImGui::TextDisabled("[M] Stop & save CSV");
        } else {
            bool hasReplay = !findReplayFiles(replayDir).empty();
            if (hasReplay) {
                ImGui::TextColored({0.4f,1.0f,0.7f,1.0f},
                    "[M] Auto-bench");
                ImGui::TextDisabled("%d exps x %d runs",
                    AUTO_BENCH_TOTAL, AUTO_BENCH_RUNS);
                ImGui::TextDisabled("Saves autobench_*.csv");
            } else {
                ImGui::TextDisabled("[M] Start %.0fs benchmark", benchmarkDuration);
                ImGui::TextDisabled("(Record replay first for");
                ImGui::TextDisabled(" full auto-benchmark)");
            }
        }

        // 스트레스 배율 표시
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        ImGui::TextColored({0.85f, 0.85f, 1.0f, 1.0f}, "Stress Test");
        ImGui::Separator();
        {
            int mult = 1 << stressLevel;
            ImVec4 stressCol = (stressLevel == 0) ? ImVec4{0.6f,0.6f,0.6f,1.f}
                             : (stressLevel <= 2)  ? ImVec4{1.0f,0.9f,0.2f,1.f}
                                                   : ImVec4{1.0f,0.3f,0.3f,1.f};
            ImGui::TextColored(stressCol, "[T] %dx  (%d objs)",
                               mult, (int)drawObjects.size());
        }

        // Ghost 모드 상태 표시
        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        if (ghostMode) {
            ImGui::TextColored({0.4f,1.0f,1.0f,1.0f}, "[G] Ghost mode  ON");
            ImGui::TextDisabled("Arrow: move cam");
            ImGui::TextDisabled("MidClick+Drag: rotate cam");
        } else {
            ImGui::TextColored({0.6f,0.6f,0.6f,1.0f}, "[G] Ghost mode  OFF");
        }

        // 시네마틱 카메라 모드 표시
        if (camera.cinematic)
            ImGui::TextColored({1.0f,0.85f,0.4f,1.0f}, "[C] Cinematic   ON");
        else
            ImGui::TextColored({0.6f,0.6f,0.6f,1.0f}, "[C] Cinematic   OFF");

        // 리플레이 표시
        ImGui::Separator();
        ImGui::TextColored({0.85f,0.85f,1.0f,1.0f}, "Replay");

        if (isRecording) {
            float elapsed = static_cast<float>(glfwGetTime()) - recordStartTime;
            // 약 0.5초마다 깜빡이는 녹화 표시를 만든다.
            bool blink = (static_cast<int>(elapsed * 2.f) % 2) == 0;
            if (blink)
                ImGui::TextColored({1.0f,0.2f,0.2f,1.0f}, "● REC  %.1fs  %d frames",
                                   elapsed, (int)recordedFrames.size());
            else
                ImGui::TextColored({0.6f,0.2f,0.2f,1.0f}, "  REC  %.1fs  %d frames",
                                   elapsed, (int)recordedFrames.size());
            ImGui::TextDisabled("[R] Stop & save");
        } else if (isReplaying && !replayFrames.empty()) {
            float elapsed = static_cast<float>(glfwGetTime()) - replayStartTime;
            float total   = replayFrames.back().time;
            float prog    = (total > 0.f) ? glm::clamp(elapsed / total, 0.f, 1.f) : 1.f;
            ImGui::TextColored({0.4f,0.7f,1.0f,1.0f}, "▶ REPLAY  %.1f / %.1fs",
                               elapsed, total);
            ImGui::ProgressBar(prog, ImVec2(-1, 6));
            ImGui::TextDisabled("[P] Stop  [G] Ghost overlay");
            ImGui::TextDisabled("1-6: toggle opts while playing");
        } else {
            ImGui::TextDisabled("[R] Start recording");
            if (!findReplayFiles(replayDir).empty())
                ImGui::TextDisabled("[P] Play latest replay");
            else
                ImGui::TextDisabled("[P] No replay saved yet");
        }
    }
    ImGui::End();
}
