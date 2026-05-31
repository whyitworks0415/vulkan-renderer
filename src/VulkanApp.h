#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <vector>
#include <optional>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "Camera.h"
#include "GLTFLoader.h"
#include "MeshLoader.h"
#include "PerformanceStats.h"
#include "SceneLoader.h"
#include "Replay.h"

// CPU와 GPU가 공유하는 데이터 타입

// 버텍스 하나의 데이터 (셰이더 입력과 1:1 대응)
struct Vertex {
    glm::vec3 pos; // 월드 공간 위치
    glm::vec3 normal; // 면법선 (flat shading)
    glm::vec3 color; // 재질 색상 (RGB 0~1)
    glm::vec2 uv; // 텍스처 UV 좌표 (텍스처 없으면 (0,0))

    // Vulkan 파이프라인에 이 구조체를 버텍스 버퍼로 바인딩하기 위한 설명자
    static VkVertexInputBindingDescription   getBindingDesc();
    static std::array<VkVertexInputAttributeDescription, 4> getAttrDescs();
};

// 카메라/조명 정보 – 매 프레임 UBO(Uniform Buffer Object) 로 GPU 에 전달
struct CameraUBO {
    alignas(16) glm::mat4 view; // 뷰 행렬 (월드 -> 카메라 공간)
    alignas(16) glm::mat4 proj; // 투영 행렬 (카메라 -> 클립 공간)
    alignas(16) glm::vec4 cameraPos; // 카메라 월드 위치 (.w = darkFloor 플래그)
    alignas(16) glm::vec4 lightDir; // 방향광 방향 (씬 쪽을 향하는 단위 벡터)
    alignas(16) glm::mat4 lightVP; // 방향광 시점 뷰·투영 (섀도맵 샘플링용)
};

// 오브젝트별 상수 – vkCmdPushConstants 로 draw call 마다 갱신 (빠른 전달)
// 총 112 bytes (64+16+16+16) — Vulkan 최소 보장 128 bytes 이내
struct PushConstants {
    glm::mat4 model; // 모델 행렬 (로컬 -> 월드 공간)            [offset   0, 64 B]
    glm::vec4 baseColor; // 기본 색상 (RGBA)                        [offset  64, 16 B]
    float     shininess; // 퐁 반사 집중도 (클수록 하이라이트 좁음)   [offset  80,  4 B]
    float     specularStrength; // 스페큘러 강도 (0 = 무광)                 [offset  84,  4 B]
    float     reflectStrength; // 반사 강도 (Fresnel 스케일)               [offset  88,  4 B]
    float     textureIndex; // 텍스처 인덱스 (-1.0 = 없음)              [offset  92,  4 B]
    glm::vec4 emissive; // rgb=발광 색상, a=발광 강도(디퍼드 경로에서 G-Buffer에 패킹)
};

// SceneLightUBO: GPU 씬 조명 배열(set 0, binding 2)
// GLTF KHR_lights_punctual 에서 추출한 최대 8개 동적 조명
// useSceneLights == 0 이면 셰이더 내부의 기본 조명을 사용
struct GpuSceneLight {
    // Point/Spot: xyz=위치, w=범위(0=무제한) / Directional: xyz=방향, w=0
    alignas(16) glm::vec4 posRange;
    // xyz=스팟 방향, w=타입(0=Point, 1=Directional, 2=Spot)
    alignas(16) glm::vec4 dirType;
    // xyz=색상*강도, w=활성 여부(0 또는 1)
    alignas(16) glm::vec4 colorEnab;
};
static constexpr int MAX_SCENE_LIGHTS = 8;
struct SceneLightUBO {
    int32_t numLights; // 실제 조명 수
    int32_t useSceneLights; // 1=GLTF 조명, 0=셰이더 기본 조명
    int32_t ambientOn; // 환경광 ON/OFF
    int32_t emissiveOn; // 발광 재질 ON/OFF
    GpuSceneLight lights[MAX_SCENE_LIGHTS]; // offset 16, 16바이트 정렬 유지
};

// DrawObject: 렌더링 가능한 오브젝트 상태
struct DrawObject {
    uint32_t      indexStart; // 글로벌 인덱스 버퍼에서 이 오브젝트가 시작하는 위치
    uint32_t      indexCount; // 인덱스 수 (삼각형 수 × 3)
    PushConstants push; // 이 오브젝트의 push constant 값 (모델 행렬 포함)

    // 월드 공간 바운딩 구 (프러스텀 컬링·오클루전 컬링에 사용)
    glm::vec3 boundCenter = {}; // 구의 중심
    float     boundRadius = 0.f; // 구의 반지름

    // LOD (Level Of Detail) 레벨
    // lods[0] = LOD1 (중간 품질), lods[1] = LOD2 (저품질/프록시)
    // count == 0 이면 해당 LOD 에서 렌더링 생략
    struct Lod { uint32_t start, count; glm::mat4 model; };
    Lod   lods[2]    = {};
    int   numLods    = 0; // 추가 LOD 레벨 수 (0 = LOD 없음)
    float lodDist[2] = {18.f, 36.f}; // LOD 전환 거리 (m): [LOD1, LOD2]

    // 인스턴싱 그룹 ID (-1 = 인스턴싱 미사용, 개별 draw call)
    int instanceGroupId = -1;

    // 텍스처 인덱스 (-1 = 텍스처 없음, 버텍스 컬러 사용)
    int textureIndex = -1;

    // 오클루전 컬링 예외 플래그 (바닥·하늘 등 항상 렌더링)
    bool skipOcclusion = false;
    bool twoSided = false;
    bool reverseFrontFace = false;
};

// OptFlags: 숫자 키로 토글하는 최적화 플래그
struct OptFlags {
    // 프러스텀 컬링: 시야 밖 오브젝트를 draw call 에서 제외 -> CPU 오버헤드 감소
    // 기본 ON: 시야각 밖 씬이 많을수록 효과가 크고, 시각적 차이 없음
    bool frustumCulling   = true; // 키 1 (기본 ON)

    // LOD: 거리에 따라 낮은 품질 메시로 교체 -> GPU 버텍스 처리량 감소
    bool lod              = false; // 키 2

    // 인스턴싱: 동일 메시를 단일 draw call 로 일괄 렌더링 -> draw call 수 대폭 감소
    bool instancing       = false; // 키 3

    // 백페이스 컬링: 뒤집힌 면을 래스터라이저에서 제거 -> GPU 프래그먼트 처리 감소
    bool backfaceCulling  = true; // 키 4 (기본 ON)

    // 깊이 정렬: 가까운 오브젝트부터 렌더링 -> Early-Z 히트율 향상 (불투명 오브젝트)
    bool depthSort        = false; // 키 5

    // 오클루전 컬링: GPU 오클루전 쿼리로 가려진 오브젝트 제거 -> GPU 프래그먼트 감소
    bool occlusionCulling = false; // 키 6

    // 원거리 컬링: 지정 거리(viewDistMax) 이상의 오브젝트 제거 -> draw call 감소
    bool viewDistCulling  = false; // 키 7

    // 소형 오브젝트 컬링: 화면 투영 반지름이 임계값(smallCullPx) 미만인 오브젝트 제거
    bool smallCulling     = false; // 키 8

    // 디퍼드 셰이딩: G-Buffer 생성 후 전체화면 조명 패스로 합성
    bool deferredShading  = false; // 키 9
};

// SceneLoadTiming: HUD용 씬 로드 단계별 시간
struct SceneLoadTiming {
    std::string mapName; // 로드한 씬 이름
    float totalMs      = 0.f; // reloadScene() 전체 소요 시간 (ms)
    float sceneParseMs = 0.f; // GLTF 파싱 시간 (ms)
    float uploadMs     = 0.f; // GPU 버퍼 업로드 시간 (ms)
    bool  valid        = false; // 유효한 측정 결과인지 여부
};


// Frustum: 프러스텀 컬링용 6개 평면
// 각 평면은 glm::vec4(nx, ny, nz, d) 형식 (안쪽이 법선 방향)
struct Frustum { glm::vec4 planes[6]; };

// Vulkan 물리 장치에서 그래픽·프리젠트 큐 패밀리 인덱스를 저장하는 헬퍼
struct QueueFamilyIndices {
    std::optional<uint32_t> graphics; // 그래픽 명령을 처리하는 큐 패밀리
    std::optional<uint32_t> present; // 스왑체인 프리젠트를 처리하는 큐 패밀리
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

// 스왑체인 지원 정보 (포맷·프리젠트 모드 목록)
struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   presentModes;
};

// VulkanApp: 메인 렌더러 클래스
// 구조:
// 실행 흐름: run() -> initWindow() -> initVulkan() -> mainLoop() -> cleanup()
// 씬 시스템:
// buildScene() -> 씬 파일 파싱 + DrawObject 목록 구성
// reloadScene() -> GPU 드레인 -> 버퍼 삭제 -> buildScene() -> 재업로드
// 최적화 기능 (OptFlags 참조):
// 1. 프러스텀 컬링   2. LOD   3. 인스턴싱
// 4. 백페이스 컬링   5. 깊이 정렬   6. 오클루전 컬링
class VulkanApp {
public:
    static constexpr int      WIDTH  = 1280; // 기본 창 너비 (픽셀)
    static constexpr int      HEIGHT = 720; // 기본 창 높이 (픽셀)
    static constexpr int      MAX_FRAMES_IN_FLIGHT = 2; // 동시 처리 최대 프레임 수 (더블 버퍼링)
    static constexpr int      MAX_INSTANCES        = 65536; // 인스턴싱 SSBO 최대 행렬 수
    static constexpr uint32_t MAX_TEXTURES         = 64; // 씬당 최대 텍스처 슬롯 수
    static constexpr int      MIN_INSTANCES_PER_DRAW = 4; // 작은 그룹은 일반 draw가 더 싸다.
    static constexpr int      OCC_ZERO_FRAMES_TO_HIDE = 3; // 오클루전 쿼리 0 샘플 지속 시 숨김

    void run(); // 앱 진입점: 초기화 -> 메인루프 -> 정리

private:
    // GLFW 창
    GLFWwindow* window = nullptr;
    void initWindow();
    // GLFW 콜백 (정적 함수 -> glfwSetXxxCallback 에 등록)
    static void framebufferResizeCallback(GLFWwindow*, int, int); // 창 크기 변경
    static void cursorPosCallback(GLFWwindow*, double, double); // 마우스 이동
    static void mouseButtonCallback(GLFWwindow*, int, int, int); // 마우스 클릭
    static void scrollCallback(GLFWwindow*, double, double); // 마우스 휠

    // Vulkan 핸들
    VkInstance               instance       = VK_NULL_HANDLE; // Vulkan 인스턴스
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE; // 검증 레이어 메신저
    VkSurfaceKHR             surface        = VK_NULL_HANDLE; // 렌더링 대상 서피스
    VkPhysicalDevice         physDevice     = VK_NULL_HANDLE; // GPU (물리 장치)
    VkDevice                 device         = VK_NULL_HANDLE; // 논리 장치
    VkQueue                  graphicsQueue  = VK_NULL_HANDLE; // 그래픽 커맨드 큐
    VkQueue                  presentQueue   = VK_NULL_HANDLE; // 프리젠트 큐

    // 스왑체인: 화면에 표시할 이미지 큐 (더블/트리플 버퍼링)
    VkSwapchainKHR             swapchain    = VK_NULL_HANDLE;
    std::vector<VkImage>       scImages; // 스왑체인 이미지 배열
    std::vector<VkImageView>   scImageViews; // 이미지 뷰 (렌더패스에서 사용)
    VkFormat                   scFormat{}; // 픽셀 포맷 (e.g. B8G8R8A8_SRGB)
    VkExtent2D                 scExtent{}; // 해상도 (픽셀)

    VkRenderPass               renderPass        = VK_NULL_HANDLE; // 렌더패스 정의

    // 렌더링 파이프라인 (4종)
    // 일반 파이프라인: 백페이스 컬링 ON/OFF 두 가지
    VkDescriptorSetLayout      descSetLayout          = VK_NULL_HANDLE;
    VkPipelineLayout           pipelineLayout         = VK_NULL_HANDLE;
    VkPipeline                 graphicsPipelineFlippedCull = VK_NULL_HANDLE;
    VkPipeline                 graphicsPipeline       = VK_NULL_HANDLE; // 일반, 컬링 ON
    VkPipeline                 graphicsPipelineNoCull = VK_NULL_HANDLE; // 일반, 컬링 OFF
    VkPipeline                 graphicsPipelineAlpha  = VK_NULL_HANDLE; // 알파 블렌딩, 컬링 OFF
    VkPipeline                 graphicsPipelineQueryOnly = VK_NULL_HANDLE; // 오클루전 쿼리용 (색·깊이 쓰기 없음)

    // 인스턴싱 파이프라인: SSBO 에서 행렬을 읽어 일괄 렌더링
    VkDescriptorSetLayout      instanceDescSetLayout       = VK_NULL_HANDLE;
    VkPipelineLayout           instancePipelineLayout      = VK_NULL_HANDLE;
    VkPipeline                 graphicsPipelineInst        = VK_NULL_HANDLE; // 인스턴싱, 컬링 ON
    VkPipeline                 graphicsPipelineInstNoCull  = VK_NULL_HANDLE; // 인스턴싱, 컬링 OFF

    // 깊이 버퍼
    // Z-테스트를 위한 깊이 이미지 (스왑체인 해상도와 동일)
    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthImageView   = VK_NULL_HANDLE;

    // G-Buffer (디퍼드 렌더링)
    // GBuf0=RGBA8(기본색+스페큘러), GBuf1=RGBA16F(법선+shininess), GBuf2=RGBA16F(월드 위치+발광)
    // 프레임 슬롯별로 별도 이미지를 둬 GPU 읽기/쓰기 충돌을 피한다.
    static constexpr int GBUFFER_COLOR_ATTACHMENTS = 4;
    VkImage        gbufImages[GBUFFER_COLOR_ATTACHMENTS][2]    = {}; // [채널][프레임]
    VkDeviceMemory gbufMemories[GBUFFER_COLOR_ATTACHMENTS][2]  = {};
    VkImageView    gbufViews[GBUFFER_COLOR_ATTACHMENTS][2]     = {};
    VkImage        gbufDepthImages[2]  = {}; // G-Buffer 전용 깊이 이미지
    VkDeviceMemory gbufDepthMemories[2]= {};
    VkImageView    gbufDepthViews[2]   = {};
    VkFramebuffer  gbufFramebuffers[2] = {};
    VkRenderPass   gbufRenderPass      = VK_NULL_HANDLE; // G-Buffer 지오메트리 패스
    VkSampler      gbufSampler         = VK_NULL_HANDLE; // 조명 패스에서 G-Buffer를 읽을 샘플러

    // 디퍼드 조명 패스가 G-Buffer 3개를 읽기 위한 디스크립터
    VkDescriptorSetLayout deferredDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      deferredDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       deferredDescSets[2]   = {}; // [프레임]

    // G-Buffer를 채우는 지오메트리 파이프라인
    VkPipeline     gbufPipeline        = VK_NULL_HANDLE; // 백페이스 컬링 ON
    VkPipeline     gbufPipelineNoCull  = VK_NULL_HANDLE; // 백페이스 컬링 OFF

    // 버텍스 버퍼 없이 전체화면 삼각형을 그린다.
    VkPipelineLayout deferredLightLayout   = VK_NULL_HANDLE;
    VkPipeline       deferredLightPipeline = VK_NULL_HANDLE;

    // 방향광(태양) 섀도맵
    // 라이트 시점에서 깊이만 기록한 뒤, 메인 패스에서 binding 3으로 샘플링한다.
    // 해상도는 스왑체인과 무관하게 고정이므로 스왑체인 재생성에 영향받지 않는다.
    // 프레임 인플라이트 충돌을 막기 위해 프레임 슬롯별 이미지를 둔다.
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;
    VkImage          shadowImages[2]        = {}; // [프레임]
    VkDeviceMemory   shadowImageMemories[2] = {};
    VkImageView      shadowImageViews[2]    = {};
    VkFramebuffer    shadowFramebuffers[2]  = {};
    VkSampler        shadowSampler          = VK_NULL_HANDLE; // clamp-to-border, 경계 밖은 밝음
    VkRenderPass     shadowRenderPass       = VK_NULL_HANDLE; // 깊이 전용 패스
    VkPipelineLayout shadowPipelineLayout   = VK_NULL_HANDLE; // push constant = lightMVP
    VkPipeline       shadowPipeline         = VK_NULL_HANDLE;

    std::vector<VkFramebuffer>   framebuffers; // 프레임버퍼 (스왑체인 이미지 수만큼)
    VkCommandPool                commandPool = VK_NULL_HANDLE; // 커맨드 버퍼 풀
    std::vector<VkCommandBuffer> commandBuffers; // 프레임당 커맨드 버퍼

    // 지오메트리 버퍼 (GPU VRAM)
    // 모든 씬 오브젝트의 버텍스/인덱스를 하나의 버퍼에 합쳐 저장
    VkBuffer       vertexBuffer       = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBuffer       indexBuffer        = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory  = VK_NULL_HANDLE;

    // Camera UBO (프레임당)
    // MAX_FRAMES_IN_FLIGHT 만큼 생성해 CPU-GPU 겹치기(overlap) 지원
    std::vector<VkBuffer>       uniformBuffers; // 메인 카메라 UBO
    std::vector<VkDeviceMemory> uniformBufferMemories;
    std::vector<void*>          uniformBuffersMapped; // 영구 매핑 포인터 (vkMapMemory)

    std::vector<VkBuffer>       cullUniformBuffers; // 오클루전 컬링용 별도 카메라 UBO
    std::vector<VkDeviceMemory> cullUniformBufferMemories;
    std::vector<void*>          cullUniformBuffersMapped;

    VkDescriptorPool             descPool = VK_NULL_HANDLE; // 일반 디스크립터 풀
    std::vector<VkDescriptorSet> descSets; // 프레임당 디스크립터 세트
    std::vector<VkDescriptorSet> cullDescSets; // 오클루전 컬링용 세트

    // 인스턴싱 SSBO (프레임당 쓰기 가능)
    // 인스턴스 행렬 배열을 SSBO(Shader Storage Buffer) 로 GPU 에 전달
    // 매 프레임 CPU 에서 업데이트 -> GPU 에서 gl_InstanceIndex 로 인덱싱
    VkDescriptorPool              instanceDescPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet>  instanceDescSets;
    std::vector<VkBuffer>         instanceSSBOs;
    std::vector<VkDeviceMemory>   instanceSSBOMemories;
    std::vector<VkBuffer>         instanceStagingBuffers;
    std::vector<VkDeviceMemory>   instanceStagingMemories;
    std::vector<void*>            instanceSSBOMapped; // 프레임별 staging 버퍼 영구 매핑 포인터

    // 오클루전 쿼리 풀
    // GPU 에 "이 바운딩 박스가 깊이 테스트를 통과하는 픽셀 수"를 질의
    // 결과를 다음 프레임에 읽어 가려진 오브젝트를 렌더링에서 제외
    VkQueryPool              occlusionQueryPool = VK_NULL_HANDLE;
    uint32_t                 occQueryCount      = 0; // 쿼리 슬롯 수 (= drawObjects 수)
    std::vector<uint64_t>    occResults; // 이전 프레임의 쿼리 결과
    std::vector<uint64_t>    occQueryBuf; // vkGetQueryPoolResults 임시 버퍼
    std::vector<uint8_t>     occVisible; // temporal hysteresis가 적용된 가시 상태
    std::vector<uint8_t>     occZeroStreak; // 연속 0 샘플 쿼리 횟수
    int                      occWarmupFrames = 0; // 첫 몇 프레임은 쿼리 결과 무시 (초기화 대기)

    // 동기화 오브젝트
    // 세마포어: GPU-GPU 동기화 (이미지 획득 -> 렌더 -> 프리젠트)
    // 펜스: CPU-GPU 동기화 (프레임이 완료될 때까지 CPU 대기)
    std::vector<VkSemaphore> imageAvailableSems; // 프레임 슬롯별 스왑체인 이미지 사용 가능 신호
    std::vector<VkSemaphore> renderFinishedSems; // 스왑체인 이미지별 렌더링 완료 신호
    std::vector<VkFence>     inFlightFences; // 프레임 인플라이트 펜스
    uint32_t currentFrame       = 0; // 현재 처리 중인 프레임 슬롯 (0 또는 1)
    bool     framebufferResized = false; // 창 크기 변경 -> 스왑체인 재생성 필요 플래그

    // 씬 / 맵 시스템
    std::vector<Vertex>          vertices; // CPU 측 버텍스 배열 (buildScene 에서 채워짐)
    std::vector<uint32_t>        indices; // CPU 측 인덱스 배열
    std::vector<DrawObject>      drawObjects; // 렌더링 가능한 오브젝트 목록
    MeshRange                    occBBoxMesh{}; // 오클루전 컬링용 단위 큐브 바운딩 박스 메시
    std::vector<GLTFTextureData> sceneTextures; // CPU 측 텍스처 데이터 (GLTF 로드 시 채워짐)

    std::string              currentMapFile  = "maps/test_city.glb"; // 현재 로드된 씬 경로
    std::vector<std::string> availableMaps; // maps/ 폴더에서 발견된 .glb 파일 목록
    int                      currentMapIndex = 0; // availableMaps 에서 현재 씬의 인덱스

    // GLTF 파일 로드 -> vertices/indices/drawObjects/sceneTextures 구성
    void buildScene();
    void rebuildInstancingGroups();

    // GPU 안전 씬 교체: GPU 드레인 -> 버퍼 삭제 -> buildScene() -> 재업로드
    void reloadScene();

    // 텍스처 리소스
    // GPU 텍스처 이미지/뷰 배열 (씬 로드마다 재생성)
    std::vector<VkImage>        texImages;
    std::vector<VkDeviceMemory> texMemories;
    std::vector<VkImageView>    texViews;

    // 공통 샘플러 (앱 생명주기 동안 유지)
    VkSampler  texSampler     = VK_NULL_HANDLE;

    // 기본 1×1 흰색 텍스처 – 텍스처 없는 슬롯을 채우기 위한 더미 (앱 생명주기 동안 유지)
    VkImage        defaultTexImage  = VK_NULL_HANDLE;
    VkDeviceMemory defaultTexMemory = VK_NULL_HANDLE;
    VkImageView    defaultTexView   = VK_NULL_HANDLE;

    void createDefaultTextureAndSampler(); // 기본 텍스처 + 샘플러 생성 (initVulkan 에서 1회)
    void createTextureResources(); // sceneTextures -> GPU 업로드 + 디스크립터 세트 갱신
    void cleanupTextureResources(); // GPU 씬 텍스처 리소스 해제 (기본 텍스처 제외)

    // 최적화 상태
    OptFlags optFlags;
    int      renderedCount       = 0; // 이번 프레임에 실제로 draw call 된 오브젝트 수
    int      culledCount         = 0; // 이번 프레임에 컬링으로 제거된 오브젝트 수
    int      instancedGroupCount = 0; // 이번 프레임의 인스턴싱 draw call 수
    SceneLoadTiming lastLoadTiming; // 마지막 씬 로드 타이밍 (HUD 표시)

    // 프레임당 재사용 버퍼 (매 프레임 힙 할당 방지)
    std::vector<int>       frameOrder; // 드로우 순서 배열 (drawObjects.size() 로 resize)
    std::vector<glm::mat4> frameInstMats; // 인스턴싱 스테이징 버퍼 (매 프레임 재사용)
    std::vector<int>       frameInstObjectIndices; // frameInstMats와 같은 순서의 drawObjects 인덱스
    std::vector<uint8_t>   frameInstancedMask; // 이번 프레임에 instanced draw로 처리된 객체

    // 인스턴싱 그룹 정의 (씬 변경 시 재구성)
    // buildScene() 에서 한 번 계산해 두고, 매 프레임 재사용 (std::map 반복 구성 비용 제거)
    struct InstGroupDef {
        int           groupId;
        uint32_t      indexStart = 0, indexCount = 0; // 이 그룹의 메시 범위
        PushConstants push{}; // 공통 push constant
        std::vector<int> members; // drawObjects 배열에서 이 그룹에 속하는 인덱스 목록
    };
    struct InstDrawEntry {
        uint32_t matrixOffset = 0;
        uint32_t instanceCount = 0;
        uint32_t groupDefIndex = 0;
    };
    std::vector<InstGroupDef> instGroupDefs; // 그룹 ID 별 정의 목록
    std::vector<InstDrawEntry> frameInstDrawEntries; // 이번 프레임의 instanced draw 목록

    bool     darkFloor   = false; // B 키: 바닥 어둡게 토글
    float    lightYaw    = 0.0f; // 스크롤 휠로 조정하는 방향광 수평 각도 (도)

    // 씬 전체 바운딩 구 (buildScene 에서 계산) — 방향광 섀도맵 ortho 절두체 피팅에 사용
    glm::vec3 sceneBoundsCenter = glm::vec3(0.0f);
    float     sceneBoundsRadius = 50.0f;
    glm::mat4 currentLightVP    = glm::mat4(1.0f); // 이번 프레임의 라이트 뷰·투영 (섀도 패스에서 재사용)
    float    viewDistMax = 50.0f; // 원거리 컬링 임계 거리 (m)
    float    smallCullPx = 2.0f; // 소형 오브젝트 컬링 임계 화면 반지름 (픽셀)
    bool     extendedFarPlane = false; // 원거리 클리핑 확장 (200 -> 5000)
    float    getFarPlane() const { return extendedFarPlane ? 5000.f : 200.f; }

    // 씬 조명 (GLTF KHR_lights_punctual)
    std::vector<SceneLight>      sceneLights; // buildScene() 에서 채워짐
    bool   sceneLightsOn = true; // 씬 조명 전체 ON/OFF
    bool   ambientOn     = true; // 환경광 ON/OFF
    bool   emissiveOn    = true; // 발광 재질 ON/OFF
    bool   sceneLightDirty = true; // 조명 상태 변경 시 true -> UBO 갱신 후 false

    std::vector<VkBuffer>        sceneLightUBOs;
    std::vector<VkDeviceMemory>  sceneLightUBOMemories;
    std::vector<void*>           sceneLightUBOMapped;

    // F11로 전체화면을 토글한다.
    bool  isFullscreen = false;
    int   windowedX = 100, windowedY = 100; // 창 모드로 복귀할 때 위치
    int   windowedW = WIDTH, windowedH = HEIGHT; // 창 모드로 복귀할 때 크기
    void  toggleFullscreen();

    // 카메라
    Camera   camera; // 배치된(placed) 카메라 – 일반 렌더링 시점
    Camera   observerCamera; // 자유 관찰자(ghost) 카메라 – G 키로 전환
    bool     ghostMode    = false; // G 키: ghost 모드 ON/OFF
    bool     prevGhostKey = false; // 엣지 감지 디바운스 (키 누름 1회만 처리)

    double   lastMouseX = 0, lastMouseY = 0;
    bool     firstMouse = true; // 창 포커스 첫 마우스 이동 시 점프 방지

    // 리플레이 시스템
    bool     isRecording     = false; // 현재 녹화 중
    bool     isReplaying     = false; // 현재 재생 중
    float    recordStartTime = 0.f; // 녹화 시작 시각 (앱 시작 후 초)
    float    replayStartTime = 0.f; // 재생 시작 시각
    int      replayFrameIdx  = 0; // 현재 재생 중인 프레임 인덱스
    std::vector<ReplayFrame> recordedFrames; // 녹화 중 누적되는 프레임 버퍼
    std::vector<ReplayFrame> replayFrames; // 재생용으로 파일에서 불러온 프레임
    std::string replayDir      = "replays"; // 리플레이 파일 저장/로드 디렉토리
    std::string lastSavedReplay; // 마지막으로 저장한 리플레이 경로

    void startRecording(); // 녹화 시작 (recordedFrames 초기화)
    void stopRecording(); // 녹화 종료 + replays/replay_NNN.replay 저장
    void startReplay(const std::string& path = ""); // 재생 시작 ("" = 최신 파일 자동 로드)
    void stopReplay(); // 재생 종료
    float    lastFrameTime = 0.0f; // 이전 프레임의 glfwGetTime() 값 (dt 계산용)

    // 프레임 시간 히스토리 (128프레임 고정 링버퍼, ImGui PlotLines 용)
    static constexpr int FRAME_HISTORY_SIZE = 128;
    float frameTimeHistBuf[FRAME_HISTORY_SIZE] = {};
    int   frameTimeHistIdx   = 0; // 다음 쓸 위치
    int   frameTimeHistCount = 0; // 현재 채워진 수 (최대 128)

    // 벤치마크 로거 (M 키 -> 5초 측정 -> CSV 저장)
    struct BenchmarkSample {
        float fps, frameTimeMs, cpuPercent, gpuPercent;
        int   drawCalls, culled;
    };
    bool  benchmarkActive   = false;
    float benchmarkDuration = 5.0f; // 측정 지속 시간 (초)
    float benchmarkElapsed  = 0.0f; // 현재까지 경과 시간
    std::vector<BenchmarkSample> benchmarkSamples;
    void startBenchmark();
    void finishBenchmark();

    // 자동 벤치마크 (리플레이 기반, M 키)
    // 리플레이 경로가 있을 때 M 키 -> 10 실험(baseline + 9 최적화 기법) × 5회
    // 실험마다 리플레이를 재생하며 성능 측정 -> 결과 CSV 저장
    struct AutoBenchRunResult {
        float avgFps, avgFtMs, minFtMs, maxFtMs;
        float avgCpu, avgGpu;
        int   avgDc, avgCulled;
    };
    struct AutoBenchExp {
        std::string                      name;
        OptFlags                         flags;
        std::vector<AutoBenchRunResult>  runs; // 완료된 실행 결과 (최대 5개)
        std::vector<BenchmarkSample>     current; // 현재 실행 중 수집 중인 샘플
    };
    static constexpr int AUTO_BENCH_RUNS  = 5;
    static constexpr int AUTO_BENCH_TOTAL = 10; // 0=baseline + 1~9=각 최적화 기법

    bool                      autoBenchActive  = false;
    int                       autoBenchExpIdx  = 0; // 현재 실험 인덱스 (0..9)
    int                       autoBenchRunIdx  = 0; // 현재 실험의 반복 인덱스 (0..4)
    int                       autoBenchSkipFrames = 0;
    std::vector<AutoBenchExp> autoBenchExps;
    OptFlags                  autoBenchSavedFlags; // 벤치마크 전 OptFlags 복원용
    bool                      autoBenchSavedGhost = false; // ghost 모드 복원용
    std::string               autoBenchReplayPath; // 사용할 리플레이 파일 경로

    void startAutoBenchmark(); // M 키 -> 자동 벤치마크 시작
    void startAutoBenchRun(); // 현재 실험/반복 리플레이 시작
    void onAutoBenchRunEnd(); // 리플레이 완료 시 호출 -> 다음 실험/반복으로 진행
    void finishAutoBenchmark(); // 전체 완료 -> CSV 저장 + OptFlags 복원

    // 스트레스 배율 (T 키: 1x→2x→4x→8x→16x→1x)
    int  stressLevel = 0; // 0=1x, 1=2x, 2=4x, 3=8x, 4=16x
    std::vector<DrawObject> baseDrawObjects; // stressLevel=0 기준 복사본
    void applyStress();

    // 기즈모 (ghost 모드에서 배치 카메라 위치·프러스텀 시각화)
    static constexpr int GIZMO_MAX_VERTS = 512; // 기즈모 버텍스 최대 수
    std::vector<VkBuffer>       gizmoVBs; // 프레임당 버텍스 버퍼
    std::vector<VkDeviceMemory> gizmoVBMemories;
    std::vector<void*>          gizmoVBMapped; // 영구 매핑 포인터 (매 프레임 CPU 쓰기)
    uint32_t                    gizmoVertCount = 0; // 이번 프레임의 기즈모 버텍스 수

    VkPipeline gizmoPipeline = VK_NULL_HANDLE; // LINE_LIST 토폴로지, 깊이 테스트 OFF

    // ImGui / 성능 오버레이
    VkDescriptorPool  imguiDescPool = VK_NULL_HANDLE; // ImGui 전용 디스크립터 풀
    PerformanceStats  perfStats; // FPS, CPU, RAM, GPU 수집기
    void initImGui(); // ImGui Vulkan 백엔드 초기화
    void cleanupImGui(); // ImGui 리소스 정리
    void drawStatsOverlay(); // 매 프레임 HUD 오버레이 렌더링

    // Vulkan 초기화 단계 (initVulkan() 에서 순서대로 호출)
    void initVulkan();
    void createInstance(); // VkInstance 생성 + 검증 레이어 설정
    void setupDebugMessenger(); // 디버그 메시지 콜백 등록
    void createSurface(); // GLFW 윈도우 서피스 생성
    void pickPhysicalDevice(); // 적합한 GPU 선택
    void createLogicalDevice(); // VkDevice + 큐 생성
    void createSwapChain(); // 스왑체인 + 이미지 생성
    void createImageViews(); // 스왑체인 이미지 뷰 생성
    void createRenderPass(); // 렌더패스 정의 (색상+깊이 attachment)
    void createDescriptorSetLayout(); // 디스크립터 셋 레이아웃 (UBO 바인딩 정의)
    void createGraphicsPipeline(); // 셰이더 컴파일 + 파이프라인 4종 생성
    void createDepthResources(); // 깊이 이미지·뷰 생성
    void createFramebuffers(); // 프레임버퍼 생성 (스왑체인 이미지 수만큼)
    void createCommandPool(); // 커맨드 풀 생성
    void createVertexBuffer(); // 버텍스 버퍼 (스테이징 -> VRAM)
    void createIndexBuffer(); // 인덱스 버퍼 (스테이징 -> VRAM)
    void createSceneBuffers(); // 버텍스+인덱스를 하나의 커맨드 버퍼로 일괄 업로드
    void createUniformBuffers(); // 카메라 UBO 버퍼 (프레임당)
    void createDescriptorPool(); // 디스크립터 풀 생성
    void createDescriptorSets(); // 디스크립터 셋 할당 + UBO 바인딩
    void createCommandBuffers(); // 커맨드 버퍼 할당
    void createSyncObjects(); // 세마포어·펜스 생성
    void recreatePresentSemaphores(); // 스왑체인 이미지별 present 대기 세마포어 재생성
    void createInstanceResources(); // 인스턴싱 SSBO + 디스크립터 셋 생성
    void createOcclusionQueryPool(); // 오클루전 쿼리 풀 생성
    void resetOcclusionState(); // 쿼리 결과와 temporal visibility 상태 초기화
    bool isOcclusionHidden(int objectIndex) const; // hysteresis 이후 숨길지 검사
    void createGizmoPipeline(); // 기즈모 렌더링 파이프라인 생성
    void createGizmoBuffers(); // 기즈모 버텍스 버퍼 (HOST_VISIBLE)
    void buildGizmoGeometry(uint32_t frame); // 매 프레임 기즈모 라인 버텍스 갱신

    void createDeferredResources(); // G-Buffer 이미지·프레임버퍼·디스크립터 생성/재생성
    void destroyDeferredResources(); // G-Buffer 이미지·프레임버퍼·디스크립터 해제

    void createShadowResources(); // 섀도맵 이미지·뷰·샘플러·렌더패스·프레임버퍼 생성
    void createShadowPipeline(); // 섀도맵 깊이 전용 파이프라인 생성
    glm::mat4 computeLightMatrix() const; // 씬 바운딩 구에 맞춘 방향광 ortho 뷰·투영

    void recreateSwapChain(); // 창 크기 변경 시 스왑체인 재생성
    void cleanupSwapChain(); // 스왑체인 관련 리소스 해제

    // 메인루프 / 프레임 처리
    void mainLoop(); // GLFW 이벤트 + drawFrame() 반복
    void drawFrame(); // 한 프레임 전체 처리: 획득 -> 레코드 -> 제출 -> 프리젠트
    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex); // GPU 커맨드 기록
    void updateUniformBuffer(uint32_t frameIndex); // 카메라/조명 UBO 갱신
    void handleOptKeys(); // 숫자 키 1~6 입력으로 OptFlags 토글

    // Vulkan 헬퍼
    QueueFamilyIndices      findQueueFamilies(VkPhysicalDevice dev); // 큐 패밀리 탐색
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev); // 스왑체인 지원 조회
    bool                    isDeviceSuitable(VkPhysicalDevice dev); // GPU 적합성 검사
    VkFormat                findDepthFormat(); // 지원되는 깊이 포맷 탐색
    VkFormat                findSupportedFormat(const std::vector<VkFormat>&,
                                                VkImageTiling, VkFormatFeatureFlags);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props); // 메모리 타입 탐색
    void     createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags,
                          VkBuffer&, VkDeviceMemory&); // 버퍼 + 메모리 할당
    void     copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size); // 펜스 동기화로 버퍼 복사
    void     createImage(uint32_t w, uint32_t h, VkFormat, VkImageTiling,
                         VkImageUsageFlags, VkMemoryPropertyFlags,
                         VkImage&, VkDeviceMemory&); // 이미지 + 메모리 할당
    VkImageView    createImageView(VkImage, VkFormat, VkImageAspectFlags); // 이미지 뷰 생성
    VkShaderModule createShaderModule(const std::vector<char>& code); // SPIR-V -> 셰이더 모듈
    static std::vector<char> readFile(const std::string& path); // 파일 전체 읽기

    // 프러스텀 컬링 헬퍼
    // VP 행렬에서 6개 절두체 평면을 추출 (GL·Vulkan 공통 방법)
    static Frustum extractFrustum(const glm::mat4& vp);
    // 구(center, radius)가 절두체 안에 있으면 true
    static bool    sphereInFrustum(const Frustum& f, glm::vec3 c, float r);

    // AABB 8 코너를 모델 행렬로 변환해 바운딩 구를 계산
    static void computeBoundSphere(DrawObject& obj, glm::vec3 bboxMin, glm::vec3 bboxMax);
    void refreshFrontFaceHints(const Camera& cam);
    bool sampleObjectFrontFaceIsClockwise(const DrawObject& obj, const Camera& cam, bool& outClockwise) const;

    void cleanup(); // 모든 Vulkan 리소스 해제 (역순 정리)
};
