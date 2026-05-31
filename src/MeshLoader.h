#pragma once
#include <string>
#include <vector>
#include <glm/glm.hpp>

struct Vertex;

// MeshRange: 글로벌 버퍼 안의 메시 범위
// Vulkan 은 단일 vertex+index 버퍼에 모든 메시를 이어붙인 후
// vkCmdDrawIndexed(indexCount, 1, indexStart, ...) 로 부분 드로우한다.
// MeshRange 는 그 '부분 범위' 를 기록하는 구조체다.
struct MeshRange {
    uint32_t  indexStart; // indices 배열에서 이 메시가 시작하는 오프셋
    uint32_t  indexCount; // 이 메시의 인덱스 수 (삼각형 수 × 3)
    glm::vec3 bboxMin; // 축 정렬 경계 박스 최솟값 (원본 좌표계)
    glm::vec3 bboxMax; // 축 정렬 경계 박스 최댓값 (원본 좌표계)
};

// LoadedMesh: 전체 범위와 공간 청크 목록
// 대형 OBJ 파일은 공간 그리드로 청크를 나눠 저장한다.
// 프러스텀 컬링이나 LOD 에서 part 단위로 컬링해 오버드로를 줄일 수 있다.
// STL 은 parts 가 1개 (= whole 과 동일).
struct LoadedMesh {
    MeshRange              whole{}; // 전체 메시를 하나의 범위로 표현
    std::vector<MeshRange> parts; // 공간 분할 청크 목록 (없으면 whole 과 동일)
};

// STL 로더
// 바이너리 STL과 ASCII STL 을 자동 감지해 파싱한다.
// 모든 버텍스에 color 를 부여하고 법선이 없으면 면법선으로 재계산한다.

// whole 만 반환 (단순 사용 시)
MeshRange loadSTL(const std::string& path,
                  std::vector<Vertex>& verts,
                  std::vector<uint32_t>& inds,
                  glm::vec3 color);

// whole + parts 반환 (LOD·청크 컬링 등 상세 사용 시)
LoadedMesh loadSTLDetailed(const std::string& path,
                           std::vector<Vertex>& verts,
                           std::vector<uint32_t>& inds,
                           glm::vec3 color);

// Wavefront OBJ 로더
// - 폴리곤 -> 삼각형 팬 분할(fan triangulation)
// - colorOverride >= 0 이면 모든 면에 그 색 사용
// - colorOverride < 0 이면 .mtl 의 Kd 값을 재질별로 사용
// - 법선이 없는 면은 면법선(flat normal) 으로 자동 계산
// - 대형 메시(2048 삼각형 이상)는 XY 그리드로 공간 청크 분할

// whole 만 반환
MeshRange loadOBJ(const std::string& path,
                  std::vector<Vertex>& verts,
                  std::vector<uint32_t>& inds,
                  glm::vec3 colorOverride = glm::vec3(-1.f));

// whole + parts 반환
LoadedMesh loadOBJDetailed(const std::string& path,
                           std::vector<Vertex>& verts,
                           std::vector<uint32_t>& inds,
                           glm::vec3 colorOverride = glm::vec3(-1.f));

// 확장자 기반 메시 로더 선택
// .stl -> loadSTL / .obj -> loadOBJ 로 자동 분기

// whole 만 반환
MeshRange loadMesh(const std::string& path,
                   std::vector<Vertex>& verts,
                   std::vector<uint32_t>& inds,
                   glm::vec3 colorOverride = glm::vec3(-1.f));

// whole + parts 반환
LoadedMesh loadMeshDetailed(const std::string& path,
                            std::vector<Vertex>& verts,
                            std::vector<uint32_t>& inds,
                            glm::vec3 colorOverride = glm::vec3(-1.f));

// makeSTLModel  –  STL/OBJ 메시용 모델 행렬 생성
// 변환 체인 (우→좌 순서로 적용):
// 1. R: X축 -90° 회전 -> Z-up(STL/SketchUp 관례) 을 Y-up(씬 관례) 으로 변환
// 2. S: 균일 스케일
// 3. T: XZ 중심 정렬 + 바닥을 Y=worldPos.y 에 맞춤 + worldPos 로 이동
// 결과: 메시 바닥이 바닥면(Y=0)에 닿고, XZ 중심이 worldPos 에 위치
glm::mat4 makeSTLModel(const MeshRange& mesh, float scale, glm::vec3 worldPos);
