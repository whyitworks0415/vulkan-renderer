#include "MeshLoader.h"
#include "VulkanApp.h"   // Vertex 구조체 정의

#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstring>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <iostream>

// ═════════════════════════════════════════════════════════════════════════════
//  STL 로더
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
//  loadBinary  –  바이너리 STL 파싱
//
//  바이너리 STL 포맷:
//    80 바이트 헤더 (무시)
//    4 바이트 삼각형 수
//    삼각형 수 × 50 바이트:
//      [법선 f32×3][v0 f32×3][v1 f32×3][v2 f32×3][속성 u16]
//
//  법선이 (0,0,0) 이면 외적으로 재계산한다.
// ─────────────────────────────────────────────────────────────────────────────
static MeshRange loadBinary(std::ifstream& f,
                            std::vector<Vertex>& verts,
                            std::vector<uint32_t>& inds,
                            glm::vec3 color)
{
    f.seekg(80); // 헤더 건너뜀
    uint32_t numTris = 0;
    f.read(reinterpret_cast<char*>(&numTris), 4); // 삼각형 수 읽기

    MeshRange r{};
    r.indexStart = static_cast<uint32_t>(inds.size()); // 현재 글로벌 인덱스 배열 끝
    r.bboxMin    = glm::vec3( std::numeric_limits<float>::max());
    r.bboxMax    = glm::vec3(-std::numeric_limits<float>::max());

    uint32_t vBase = static_cast<uint32_t>(verts.size());
    verts.reserve(verts.size() + numTris * 3); // 미리 메모리 확보 (재할당 방지)
    inds.reserve(inds.size()  + numTris * 3);

    for (uint32_t i = 0; i < numTris; ++i) {
        float raw[12]; // [법선 xyz][v0 xyz][v1 xyz][v2 xyz]
        f.read(reinterpret_cast<char*>(raw), 48);
        f.ignore(2); // 속성 바이트 카운트 (무시)

        glm::vec3 n(raw[0], raw[1], raw[2]);
        glm::vec3 v0(raw[3], raw[4], raw[5]);
        glm::vec3 v1(raw[6], raw[7], raw[8]);
        glm::vec3 v2(raw[9], raw[10], raw[11]);

        // 일부 STL 익스포터는 법선을 (0,0,0) 으로 남겨두므로 외적으로 재계산
        if (glm::length(n) < 1e-6f)
            n = glm::normalize(glm::cross(v1 - v0, v2 - v0));
        else
            n = glm::normalize(n);

        uint32_t base = static_cast<uint32_t>(verts.size());
        verts.push_back({v0, n, color});
        verts.push_back({v1, n, color});
        verts.push_back({v2, n, color});
        inds.push_back(base);
        inds.push_back(base + 1);
        inds.push_back(base + 2);

        // AABB 갱신
        for (auto& v : {v0, v1, v2}) {
            r.bboxMin = glm::min(r.bboxMin, v);
            r.bboxMax = glm::max(r.bboxMax, v);
        }
    }

    r.indexCount = static_cast<uint32_t>(inds.size()) - r.indexStart;
    return r;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadAscii  –  ASCII STL 파싱
//
//  ASCII STL 포맷 (텍스트):
//    solid <name>
//      facet normal nx ny nz
//        outer loop
//          vertex x y z
//          vertex x y z
//          vertex x y z
//        endloop
//      endfacet
//    endsolid
//
//  파일 전체를 한 번에 읽어 istringstream 으로 토큰 파싱한다.
// ─────────────────────────────────────────────────────────────────────────────
static MeshRange loadAscii(std::ifstream& f,
                           std::vector<Vertex>& verts,
                           std::vector<uint32_t>& inds,
                           glm::vec3 color)
{
    f.seekg(0); // 파일 처음으로 되돌림
    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    std::istringstream ss(content);

    MeshRange r{};
    r.indexStart = static_cast<uint32_t>(inds.size());
    r.bboxMin    = glm::vec3( std::numeric_limits<float>::max());
    r.bboxMax    = glm::vec3(-std::numeric_limits<float>::max());

    std::string token;
    glm::vec3 faceNormal(0, 0, 1); // 현재 면의 법선 (기본값: +Z)
    std::vector<glm::vec3> triVerts;
    triVerts.reserve(3);

    while (ss >> token) {
        if (token == "facet") {
            ss >> token; // "normal" 키워드 소비
            float nx, ny, nz;
            ss >> nx >> ny >> nz;
            faceNormal = glm::vec3(nx, ny, nz);
            if (glm::length(faceNormal) < 1e-6f) faceNormal = {0,0,1};
            else faceNormal = glm::normalize(faceNormal);
            triVerts.clear();
        } else if (token == "vertex") {
            float x, y, z;
            ss >> x >> y >> z;
            triVerts.push_back({x, y, z});
        } else if (token == "endfacet") {
            if (triVerts.size() == 3) {
                glm::vec3 n = faceNormal;
                // 법선이 없으면 외적으로 재계산
                if (glm::length(n) < 1e-6f)
                    n = glm::normalize(glm::cross(triVerts[1]-triVerts[0], triVerts[2]-triVerts[0]));

                uint32_t base = static_cast<uint32_t>(verts.size());
                for (auto& v : triVerts) {
                    verts.push_back({v, n, color});
                    r.bboxMin = glm::min(r.bboxMin, v);
                    r.bboxMax = glm::max(r.bboxMax, v);
                }
                inds.push_back(base);
                inds.push_back(base + 1);
                inds.push_back(base + 2);
            }
        }
    }

    r.indexCount = static_cast<uint32_t>(inds.size()) - r.indexStart;
    return r;
}

// STL 의 경우 항상 part 가 1개(= whole) 이므로 래퍼로 단일 파트 생성
static LoadedMesh makeSinglePartMesh(const MeshRange& whole) {
    LoadedMesh out{};
    out.whole = whole;
    if (whole.indexCount > 0)
        out.parts.push_back(whole);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadSTLDetailed  –  바이너리/ASCII 자동 감지 후 파싱, whole+parts 반환
//
//  감지 방법:
//    1. 헤더 5바이트가 "solid" 이면 ASCII 후보
//    2. 단, 일부 바이너리 STL 도 "solid" 로 시작하므로
//       파일 크기와 (80 + 4 + n×50) 기대값 비교로 최종 판별
// ─────────────────────────────────────────────────────────────────────────────
LoadedMesh loadSTLDetailed(const std::string& path,
                           std::vector<Vertex>& verts,
                           std::vector<uint32_t>& inds,
                           glm::vec3 color)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open STL: " + path);

    std::streamsize fileSize = f.tellg(); // 파일 크기 (바이트)
    f.seekg(0);

    // 헤더 앞 5바이트 읽어 ASCII 여부 1차 판별
    char header[6] = {};
    f.read(header, 5);
    f.seekg(0);

    bool isAscii = false;
    if (strncmp(header, "solid", 5) == 0) {
        // "solid" 로 시작하더라도 바이너리일 수 있음
        // 삼각형 수 n 을 읽어 기대 파일 크기(80+4+n*50) 와 비교
        uint32_t n = 0;
        f.seekg(80);
        f.read(reinterpret_cast<char*>(&n), 4);
        std::streamsize expectedBinary = 80 + 4 + (std::streamsize)n * 50;
        isAscii = (std::abs(fileSize - expectedBinary) > 10); // 10바이트 허용 오차
        f.seekg(0);
    }

    MeshRange whole = isAscii ? loadAscii(f, verts, inds, color)
                              : loadBinary(f, verts, inds, color);
    return makeSinglePartMesh(whole);
}

// whole 만 필요한 경우의 편의 래퍼
MeshRange loadSTL(const std::string& path,
                  std::vector<Vertex>& verts,
                  std::vector<uint32_t>& inds,
                  glm::vec3 color)
{
    return loadSTLDetailed(path, verts, inds, color).whole;
}

// ═════════════════════════════════════════════════════════════════════════════
//  모델 행렬 빌더
// ═════════════════════════════════════════════════════════════════════════════
//
//  STL/SketchUp OBJ 는 Z-up 좌표계를 사용하고, 씬은 Y-up 을 사용한다.
//  변환 체인 (우→좌 순서로 곱함):
//    R: X축 -90° 회전 → (X, Y_stl, Z_stl) → (X, Z_stl, -Y_stl)
//       즉, 씬의 Y(높이) = STL 의 Z, 씬의 Z(깊이) = -STL 의 Y
//    S: 균일 스케일
//    T: XZ 중심 정렬 + 바닥을 worldPos.y 에 맞춤 + worldPos 로 이동
//
glm::mat4 makeSTLModel(const MeshRange& mesh, float scale, glm::vec3 worldPos)
{
    // R: Z-up → Y-up (X축 -90° 회전)
    glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1, 0, 0));

    // S: 균일 스케일
    glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(scale));

    // R·S 이후 씬 공간의 AABB:
    //   씬 X: [bboxMin.x * scale, bboxMax.x * scale]
    //   씬 Y: [bboxMin.z * scale, bboxMax.z * scale]  (Z → Y)
    //   씬 Z: [-bboxMax.y * scale, -bboxMin.y * scale] (Y → -Z)
    float xCenter = (mesh.bboxMin.x + mesh.bboxMax.x) * 0.5f * scale; // XZ 수평 중심
    float yBottom =  mesh.bboxMin.z * scale; // 메시 바닥 (STL_Z 최솟값 = 씬_Y 최솟값)
    float zCenter = -(mesh.bboxMin.y + mesh.bboxMax.y) * 0.5f * scale; // 깊이 중심

    // 메시를 원점으로 이동시킨 뒤 worldPos 로 옮기는 오프셋
    glm::vec3 alignOffset(-xCenter, -yBottom, -zCenter);

    glm::mat4 T = glm::translate(glm::mat4(1.0f), worldPos + alignOffset);

    // 최종: model = T * S * R  (버텍스에는 오른쪽부터 적용: R → S → T)
    return T * S * R;
}

// ═════════════════════════════════════════════════════════════════════════════
//  OBJ 로더
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
//  parseFaceToken  –  OBJ 면 토큰 파싱
//  OBJ 면 토큰 형식: "v" | "v/vt" | "v//vn" | "v/vt/vn"
//  포지션 인덱스(1-based)를 반환하고 법선 인덱스를 outNrmIdx 에 채움
//  음수 인덱스(상대 참조) 는 호출자가 처리
// ─────────────────────────────────────────────────────────────────────────────
static int parseFaceToken(const std::string& tok, int* outNrmIdx) {
    int vi = 0, vni = 0;
    auto s1 = tok.find('/');
    vi = std::stoi(tok.substr(0, s1)); // 포지션 인덱스
    if (s1 != std::string::npos) {
        auto s2 = tok.find('/', s1 + 1);
        if (s2 != std::string::npos && s2 > s1 + 1)
            vni = std::stoi(tok.substr(s2 + 1)); // v/vt/vn 형식
        else if (s2 == std::string::npos && s1 + 1 < tok.size())
            {} // v/vt 형식 (텍스쳐만 있고 법선 없음)
        else if (s2 != std::string::npos)
            vni = std::stoi(tok.substr(s2 + 1)); // v//vn 형식
    }
    if (outNrmIdx) *outNrmIdx = vni;
    return vi;
}

// ─────────────────────────────────────────────────────────────────────────────
//  parseMTL  –  .mtl 파일에서 재질 이름 → Kd(색상) 맵 구성
//  colorOverride >= 0 이면 호출되지 않음 (외부 색이 우선)
// ─────────────────────────────────────────────────────────────────────────────
static std::unordered_map<std::string, glm::vec3>
parseMTL(const std::string& dir, const std::string& mtlFilename) {
    std::unordered_map<std::string, glm::vec3> mats;
    std::string path = dir + "/" + mtlFilename;
    std::ifstream f(path);
    if (!f.is_open()) return mats; // .mtl 없으면 빈 맵 반환 (색상 없음으로 처리)

    std::string cur;  // 현재 처리 중인 재질 이름
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string cmd; ss >> cmd;
        if (cmd == "newmtl") {
            ss >> cur; // 새 재질 시작
        } else if ((cmd == "Kd" || cmd == "kd") && !cur.empty()) {
            // Diffuse 색상 (RGB 0~1)
            glm::vec3 c;
            ss >> c.r >> c.g >> c.b;
            mats[cur] = c;
        }
    }
    return mats;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadOBJDetailed  –  OBJ 파싱 + 공간 청크 분할, whole+parts 반환
//
//  처리 흐름:
//    1. v/vn/mtllib/usemtl/f 명령 파싱
//    2. 모든 면을 RawFace 로 임시 저장
//    3. 삼각형 수 >= 2048 이면 XY 그리드로 공간 분할 (청크)
//       → 각 청크를 독립 MeshRange(part)로 저장
//       → 추후 청크 단위 프러스텀 컬링 가능
//    4. 법선: 파일에 있으면 사용, 없으면 면법선 계산
//
//  폴리곤 처리: fan triangulation (0-1-2, 0-2-3, ... 순으로 분할)
// ─────────────────────────────────────────────────────────────────────────────
LoadedMesh loadOBJDetailed(const std::string& path,
                           std::vector<Vertex>& verts,
                           std::vector<uint32_t>& inds,
                           glm::vec3 colorOverride)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("loadOBJ: cannot open " + path);

    // .mtl 파일 검색용 디렉토리 (OBJ 파일과 같은 폴더로 가정)
    std::string dir = ".";
    auto slash = path.find_last_of("/\\");
    if (slash != std::string::npos) dir = path.substr(0, slash);

    // OBJ 원시 데이터 버퍼
    std::vector<glm::vec3> positions; // 'v' 명령으로 수집한 위치 목록
    std::vector<glm::vec3> normals;   // 'vn' 명령으로 수집한 법선 목록
    std::unordered_map<std::string, glm::vec3> materials; // 재질명 → Kd 색상

    // colorOverride >= 0 이면 단일 색상 사용, 아니면 MTL 의 Kd 사용
    glm::vec3 curColor = (colorOverride.r >= 0.f) ? colorOverride : glm::vec3(0.8f);

    // 면을 일단 임시 저장 (나중에 공간 분할 후 버텍스 배열에 채움)
    struct RawFace {
        std::array<int,3> posIdx; // 포지션 인덱스 (1-based, 음수 = 상대)
        std::array<int,3> nrmIdx; // 법선 인덱스 (1-based, 0 = 없음)
        glm::vec3 color;           // 이 면의 색상 (usemtl 기준)
    };
    std::vector<RawFace> rawFaces;
    int posCount = 0; // 상대 인덱스 해석을 위해 face 읽는 시점의 포지션 수 기록

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash); // 주석 제거
        std::istringstream ss(line);
        std::string cmd; ss >> cmd;

        if (cmd == "v") {
            // 버텍스 위치
            glm::vec3 p; ss >> p.x >> p.y >> p.z;
            positions.push_back(p);
        }
        else if (cmd == "vn") {
            // 버텍스 법선 (옵션)
            glm::vec3 n; ss >> n.x >> n.y >> n.z;
            normals.push_back(n);
        }
        else if (cmd == "mtllib") {
            // 재질 파일 참조 – colorOverride 가 없을 때만 로드
            std::string mtlFile; ss >> mtlFile;
            if (colorOverride.r < 0.f)
                materials = parseMTL(dir, mtlFile);
        }
        else if (cmd == "usemtl") {
            // 이후 면에 적용할 재질 변경
            std::string name; ss >> name;
            if (colorOverride.r < 0.f) {
                auto it = materials.find(name);
                if (it != materials.end()) curColor = it->second;
            }
        }
        else if (cmd == "f") {
            // 면 정의: 토큰 수집 후 fan triangulation
            std::vector<std::string> tokens;
            std::string tok;
            while (ss >> tok) tokens.push_back(tok);
            if (tokens.size() < 3) continue;

            posCount = (int)positions.size(); // 상대 인덱스 해석 기준
            for (int t = 1; t + 1 < (int)tokens.size(); ++t) {
                RawFace face;
                face.color = curColor;
                int vi[3] = {0, t, t + 1}; // fan: [0, t, t+1]
                for (int k = 0; k < 3; ++k) {
                    int nk = 0;
                    int pk = parseFaceToken(tokens[vi[k]], &nk);
                    // 음수 인덱스 → 상대 참조로 변환 (OBJ 표준)
                    face.posIdx[k] = pk < 0 ? posCount + pk + 1 : pk;
                    face.nrmIdx[k] = nk < 0 ? (int)normals.size() + nk + 1 : nk;
                }
                rawFaces.push_back(face);
            }
        }
    }

    // ── whole 범위 초기화 ───────────────────────────────────────────────────
    LoadedMesh out{};
    out.whole.indexStart = static_cast<uint32_t>(inds.size());
    out.whole.bboxMin    = glm::vec3( std::numeric_limits<float>::max());
    out.whole.bboxMax    = glm::vec3(-std::numeric_limits<float>::max());

    if (rawFaces.empty()) {
        out.whole.indexCount = 0;
        out.whole.bboxMin = {};
        out.whole.bboxMax = {};
        return out;
    }

    // 인덱스 → 포지션 변환 람다 (범위 체크 포함)
    auto resolvePos = [&](int idx1Based) -> glm::vec3 {
        int idx = idx1Based - 1;
        if (idx < 0 || idx >= (int)positions.size()) return {};
        return positions[idx];
    };

    // ── 전체 AABB 와 XY 스팬 계산 (공간 분할 그리드 크기 결정용) ───────────
    glm::vec2 minXY( std::numeric_limits<float>::max());
    glm::vec2 maxXY(-std::numeric_limits<float>::max());
    for (const auto& face : rawFaces) {
        for (int k = 0; k < 3; ++k) {
            glm::vec3 p = resolvePos(face.posIdx[k]);
            out.whole.bboxMin = glm::min(out.whole.bboxMin, p);
            out.whole.bboxMax = glm::max(out.whole.bboxMax, p);
            minXY = glm::min(minXY, glm::vec2(p.x, p.y));
            maxXY = glm::max(maxXY, glm::vec2(p.x, p.y));
        }
    }

    // ── 공간 그리드 분할 (대형 메시 최적화) ────────────────────────────────
    // 2048 삼각형 미만이면 분할하지 않음 (오버헤드가 이득보다 큼)
    const int triCount = (int)rawFaces.size();
    const int targetTrisPerChunk = 768;  // 청크당 목표 삼각형 수
    const int maxChunkAxis = 16;         // 한 축 최대 청크 수 (16×16 = 256 청크)
    glm::vec2 spanXY = maxXY - minXY;

    int gridX = 1, gridY = 1;
    if (triCount >= 2048 && (spanXY.x > 1e-3f || spanXY.y > 1e-3f)) {
        int desiredChunks = std::max(1, (triCount + targetTrisPerChunk - 1) / targetTrisPerChunk);
        // 가로세로 비율에 맞춰 그리드 크기 결정 (정사각형에 가깝게)
        float aspect = (spanXY.y > 1e-3f) ? (spanXY.x / spanXY.y) : 1.f;
        aspect = std::max(aspect, 1e-3f);

        gridX = std::clamp((int)std::round(std::sqrt((float)desiredChunks * aspect)), 1, maxChunkAxis);
        gridY = std::clamp((desiredChunks + gridX - 1) / gridX, 1, maxChunkAxis);
    }

    // ── 면 → 그리드 버킷 분류 (중심점 좌표 기준) ───────────────────────────
    std::vector<std::vector<int>> buckets(gridX * gridY);
    for (int fi = 0; fi < triCount; ++fi) {
        const auto& face = rawFaces[fi];
        glm::vec3 p0 = resolvePos(face.posIdx[0]);
        glm::vec3 p1 = resolvePos(face.posIdx[1]);
        glm::vec3 p2 = resolvePos(face.posIdx[2]);
        glm::vec2 centroid((p0.x + p1.x + p2.x) / 3.f,
                           (p0.y + p1.y + p2.y) / 3.f);

        // 정규화 좌표 [0,1] → 그리드 셀 인덱스
        int bx = 0, by = 0;
        if (gridX > 1 && spanXY.x > 1e-3f) {
            float tx = (centroid.x - minXY.x) / spanXY.x;
            bx = std::clamp((int)(tx * gridX), 0, gridX - 1);
        }
        if (gridY > 1 && spanXY.y > 1e-3f) {
            float ty = (centroid.y - minXY.y) / spanXY.y;
            by = std::clamp((int)(ty * gridY), 0, gridY - 1);
        }
        buckets[by * gridX + bx].push_back(fi);
    }

    // ── 각 버킷 → MeshRange(part) 생성 ────────────────────────────────────
    for (const auto& bucket : buckets) {
        if (bucket.empty()) continue; // 빈 버킷 건너뜀

        MeshRange part{};
        part.indexStart = static_cast<uint32_t>(inds.size());
        part.bboxMin    = glm::vec3( std::numeric_limits<float>::max());
        part.bboxMax    = glm::vec3(-std::numeric_limits<float>::max());

        for (int fi : bucket) {
            const auto& face = rawFaces[fi];
            glm::vec3 p[3];
            for (int k = 0; k < 3; ++k)
                p[k] = resolvePos(face.posIdx[k]);

            // 면법선 계산 (외적)
            glm::vec3 rawNormal = glm::cross(p[1] - p[0], p[2] - p[0]);
            glm::vec3 faceNormal = (glm::dot(rawNormal, rawNormal) > 1e-12f)
                                 ? glm::normalize(rawNormal)
                                 : glm::vec3(0.f, 0.f, 1.f); // 퇴화 삼각형 폴백

            uint32_t base = static_cast<uint32_t>(verts.size());
            for (int k = 0; k < 3; ++k) {
                Vertex vtx{};
                vtx.pos   = p[k];
                vtx.color = face.color;

                // 파일에 법선이 있으면 사용, 없으면 면법선 대입
                int ni = face.nrmIdx[k] - 1;
                if (ni >= 0 && ni < (int)normals.size()) {
                    glm::vec3 n = normals[ni];
                    vtx.normal = (glm::dot(n, n) > 1e-6f) ? glm::normalize(n) : faceNormal;
                } else {
                    vtx.normal = faceNormal;
                }

                part.bboxMin = glm::min(part.bboxMin, vtx.pos);
                part.bboxMax = glm::max(part.bboxMax, vtx.pos);
                verts.push_back(vtx);
            }

            inds.push_back(base);
            inds.push_back(base + 1);
            inds.push_back(base + 2);
        }

        part.indexCount = static_cast<uint32_t>(inds.size()) - part.indexStart;
        out.parts.push_back(part);
    }

    out.whole.indexCount = static_cast<uint32_t>(inds.size()) - out.whole.indexStart;
    if (out.parts.empty())
        out.parts.push_back(out.whole); // 분할 없으면 whole = 유일 part

    std::cout << "  [OBJ] " << path << " -> "
              << triCount << " triangles, "
              << out.parts.size() << " spatial chunk(s)\n";
    return out;
}

// whole 만 필요한 경우의 편의 래퍼
MeshRange loadOBJ(const std::string& path,
                  std::vector<Vertex>& verts,
                  std::vector<uint32_t>& inds,
                  glm::vec3 colorOverride)
{
    return loadOBJDetailed(path, verts, inds, colorOverride).whole;
}

// ═════════════════════════════════════════════════════════════════════════════
//  확장자 기반 통합 디스패처
// ═════════════════════════════════════════════════════════════════════════════

LoadedMesh loadMeshDetailed(const std::string& path,
                            std::vector<Vertex>& verts,
                            std::vector<uint32_t>& inds,
                            glm::vec3 colorOverride)
{
    // 확장자 추출 후 소문자 변환
    auto ext = path.substr(path.find_last_of('.') + 1);
    for (auto& c : ext) c = (char)tolower(c);

    if (ext == "obj")
        return loadOBJDetailed(path, verts, inds, colorOverride);
    else // stl (기본값)
    {
        // STL 은 colorOverride < 0 이면 기본 회색으로 대체
        glm::vec3 col = (colorOverride.r >= 0.f) ? colorOverride : glm::vec3(0.8f);
        return loadSTLDetailed(path, verts, inds, col);
    }
}

// whole 만 필요한 경우의 편의 래퍼
MeshRange loadMesh(const std::string& path,
                   std::vector<Vertex>& verts,
                   std::vector<uint32_t>& inds,
                   glm::vec3 colorOverride)
{
    return loadMeshDetailed(path, verts, inds, colorOverride).whole;
}
