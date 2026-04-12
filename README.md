# Vulkan 3D Renderer

Vulkan API 기반 실시간 3D 렌더러. 다양한 렌더링 최적화 기법을 개별적으로 ON/OFF하여 성능 차이를 시각적으로 비교할 수 있다.

## 요구사항

| 항목 | 버전 |
|------|------|
| C++ 컴파일러 | C++17 이상 (MSVC 권장) |
| CMake | 3.20 이상 |
| Vulkan SDK | 1.3 이상 |
| vcpkg | 최신 |

### vcpkg 패키지

```
vcpkg install glfw3 glm imgui[core,glfw-binding,vulkan-binding] tinygltf tinyobjloader
```

## 빌드

### 1. 셰이더 컴파일 (SPIR-V)

Vulkan SDK의 `glslc`를 사용하여 셰이더를 컴파일한다.

```bash
mkdir -p shaders/spv

glslc shaders/scene.vert           -o shaders/spv/scene.vert.spv
glslc shaders/scene.frag           -o shaders/spv/scene.frag.spv
glslc shaders/scene_instanced.vert -o shaders/spv/scene_instanced.vert.spv
glslc shaders/gbuffer.frag         -o shaders/spv/gbuffer.frag.spv
glslc shaders/deferred_light.vert  -o shaders/spv/deferred_light.vert.spv
glslc shaders/deferred_light.frag  -o shaders/spv/deferred_light.frag.spv
glslc shaders/gizmo.vert           -o shaders/spv/gizmo.vert.spv
glslc shaders/gizmo.frag           -o shaders/spv/gizmo.frag.spv
```

Windows에서 `glslc`는 보통 `C:\VulkanSDK\<version>\Bin\glslc.exe`에 있다.

### 2. CMake 빌드

```bash
# vcpkg 툴체인 경로를 지정하여 CMake 구성
cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=<vcpkg-root>/scripts/buildsystems/vcpkg.cmake

# Release 빌드
cmake --build build --config Release
```

빌드 완료 시 `build/Release/VulkanRenderer.exe`가 생성되며, 셰이더/에셋/맵이 자동 복사된다.

### 3. 맵 파일 배치

`.glb`, `.gltf`, `.obj` 맵 파일을 `maps/` 폴더 또는 하위 폴더에 넣는다. 프로그램이 재귀적으로 탐색한다.

```
maps/
  test_city.glb
  InfernoWorld/
    InfernoWorld.glb
    InfernoWorld.mtl
    Lava.png
    ...
```

텍스처 파일(`.png`, `.jpg` 등)은 `.glb`/`.obj`와 같은 폴더에 배치한다.

## 실행

```bash
cd build/Release
./VulkanRenderer.exe
```

## 조작법

### 카메라

| 키 | 동작 |
|---|---|
| W/A/S/D | 이동 |
| 마우스 | 시점 회전 |
| Shift | 빠른 이동 |
| 스크롤 휠 | 방향광 회전 |

### 맵 / 모드

| 키 | 동작 |
|---|---|
| TAB | 다음 맵으로 전환 |
| G | Ghost(관찰자) 모드 ON/OFF |
| F11 | 전체화면 토글 |
| ESC | 종료 |

### 최적화 기법 토글

| 키 | 기법 |
|---|---|
| 1 | Frustum Culling |
| 2 | LOD (Level of Detail) |
| 3 | GPU Instancing |
| 4 | Backface Culling |
| 5 | Front-Back Depth Sort |
| 6 | Occlusion Culling |
| 7 | View Distance Culling |
| 8 | Small Object Culling |
| 9 | Deferred Shading |
| 0 | 전체 ON/OFF 토글 |

### 렌더링 설정

| 키 | 동작 |
|---|---|
| B | 바닥 밝기 토글 (밝은/어두운) |
| F | Far Plane 확장 (200 / 5000) |
| L | Scene Lights ON/OFF |
| N | Ambient Light ON/OFF |
| V | Emissive ON/OFF |

### 녹화 / 리플레이 / 벤치마크

| 키 | 동작 |
|---|---|
| R | 카메라 경로 녹화 시작/정지 |
| P | 리플레이 재생/정지 |
| M | 벤치마크 시작 (리플레이 있으면 자동 벤치마크: 10실험 x 5회) |
| T | 스트레스 배율 순환 (1x / 2x / 4x / 8x / 16x) |

### 벤치마크 결과

벤치마크 완료 시 `results/` 폴더에 CSV 파일이 저장된다.

- `bench_*.csv` - 수동 벤치마크 (5초 측정)
- `autobench_*.csv` - 자동 벤치마크 (10 실험별 baseline 대비 FPS 개선율 포함)

## 프로젝트 구조

```
src/
  VulkanApp.cpp/h    - 메인 렌더러 (Vulkan 초기화, 렌더 루프, UI)
  Camera.cpp/h       - FPS 카메라
  GLTFLoader.cpp/h   - .glb/.gltf 로더 (tinygltf)
  OBJLoader.cpp/h    - .obj 로더 (tinyobjloader)
  MeshLoader.cpp/h   - .stl/.scene 로더
  SceneLoader.cpp/h  - 맵 파일 탐색
  Replay.cpp/h       - 카메라 경로 녹화/재생
  PerformanceStats.*  - FPS/CPU/GPU 통계
shaders/
  scene.vert/frag           - Forward 렌더링
  scene_instanced.vert      - GPU Instancing
  gbuffer.frag              - G-Buffer (Deferred)
  deferred_light.vert/frag  - Deferred Lighting
  gizmo.vert/frag           - Ghost 모드 시각화
maps/                       - 맵 파일 (.glb, .gltf, .obj)
```
