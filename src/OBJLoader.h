#pragma once
#include <string>
#include <vector>

struct Vertex;
struct DrawObject;
struct GLTFTextureData;
struct GLTFSceneDesc;

// loadOBJScene
// .obj 파일을 읽어 각 named object/group(o/g 태그)을 별도의 DrawObject 로
// 변환한다. 오브젝트마다 개별 바운딩 구를 계산하므로 프러스텀/오클루전 컬링을
// 그대로 적용할 수 있다.
// 텍스처: .mtl 의 map_Kd 경로를 읽어 stb_image 로 RGBA8 디코딩한다.
// 텍스처가 없는 오브젝트는 Kd(확산광 색상)를 버텍스 컬러로 사용한다.
// 인자:
// path      – .obj 파일 경로
// verts     – 글로벌 버텍스 배열 (데이터를 뒤에 추가함)
// inds      – 글로벌 인덱스 배열 (데이터를 뒤에 추가함)
// objects   – DrawObject 목록 (항목을 뒤에 추가함)
// sceneDesc – 씬 이름·카메라 초기 위치 반환
// textures  – RGBA8 디코딩된 텍스처 목록 (항목을 뒤에 추가함)
// 반환: 성공 true, 실패 false
bool loadOBJScene(const std::string&          path,
                  std::vector<Vertex>&         verts,
                  std::vector<uint32_t>&       inds,
                  std::vector<DrawObject>&     objects,
                  GLTFSceneDesc&               sceneDesc,
                  std::vector<GLTFTextureData>& textures);
