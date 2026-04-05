#pragma once
#include <string>
#include <vector>

// maps/ 디렉토리에서 .glb / .gltf 파일 경로 목록을 이름순으로 반환 (비재귀)
std::vector<std::string> findMapFiles(const std::string& dir);
