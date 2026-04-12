#include "SceneLoader.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  findMapFiles  –  dir 및 하위 폴더에서 *.glb / *.gltf / *.obj 파일을 재귀 탐색
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> findMapFiles(const std::string& dir) {
    std::vector<std::string> result;
    namespace fs = std::filesystem;

    std::error_code ec;
    for (auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        // 소문자 변환
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".glb" || ext == ".gltf" || ext == ".obj") {
            // 경로 구분자를 '/'로 통일
            std::string p = entry.path().string();
            std::replace(p.begin(), p.end(), '\\', '/');
            result.push_back(p);
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}
