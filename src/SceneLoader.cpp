#include "SceneLoader.h"

#include <algorithm>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <dirent.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  findMapFiles  –  dir 안의 *.glb / *.gltf 파일 목록을 이름순으로 반환
//  maps/ 폴더에 Blender 에서 내보낸 GLB/GLTF 파일을 두면 자동으로 탐색된다.
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> findMapFiles(const std::string& dir) {
    std::vector<std::string> result;

#ifdef _WIN32
    for (const char* pattern : {"*.glb", "*.gltf"}) {
        WIN32_FIND_DATAA fd;
        std::string fullPattern = dir + "\\" + pattern;
        HANDLE hFind = FindFirstFileA(fullPattern.c_str(), &fd);
        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                    result.push_back(dir + "/" + fd.cFileName);
            } while (FindNextFileA(hFind, &fd));
            FindClose(hFind);
        }
    }
#else
    DIR* d = opendir(dir.c_str());
    if (d) {
        struct dirent* ent;
        while ((ent = readdir(d)) != nullptr) {
            std::string fname = ent->d_name;
            auto hasSuffix = [&](const std::string& s) {
                return fname.size() > s.size() &&
                       fname.substr(fname.size() - s.size()) == s;
            };
            if (hasSuffix(".glb") || hasSuffix(".gltf"))
                result.push_back(dir + "/" + fname);
        }
        closedir(d);
    }
#endif

    std::sort(result.begin(), result.end());
    return result;
}
