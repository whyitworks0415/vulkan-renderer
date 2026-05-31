#include "SceneLoader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string trim(std::string s) {
    auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [&](char c) { return !isSpace((unsigned char)c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [&](char c) { return !isSpace((unsigned char)c); }).base(), s.end());
    return s;
}

std::string lowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

std::string normalizedPath(std::filesystem::path path) {
    std::string p = path.generic_string();
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

bool isMapExtension(const std::string& ext) {
    return ext == ".scene" || ext == ".glb" || ext == ".gltf" ||
           ext == ".obj" || ext == ".blend";
}

int mapDirectoryPriority(const std::string& path) {
    std::string p = lowerCopy(path);
    if (p.find("/map/") != std::string::npos) return 0;
    if (p.find("/maps/") != std::string::npos) return 1;
    return 2;
}

bool naturalLess(const std::string& a, const std::string& b) {
    int pa = mapDirectoryPriority(a);
    int pb = mapDirectoryPriority(b);
    if (pa != pb) return pa < pb;

    std::string la = lowerCopy(a);
    std::string lb = lowerCopy(b);
    size_t ia = 0, ib = 0;
    while (ia < la.size() && ib < lb.size()) {
        if (std::isdigit((unsigned char)la[ia]) && std::isdigit((unsigned char)lb[ib])) {
            size_t ea = ia;
            size_t eb = ib;
            while (ea < la.size() && std::isdigit((unsigned char)la[ea])) ++ea;
            while (eb < lb.size() && std::isdigit((unsigned char)lb[eb])) ++eb;

            std::string na = la.substr(ia, ea - ia);
            std::string nb = lb.substr(ib, eb - ib);
            na.erase(0, std::min(na.find_first_not_of('0'), na.size() - 1));
            nb.erase(0, std::min(nb.find_first_not_of('0'), nb.size() - 1));
            if (na.size() != nb.size()) return na.size() < nb.size();
            if (na != nb) return na < nb;
            ia = ea;
            ib = eb;
            continue;
        }
        if (la[ia] != lb[ib]) return la[ia] < lb[ib];
        ++ia;
        ++ib;
    }
    return la.size() < lb.size();
}

} // namespace

std::vector<std::string> findMapFiles(const std::string& dir) {
    std::vector<std::string> result;
    namespace fs = std::filesystem;

    std::error_code ec;
    for (auto& entry : fs::recursive_directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = lowerCopy(entry.path().extension().string());
        if (!isMapExtension(ext)) continue;

        if (ext == ".blend") {
            fs::path glb = entry.path();
            glb.replace_extension(".glb");
            if (fs::exists(glb, ec)) continue;
        }

        result.push_back(normalizedPath(entry.path()));
    }

    std::sort(result.begin(), result.end(), naturalLess);
    return result;
}

bool loadScene(const std::string& path, SceneDesc& out) {
    std::ifstream file(path);
    if (!file.is_open()) return false;

    out = SceneDesc{};
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;

        std::istringstream ss(line);
        std::string cmd;
        ss >> cmd;

        if (cmd == "name") {
            std::string rest;
            std::getline(ss, rest);
            rest = trim(rest);
            if (!rest.empty()) out.name = rest;
        } else if (cmd == "camera_pos") {
            ss >> out.cameraPos.x >> out.cameraPos.y >> out.cameraPos.z;
        } else if (cmd == "camera_yaw") {
            ss >> out.cameraYaw;
        } else if (cmd == "camera_pitch") {
            ss >> out.cameraPitch;
        } else if (cmd == "floor") {
            out.hasFloor = true;
            ss >> out.floorSize >> out.floorDivs;
        } else if (cmd == "mesh") {
            SceneMeshDesc m;
            ss >> m.id >> m.path
               >> m.color.r >> m.color.g >> m.color.b >> m.alpha
               >> m.shininess >> m.specular >> m.reflect;
            if (!m.id.empty() && !m.path.empty())
                out.meshes.push_back(m);
        } else if (cmd == "object") {
            SceneObjectDesc obj;
            ss >> obj.meshId >> obj.pos.x >> obj.pos.y >> obj.pos.z >> obj.scale;
            if (!(ss >> obj.instanceGroup)) obj.instanceGroup = -1;
            if (!obj.meshId.empty())
                out.objects.push_back(obj);
        } else if (cmd == "grid") {
            SceneGridDesc g;
            ss >> g.meshId >> g.cells >> g.cellSize >> g.instanceGroup;
            std::string tok;
            while (ss >> tok) {
                if (tok == "even_only") {
                    g.evenOnly = true;
                } else if (tok == "exclude") {
                    g.hasExclude = true;
                    ss >> g.exC0 >> g.exC1 >> g.exR0 >> g.exR1;
                } else if (tok == "range") {
                    g.hasRange = true;
                    ss >> g.ranC0 >> g.ranC1 >> g.ranR0 >> g.ranR1;
                } else if (tok == "hollow") {
                    g.hasHollow = true;
                    ss >> g.holC0 >> g.holC1 >> g.holR0 >> g.holR1;
                } else if (tok == "scale") {
                    ss >> g.scale;
                }
            }
            if (!g.meshId.empty() && g.cells > 0)
                out.grids.push_back(g);
        }
    }

    return !out.meshes.empty() || out.hasFloor || !out.objects.empty() || !out.grids.empty();
}
