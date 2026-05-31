#pragma once

#include <string>
#include <vector>

#include <glm/glm.hpp>

struct SceneMeshDesc {
    std::string id;
    std::string path;
    glm::vec3   color = {0.8f, 0.8f, 0.8f};
    float       alpha = 1.0f;
    float       shininess = 32.0f;
    float       specular = 0.3f;
    float       reflect = 0.0f;
};

struct SceneGridDesc {
    std::string meshId;
    int         cells = 0;
    float       cellSize = 1.0f;
    int         instanceGroup = -1;
    float       scale = 1.0f;
    bool        evenOnly = false;
    bool        hasExclude = false;
    int         exC0 = 0, exC1 = 0, exR0 = 0, exR1 = 0;
    bool        hasRange = false;
    int         ranC0 = 0, ranC1 = 0, ranR0 = 0, ranR1 = 0;
    bool        hasHollow = false;
    int         holC0 = 0, holC1 = 0, holR0 = 0, holR1 = 0;
};

struct SceneObjectDesc {
    std::string meshId;
    glm::vec3   pos = {};
    float       scale = 1.0f;
    int         instanceGroup = -1;
};

struct SceneDesc {
    std::string name = "Scene";
    glm::vec3   cameraPos = {0.0f, 1.8f, -8.0f};
    float       cameraYaw = 90.0f;
    float       cameraPitch = -5.0f;
    bool        hasFloor = false;
    float       floorSize = 60.0f;
    int         floorDivs = 1;
    std::vector<SceneMeshDesc>   meshes;
    std::vector<SceneGridDesc>   grids;
    std::vector<SceneObjectDesc> objects;
};

std::vector<std::string> findMapFiles(const std::string& dir);
bool loadScene(const std::string& path, SceneDesc& out);
