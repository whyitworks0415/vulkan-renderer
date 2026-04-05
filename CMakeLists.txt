cmake_minimum_required(VERSION 3.20)
project(VulkanRenderer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Vulkan ────────────────────────────────────────────────────────────────────
find_package(Vulkan REQUIRED)

# ── GLFW (via vcpkg or system) ────────────────────────────────────────────────
find_package(glfw3 REQUIRED)

# ── GLM (header-only) ─────────────────────────────────────────────────────────
find_package(glm REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_path(TINYGLTF_INCLUDE_DIRS "tiny_gltf.h")

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/VulkanApp.cpp
    src/Camera.cpp
    src/MeshLoader.cpp
    src/PerformanceStats.cpp
    src/SceneLoader.cpp
    src/Replay.cpp
    src/GLTFLoader.cpp
    src/stb_image_impl.cpp
)

add_executable(VulkanRenderer ${SOURCES})

target_include_directories(VulkanRenderer PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Vulkan_INCLUDE_DIRS}
    ${TINYGLTF_INCLUDE_DIRS}
)

target_link_libraries(VulkanRenderer PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui::imgui
)

# ── Copy shaders dir next to executable ───────────────────────────────────────
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/shaders/spv
        $<TARGET_FILE_DIR:VulkanRenderer>/shaders/spv
    COMMENT "Copying compiled SPIR-V shaders"
)

# ── Copy STL assets next to executable ────────────────────────────────────────
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:VulkanRenderer>/assets
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/CITY.obj
        ${CMAKE_SOURCE_DIR}/CITY.mtl
        ${CMAKE_SOURCE_DIR}/cube.stl
        ${CMAKE_SOURCE_DIR}/EinsteinBustTongue.stl
        ${CMAKE_SOURCE_DIR}/pine_tree.stl
        ${CMAKE_SOURCE_DIR}/rect_200x200x500.stl
        ${CMAKE_SOURCE_DIR}/rect_200x200x500_hole.stl
        ${CMAKE_SOURCE_DIR}/sphere_r150.stl
        $<TARGET_FILE_DIR:VulkanRenderer>/assets/
    COMMENT "Copying STL assets"
)

# ── Copy maps directory next to executable ────────────────────────────────────
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/maps
        $<TARGET_FILE_DIR:VulkanRenderer>/maps
    COMMENT "Copying scene maps"
)

# ── Create replays directory next to executable ───────────────────────────────
add_custom_command(TARGET VulkanRenderer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:VulkanRenderer>/replays
    COMMENT "Creating replays directory"
)

# ── Preprocessor ──────────────────────────────────────────────────────────────
target_compile_definitions(VulkanRenderer PRIVATE
    $<$<CONFIG:Debug>:ENABLE_VALIDATION_LAYERS>
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE
)

# Suppress code-page warning on MSVC (source files are UTF-8)
if(MSVC)
    target_compile_options(VulkanRenderer PRIVATE /utf-8)
endif()
