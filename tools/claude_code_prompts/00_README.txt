Claude Code Prompt Bundle

Project:
- Windows C++17 Vulkan renderer
- Main files: src/VulkanApp.cpp, src/VulkanApp.h, src/MeshLoader.cpp, src/SceneLoader.cpp, src/PerformanceStats.cpp
- Build system: CMake

Observed performance context:
- Map switching is synchronous and stalls the app.
- reloadScene() calls vkDeviceWaitIdle() and destroys/recreates scene buffers.
- copyBuffer() calls vkQueueWaitIdle() for each upload.
- Grid maps are expanded into many DrawObject entries.
- Frustum culling, LOD, instancing, and occlusion culling are not a good default profile right now.
- LOD path exists in code but real LOD data is effectively missing.
- Mesh loading duplicates vertices per triangle and ASCII STL parsing reads the full file into memory.
- Heavy map examples:
  - cube_grid: about 129 draw objects, about 8.7k triangles
  - city_blocks: about 113 draw objects, about 8.5k triangles
  - sphere_arena: about 193 draw objects, about 515k triangles
  - pine_forest: about 114 draw objects, about 941k triangles
  - EinsteinBustTongue.stl alone is about 885k triangles and about 44 MB on disk

Recommended execution order:
1. 01_baseline_and_profiling.txt
2. 02_scene_reload_and_upload_stalls.txt
3. 03_mesh_cache_and_reload_reuse.txt
4. 04_runtime_cpu_path_optimization.txt
5. 05_lod_support_and_scene_extensions.txt
6. 06_asset_analysis_and_heavy_mesh_policy.txt

General rules for each prompt:
- Work only on the requested part. Do not bundle unrelated refactors.
- Preserve existing behavior unless the prompt explicitly asks for a default-policy change.
- Prefer small, reviewable patches.
- Build with `cmake --build build --config Release` after changes if possible.
- If runtime verification is not possible, state exactly what was not verified.
- Do not revert unrelated local changes.

