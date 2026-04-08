@echo off
REM Compile GLSL shaders to SPIR-V
REM Requires glslc (from Vulkan SDK) to be on PATH
REM Run this script from the project root before building.

if not exist "shaders\spv" mkdir "shaders\spv"

echo Compiling scene.vert ...
glslc shaders/scene.vert -o shaders/spv/scene.vert.spv
if errorlevel 1 ( echo FAILED: scene.vert & exit /b 1 )

echo Compiling scene.frag ...
glslc shaders/scene.frag -o shaders/spv/scene.frag.spv
if errorlevel 1 ( echo FAILED: scene.frag & exit /b 1 )

echo Compiling scene_instanced.vert ...
glslc shaders/scene_instanced.vert -o shaders/spv/scene_instanced.vert.spv
if errorlevel 1 ( echo FAILED: scene_instanced.vert & exit /b 1 )

echo Compiling gizmo.vert ...
glslc shaders/gizmo.vert -o shaders/spv/gizmo.vert.spv
if errorlevel 1 ( echo FAILED: gizmo.vert & exit /b 1 )

echo Compiling gizmo.frag ...
glslc shaders/gizmo.frag -o shaders/spv/gizmo.frag.spv
if errorlevel 1 ( echo FAILED: gizmo.frag & exit /b 1 )

echo Compiling gbuffer.frag ...
glslc shaders/gbuffer.frag -o shaders/spv/gbuffer.frag.spv
if errorlevel 1 ( echo FAILED: gbuffer.frag & exit /b 1 )

echo Compiling deferred_light.vert ...
glslc shaders/deferred_light.vert -o shaders/spv/deferred_light.vert.spv
if errorlevel 1 ( echo FAILED: deferred_light.vert & exit /b 1 )

echo Compiling deferred_light.frag ...
glslc shaders/deferred_light.frag -o shaders/spv/deferred_light.frag.spv
if errorlevel 1 ( echo FAILED: deferred_light.frag & exit /b 1 )

echo All shaders compiled successfully.
