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

echo All shaders compiled successfully.
