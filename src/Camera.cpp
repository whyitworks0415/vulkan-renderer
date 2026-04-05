#include "Camera.h"
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

// pitch 를 이 각도로 클램프해 gimbal lock 방지
static constexpr float kPitchLimit = 89.0f;

void Camera::clampPitch() {
    pitch = std::clamp(pitch, -kPitchLimit, kPitchLimit);
}

// ─────────────────────────────────────────────────────────────────────────────
//  getCameraFront
//  yaw·pitch 로부터 단위 전방 벡터를 계산한다.
//  구면 좌표 변환:
//    x = cos(pitch) * cos(yaw)
//    y = sin(pitch)             ← 수직 성분
//    z = cos(pitch) * sin(yaw)
// ─────────────────────────────────────────────────────────────────────────────
glm::vec3 Camera::getCameraFront() const {
    float yawR   = glm::radians(yaw);
    float pitchR = glm::radians(pitch);
    return glm::normalize(glm::vec3{
        std::cos(pitchR) * std::cos(yawR),
        std::sin(pitchR),
        std::cos(pitchR) * std::sin(yawR)
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  getHorizontalForward
//  pitch 를 0으로 고정한 수평 전방 벡터.
//  WASD 이동에 사용 → 위아래를 바라보고 있어도 수평으로만 이동
// ─────────────────────────────────────────────────────────────────────────────
glm::vec3 Camera::getHorizontalForward() const {
    float yawR = glm::radians(yaw);
    return glm::normalize(glm::vec3{std::cos(yawR), 0.0f, std::sin(yawR)});
}

// 수평 전방과 월드 업(Y) 의 외적으로 오른쪽 벡터를 구한다
glm::vec3 Camera::getRight() const {
    return glm::normalize(glm::cross(getHorizontalForward(), glm::vec3{0, 1, 0}));
}

// glm::lookAt 으로 뷰 행렬 반환 (position → position+front, up=Y)
glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(position, position + getCameraFront(), glm::vec3{0, 1, 0});
}

// ─────────────────────────────────────────────────────────────────────────────
//  processKeyboard  –  WASD + Space/Shift 이동
//  왼쪽 Ctrl 키를 누르면 fastSpeed 로 전환 (탐색 시 빠르게 이동)
// ─────────────────────────────────────────────────────────────────────────────
void Camera::processKeyboard(GLFWwindow* window, float dt) {
    bool  fast  = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS;
    float speed = fast ? fastSpeed : normalSpeed;
    float dist  = speed * dt; // 이번 프레임 이동 거리 (m)

    glm::vec3 fwd   = getHorizontalForward();
    glm::vec3 right = getRight();

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) position += fwd   * dist; // 앞
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) position -= fwd   * dist; // 뒤
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) position -= right * dist; // 왼
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) position += right * dist; // 오른
    if (glfwGetKey(window, GLFW_KEY_SPACE)      == GLFW_PRESS) position.y += dist; // 위
    if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) position.y -= dist; // 아래
}

// ─────────────────────────────────────────────────────────────────────────────
//  processArrowKeys  –  방향키로 배치 카메라(placed camera) 이동
//  ghost 모드에서 원래 카메라 위치를 조정할 때 사용
// ─────────────────────────────────────────────────────────────────────────────
void Camera::processArrowKeys(GLFWwindow* window, float dt) {
    float dist  = normalSpeed * dt;
    glm::vec3 fwd   = getHorizontalForward();
    glm::vec3 right = getRight();

    if (glfwGetKey(window, GLFW_KEY_UP)    == GLFW_PRESS) position += fwd   * dist;
    if (glfwGetKey(window, GLFW_KEY_DOWN)  == GLFW_PRESS) position -= fwd   * dist;
    if (glfwGetKey(window, GLFW_KEY_LEFT)  == GLFW_PRESS) position -= right * dist;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) position += right * dist;
    // Page Up / Page Down: 배치 카메라 수직 이동
    if (glfwGetKey(window, GLFW_KEY_PAGE_UP)   == GLFW_PRESS) position.y += dist;
    if (glfwGetKey(window, GLFW_KEY_PAGE_DOWN)  == GLFW_PRESS) position.y -= dist;
}

// ─────────────────────────────────────────────────────────────────────────────
//  processMouseDelta  –  마우스 드래그 델타로 yaw·pitch 업데이트
//  dy 가 양수(마우스 아래로) → pitch 감소(위를 봄) : 반전하여 직관적 조작
// ─────────────────────────────────────────────────────────────────────────────
void Camera::processMouseDelta(float dx, float dy) {
    yaw   += dx * mouseSensitivity;
    pitch -= dy * mouseSensitivity; // dy 부호 반전: 마우스를 내리면 위를 봄
    clampPitch();
}
