#include "VulkanApp.h"
#include <iostream>
#include <stdexcept>

// 프로그램 진입점.
// VulkanApp 객체를 생성하고 run()을 호출한다.
// 초기화~메인루프~정리까지 모든 흐름이 run() 내부에서 처리된다.
int main() {
    try {
        VulkanApp app;
        app.run();
    } catch (const std::exception& e) {
        // Vulkan 초기화 실패, 셰이더 파일 없음 등 치명적 오류를 콘솔에 출력
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
