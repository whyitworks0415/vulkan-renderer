#pragma once
#include <vector>
#include <string>
#include <glm/glm.hpp>

// Replay: 카메라 경로 녹화와 재생
// 녹화: 매 프레임 카메라 위치·방향을 ReplayFrame 으로 저장
// 재생: 저장된 프레임을 시간순으로 보간하며 카메라에 적용

// 녹화된 카메라 스냅샷 한 장
struct ReplayFrame {
    float     time; // 녹화 시작으로부터의 경과 시간 (초)
    glm::vec3 pos; // 카메라 월드 위치
    float     yaw; // 수평 회전각 (도)
    float     pitch; // 수직 회전각 (도)
};

// 리플레이 바이너리 파일 포맷:
// magic   : char[4]    = "VRPL"          (파일 식별자)
// version : uint32     = 1               (버전)
// count   : uint32     = 프레임 수
// frames  : [time(f32), x,y,z(f32×3), yaw(f32), pitch(f32)] × count 형식으로 저장

// frames 를 path 에 바이너리로 저장. 실패 시 false 반환
bool saveReplay(const std::string& path, const std::vector<ReplayFrame>& frames);

// path 에서 바이너리를 읽어 frames 에 채워 넣음. 실패 시 false 반환
bool loadReplay(const std::string& path, std::vector<ReplayFrame>& frames);

// dir 안의 *.replay 파일 목록을 이름순으로 반환
std::vector<std::string> findReplayFiles(const std::string& dir);

// dir/replay_001.replay, _002, ... 형식으로 다음 번호의 경로를 반환
// (dir 가 없으면 생성)
std::string nextReplayPath(const std::string& dir);
