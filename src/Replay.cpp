#include "Replay.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdio>

#ifdef _WIN32
#  include <windows.h>
#  include <direct.h>
#else
#  include <dirent.h>
#  include <sys/stat.h>
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  saveReplay  –  녹화된 프레임 배열을 바이너리 파일로 저장
//  헤더: "VRPL" + 버전(1) + 프레임 수
//  본문: 각 프레임마다 [time, x, y, z, yaw, pitch] (모두 float32)
// ─────────────────────────────────────────────────────────────────────────────
bool saveReplay(const std::string& path, const std::vector<ReplayFrame>& frames) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Replay] Cannot write: " << path << "\n";
        return false;
    }

    // 파일 헤더 기록
    f.write("VRPL", 4);
    uint32_t ver = 1, count = (uint32_t)frames.size();
    f.write(reinterpret_cast<const char*>(&ver),   4);
    f.write(reinterpret_cast<const char*>(&count), 4);

    // 각 프레임의 카메라 상태를 순서대로 기록
    for (auto& fr : frames) {
        f.write(reinterpret_cast<const char*>(&fr.time),  4); // 경과 시간
        f.write(reinterpret_cast<const char*>(&fr.pos.x), 4); // X 위치
        f.write(reinterpret_cast<const char*>(&fr.pos.y), 4); // Y 위치
        f.write(reinterpret_cast<const char*>(&fr.pos.z), 4); // Z 위치
        f.write(reinterpret_cast<const char*>(&fr.yaw),   4); // 수평 회전
        f.write(reinterpret_cast<const char*>(&fr.pitch), 4); // 수직 회전
    }
    std::cout << "[Replay] Saved " << count << " frames to " << path << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  loadReplay  –  바이너리 파일에서 프레임 배열을 읽어온다
//  magic 검증 → 버전/카운트 읽기 → 프레임 배열 채우기
// ─────────────────────────────────────────────────────────────────────────────
bool loadReplay(const std::string& path, std::vector<ReplayFrame>& frames) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "[Replay] Cannot open: " << path << "\n";
        return false;
    }

    // magic 검증: 파일이 올바른 리플레이 포맷인지 확인
    char magic[4];
    f.read(magic, 4);
    if (magic[0]!='V'||magic[1]!='R'||magic[2]!='P'||magic[3]!='L') {
        std::cerr << "[Replay] Bad magic in " << path << "\n";
        return false;
    }

    // 버전·프레임 수 읽기 (버전은 현재 사용하지 않으나 하위 호환용으로 보존)
    uint32_t ver, count;
    f.read(reinterpret_cast<char*>(&ver),   4);
    f.read(reinterpret_cast<char*>(&count), 4);

    // 프레임 배열 크기 확보 후 데이터 채우기
    frames.resize(count);
    for (auto& fr : frames) {
        f.read(reinterpret_cast<char*>(&fr.time),  4);
        f.read(reinterpret_cast<char*>(&fr.pos.x), 4);
        f.read(reinterpret_cast<char*>(&fr.pos.y), 4);
        f.read(reinterpret_cast<char*>(&fr.pos.z), 4);
        f.read(reinterpret_cast<char*>(&fr.yaw),   4);
        f.read(reinterpret_cast<char*>(&fr.pitch), 4);
    }
    std::cout << "[Replay] Loaded " << count << " frames from " << path << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  findReplayFiles  –  dir 안의 *.replay 파일 목록을 이름순 정렬 후 반환
//  Windows: Win32 FindFirstFile API 사용
//  Linux/macOS: POSIX opendir/readdir 사용
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> findReplayFiles(const std::string& dir) {
    std::vector<std::string> result;
#ifdef _WIN32
    WIN32_FIND_DATAA fd;
    std::string pattern = dir + "\\*.replay";
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            // 디렉토리는 제외, 파일만 추가
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                result.push_back(dir + "/" + fd.cFileName);
        } while (FindNextFileA(h, &fd));
        FindClose(h);
    }
#else
    DIR* d = opendir(dir.c_str());
    if (d) {
        struct dirent* e;
        while ((e = readdir(d))) {
            std::string n = e->d_name;
            if (n.size()>7 && n.substr(n.size()-7)==".replay")
                result.push_back(dir+"/"+n);
        }
        closedir(d);
    }
#endif
    std::sort(result.begin(), result.end()); // 이름순 정렬 → 번호순과 동일
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  nextReplayPath  –  dir 안의 기존 파일 수에 +1 한 번호로 경로 생성
//  예: 기존 replay_001, _002 가 있으면 → replay_003.replay 반환
//  dir 가 없으면 mkdir 으로 생성
// ─────────────────────────────────────────────────────────────────────────────
std::string nextReplayPath(const std::string& dir) {
    // 디렉토리 없으면 생성 (이미 있어도 무시됨)
#ifdef _WIN32
    _mkdir(dir.c_str());
#else
    mkdir(dir.c_str(), 0755);
#endif
    auto existing = findReplayFiles(dir);
    int  next = (int)existing.size() + 1; // 현재 파일 개수 + 1 = 다음 번호
    char buf[64];
    std::snprintf(buf, sizeof(buf), "/replay_%03d.replay", next);
    return dir + buf;
}
