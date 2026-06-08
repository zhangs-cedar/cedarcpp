#pragma once
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace ng {

inline std::vector<std::filesystem::path> list_files(
    const std::filesystem::path& dir,
    const std::vector<std::string>& exts) {
    std::vector<std::filesystem::path> files;
    if (!std::filesystem::exists(dir)) return files;

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (std::find(exts.begin(), exts.end(), ext) != exts.end()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

inline void ensure_dir(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
}

} // namespace ng
