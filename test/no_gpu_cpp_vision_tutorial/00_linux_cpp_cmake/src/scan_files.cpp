#include <filesystem>
#include <iostream>
#include <vector>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const auto input = args.get("input", "data/images");

    ng::Timer timer;
    const auto files = ng::list_files(input, {".png", ".jpg", ".jpeg", ".bmp", ".txt", ".xyz"});

    std::cout << "input: " << input << "\n";
    std::cout << "files: " << files.size() << "\n";
    for (const auto& p : files) {
        std::cout << "  " << p.string() << "\n";
    }
    std::cout << "cost_ms: " << timer.elapsed_ms() << "\n";
    return files.empty() ? 2 : 0;
}
