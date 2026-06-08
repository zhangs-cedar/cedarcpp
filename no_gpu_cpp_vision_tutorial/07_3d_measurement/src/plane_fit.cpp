#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

namespace fs = std::filesystem;

struct Point3D { double x, y, z; };

static std::vector<Point3D> read_xyz(const fs::path& path) {
    std::ifstream is(path);
    std::vector<Point3D> pts;
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Point3D p{};
        if (ss >> p.x >> p.y >> p.z) pts.push_back(p);
    }
    return pts;
}

static bool solve_3x3(double A[3][3], double b[3], double x[3]) {
    for (int i = 0; i < 3; ++i) {
        int pivot = i;
        for (int r = i + 1; r < 3; ++r) {
            if (std::abs(A[r][i]) > std::abs(A[pivot][i])) pivot = r;
        }
        if (std::abs(A[pivot][i]) < 1e-12) return false;
        if (pivot != i) {
            for (int c = 0; c < 3; ++c) std::swap(A[i][c], A[pivot][c]);
            std::swap(b[i], b[pivot]);
        }
        const double div = A[i][i];
        for (int c = i; c < 3; ++c) A[i][c] /= div;
        b[i] /= div;
        for (int r = 0; r < 3; ++r) {
            if (r == i) continue;
            const double factor = A[r][i];
            for (int c = i; c < 3; ++c) A[r][c] -= factor * A[i][c];
            b[r] -= factor * b[i];
        }
    }
    x[0] = b[0]; x[1] = b[1]; x[2] = b[2];
    return true;
}

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const fs::path input = args.get("input", "data/pointcloud/plane.xyz");
    const fs::path output = args.get("output", "out/07_plane.json");
    ng::ensure_dir(output.parent_path());

    ng::Timer timer;
    const auto pts = read_xyz(input);
    if (pts.size() < 3) {
        std::cerr << "Need at least 3 points: " << input << "\n";
        return 2;
    }

    // Fit z = ax + by + c by least squares.
    double Sx=0, Sy=0, Sz=0, Sxx=0, Syy=0, Sxy=0, Sxz=0, Syz=0;
    const double n = static_cast<double>(pts.size());
    for (const auto& p : pts) {
        Sx += p.x; Sy += p.y; Sz += p.z;
        Sxx += p.x * p.x; Syy += p.y * p.y; Sxy += p.x * p.y;
        Sxz += p.x * p.z; Syz += p.y * p.z;
    }

    double A[3][3] = {{Sxx, Sxy, Sx}, {Sxy, Syy, Sy}, {Sx, Sy, n}};
    double b[3] = {Sxz, Syz, Sz};
    double x[3] = {0, 0, 0};
    if (!solve_3x3(A, b, x)) {
        std::cerr << "Plane fit failed. Points may be degenerated.\n";
        return 3;
    }
    const double a = x[0], bb = x[1], c = x[2];

    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (const auto& p : pts) {
        const double pred = a * p.x + bb * p.y + c;
        const double err = p.z - pred;
        max_abs = std::max(max_abs, std::abs(err));
        mean_abs += std::abs(err);
    }
    mean_abs /= n;

    const double cost_ms = timer.elapsed_ms();
    std::ofstream os(output);
    os << std::fixed << std::setprecision(6);
    os << "{\n";
    os << "  \"model\": \"z = a*x + b*y + c\",\n";
    os << "  \"a\": " << a << ",\n";
    os << "  \"b\": " << bb << ",\n";
    os << "  \"c\": " << c << ",\n";
    os << "  \"points\": " << pts.size() << ",\n";
    os << "  \"mean_abs_error\": " << mean_abs << ",\n";
    os << "  \"max_abs_error\": " << max_abs << ",\n";
    os << "  \"cost_ms\": " << cost_ms << "\n";
    os << "}\n";

    std::cout << "plane: z = " << a << "*x + " << bb << "*y + " << c << "\n";
    std::cout << "mean_abs_error=" << mean_abs << " max_abs_error=" << max_abs << " cost_ms=" << cost_ms << "\n";
    return 0;
}
