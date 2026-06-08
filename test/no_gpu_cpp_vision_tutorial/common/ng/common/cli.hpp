#pragma once
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace ng {

class Args {
public:
    explicit Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (key.rfind("--", 0) == 0) {
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                    values_[key.substr(2)] = argv[++i];
                } else {
                    values_[key.substr(2)] = "true";
                }
            } else {
                positionals_.push_back(key);
            }
        }
    }

    bool has(const std::string& key) const { return values_.find(key) != values_.end(); }

    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = values_.find(key);
        return it == values_.end() ? def : it->second;
    }

    int get_int(const std::string& key, int def = 0) const {
        auto s = get(key, "");
        if (s.empty()) return def;
        return std::atoi(s.c_str());
    }

    double get_double(const std::string& key, double def = 0.0) const {
        auto s = get(key, "");
        if (s.empty()) return def;
        return std::atof(s.c_str());
    }

private:
    std::map<std::string, std::string> values_;
    std::vector<std::string> positionals_;
};

inline void print_usage(const std::string& app, const std::string& usage) {
    std::cerr << "Usage: " << app << " " << usage << "\n";
}

} // namespace ng
