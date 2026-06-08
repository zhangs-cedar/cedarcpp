#include <iostream>
#include "infer_engine.h"
#include "ng/common/cli.hpp"

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const auto backend = args.get("backend", "cpu");
    auto engine = create_engine(backend);
    auto input = make_dummy_input(3, 64, 64);
    auto result = engine->infer(input);

    std::cout << "ok=" << (result.ok ? "true" : "false") << "\n";
    std::cout << "backend=" << result.backend << "\n";
    std::cout << "message=" << result.message << "\n";
    std::cout << "cost_ms=" << result.cost_ms << "\n";
    std::cout << "output_shape=";
    for (auto v : result.output.shape) std::cout << v << ' ';
    std::cout << "\noutput=";
    for (auto v : result.output.data) std::cout << v << ' ';
    std::cout << "\n";
    return result.ok ? 0 : 3;
}
