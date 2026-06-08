#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include "ng/common/cli.hpp"
#include "ng/common/fs.hpp"
#include "ng/common/timer.hpp"

namespace fs = std::filesystem;

template <typename T>
class BlockingQueue {
public:
    explicit BlockingQueue(size_t capacity) : capacity_(capacity) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) return false;
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    bool closed_ = false;
    size_t capacity_;
};

struct Frame {
    int id = 0;
    std::string path;
    std::vector<float> tensor;
    int class_id = -1;
    float score = 0.0f;
    double read_ms = 0.0;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    std::chrono::steady_clock::time_point start;
};

static void sleep_ms(int ms) {
    if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main(int argc, char** argv) {
    ng::Args args(argc, argv);
    const fs::path input_dir = args.get("input", "data/images");
    const fs::path output_dir = args.get("output", "out/05_pipeline");
    const int repeat = args.get_int("repeat", 100);
    const int queue_cap = args.get_int("queue-cap", 8);
    const int pre_ms = args.get_int("pre-ms", 2);
    const int infer_ms = args.get_int("infer-ms", 8);
    const int post_ms = args.get_int("post-ms", 2);

    ng::ensure_dir(output_dir);
    const auto files = ng::list_files(input_dir, {".png", ".jpg", ".jpeg", ".bmp"});
    if (files.empty()) {
        std::cerr << "No images found in " << input_dir << "\n";
        return 2;
    }

    BlockingQueue<Frame> q0(queue_cap), q1(queue_cap), q2(queue_cap), q3(queue_cap);
    std::atomic<int> written{0};
    const int total = repeat;

    std::ofstream csv(output_dir / "pipeline_metrics.csv");
    csv << "id,path,preprocess_ms,infer_ms,postprocess_ms,total_ms,class_id,score\n";
    std::mutex csv_mutex;

    ng::Timer global_timer;

    std::thread reader([&] {
        for (int i = 0; i < total; ++i) {
            ng::Timer timer;
            Frame f;
            f.id = i;
            f.path = files[static_cast<size_t>(i) % files.size()].string();
            f.start = std::chrono::steady_clock::now();
            sleep_ms(1);
            f.read_ms = timer.elapsed_ms();
            if (!q0.push(std::move(f))) break;
        }
        q0.close();
    });

    std::thread preprocessor([&] {
        Frame f;
        while (q0.pop(f)) {
            ng::Timer timer;
            sleep_ms(pre_ms);
            f.tensor.assign(3 * 64 * 64, static_cast<float>((f.id % 255) / 255.0));
            f.preprocess_ms = timer.elapsed_ms();
            if (!q1.push(std::move(f))) break;
        }
        q1.close();
    });

    std::thread infer([&] {
        Frame f;
        while (q1.pop(f)) {
            ng::Timer timer;
            sleep_ms(infer_ms);
            f.score = f.tensor.empty() ? 0.0f : f.tensor[0];
            f.class_id = f.score > 0.5f ? 1 : 0;
            f.infer_ms = timer.elapsed_ms();
            if (!q2.push(std::move(f))) break;
        }
        q2.close();
    });

    std::thread postprocessor([&] {
        Frame f;
        while (q2.pop(f)) {
            ng::Timer timer;
            sleep_ms(post_ms);
            f.postprocess_ms = timer.elapsed_ms();
            const auto end = std::chrono::steady_clock::now();
            f.total_ms = std::chrono::duration<double, std::milli>(end - f.start).count();
            if (!q3.push(std::move(f))) break;
        }
        q3.close();
    });

    std::thread writer([&] {
        Frame f;
        while (q3.pop(f)) {
            {
                std::lock_guard<std::mutex> lock(csv_mutex);
                csv << f.id << ',' << f.path << ',' << f.preprocess_ms << ',' << f.infer_ms << ','
                    << f.postprocess_ms << ',' << f.total_ms << ',' << f.class_id << ',' << f.score << '\n';
            }
            ++written;
        }
    });

    reader.join();
    preprocessor.join();
    infer.join();
    postprocessor.join();
    writer.join();

    const double sec = global_timer.elapsed_ms() / 1000.0;
    const double fps = sec > 0 ? written.load() / sec : 0.0;
    std::cout << "processed=" << written.load() << " seconds=" << sec << " throughput_fps=" << fps << "\n";
    return written == total ? 0 : 4;
}
