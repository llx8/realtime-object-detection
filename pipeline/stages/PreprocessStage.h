#pragma once

#include "pipeline/types.h"
#include "pipeline/utils/SPSCQueue.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace yolo_pipeline {

class PreprocessStage {
public:
    PreprocessStage(SPSCQueue<std::shared_ptr<Frame>>* input_queue,
                    SPSCQueue<std::shared_ptr<Frame>>* output_queue,
                    int input_size);
    void start();
    void stop();
    void run();

private:
    SPSCQueue<std::shared_ptr<Frame>>* input_queue_;
    SPSCQueue<std::shared_ptr<Frame>>* output_queue_;
    int input_size_;
    std::vector<uint8_t> resized_buf_;
    std::vector<uint8_t> padded_buf_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace yolo_pipeline
