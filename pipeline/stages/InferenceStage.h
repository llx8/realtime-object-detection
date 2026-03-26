#pragma once

#include "pipeline/types.h"
#include "pipeline/utils/SPSCQueue.h"

#include <ncnn/net.h>

#include <atomic>
#include <memory>
#include <thread>

namespace yolo_pipeline {

class InferenceStage {
public:
    InferenceStage(SPSCQueue<std::shared_ptr<Frame>>* input_queue,
                   SPSCQueue<InferenceFrame>* output_queue,
                   ncnn::Net* net, int input_size);
    void start();
    void stop();
    void run();

private:
    SPSCQueue<std::shared_ptr<Frame>>* input_queue_;
    SPSCQueue<InferenceFrame>* output_queue_;
    ncnn::Net* net_;
    int input_size_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace yolo_pipeline
