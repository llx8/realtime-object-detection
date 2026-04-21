#pragma once

#include "pipeline/types.h"
#include "pipeline/utils/SPSCQueue.h"
#include "pipeline/utils/FramePool.h"

#include <atomic>
#include <memory>
#include <thread>

namespace yolo_pipeline {

class LinuxV4L2Source;

class CaptureStage {
public:
    CaptureStage(LinuxV4L2Source* camera, FramePool* pool,
                 SPSCQueue<std::shared_ptr<Frame>>* output_queue);
    void start();
    void stop();
    void run();

private:
    LinuxV4L2Source* camera_;
    FramePool* pool_;
    SPSCQueue<std::shared_ptr<Frame>>* output_queue_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

} // namespace yolo_pipeline
