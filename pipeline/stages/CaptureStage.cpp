#include "pipeline/stages/CaptureStage.h"
#include "pipeline/hardware/LinuxV4L2Source.h"
#include "pipeline/Logger.h"
#include <chrono>

namespace yolo_pipeline {

CaptureStage::CaptureStage(LinuxV4L2Source* camera, FramePool* pool,
                           SPSCQueue<std::shared_ptr<Frame>>* output_queue)
    : camera_(camera), pool_(pool), output_queue_(output_queue) {}

void CaptureStage::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&CaptureStage::run, this);
}

void CaptureStage::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void CaptureStage::run() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<Frame> frame = pool_->acquire();
        if (!frame) {
            
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        if (!camera_->read(frame.get())) {
            frame.reset();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        while (running_.load(std::memory_order_acquire)) {
            if (output_queue_->try_push(frame)) {
                break;
            }
        }
    }
}

} // namespace yolo_pipeline