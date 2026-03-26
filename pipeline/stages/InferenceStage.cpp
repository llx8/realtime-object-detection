#include "pipeline/stages/InferenceStage.h"
#include "pipeline/Logger.h"
#include <chrono>
#include <cstdio>
#include <string>

namespace yolo_pipeline {

InferenceStage::InferenceStage(SPSCQueue<std::shared_ptr<Frame>>* input_queue,
                               SPSCQueue<InferenceFrame>* output_queue,
                               ncnn::Net* net, int input_size)
    : input_queue_(input_queue), output_queue_(output_queue),
      net_(net), input_size_(input_size) {}

void InferenceStage::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&InferenceStage::run, this);
}

void InferenceStage::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void InferenceStage::run() {
    int frame_count = 0;
    auto fps_start = std::chrono::steady_clock::now();

    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<Frame> frame;
        if (!input_queue_->try_pop(frame)) {
           
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        ncnn::Mat input(input_size_, input_size_, 3,
                        reinterpret_cast<float*>(frame->preprocessed_data));

        ncnn::Extractor ex = net_->create_extractor();
        ex.input("in0", input);

        ncnn::Mat output;
        int ret = ex.extract("out0", output);

        if (ret != 0) {
            LOGE("extract failed: %d", ret);
            continue;
        }

        ++frame_count;
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_start).count();
        if (elapsed >= 2.0) {
            LOGI("inference FPS: %.1f", frame_count / elapsed);
            frame_count = 0;
            fps_start = now;
        }

        InferenceFrame inf_frame;
        inf_frame.frame = frame;
        inf_frame.output = output;

        while (running_.load(std::memory_order_acquire)) {
            if (output_queue_->try_push(inf_frame)) {
                break;
            }
           
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

} // namespace yolo_pipeline