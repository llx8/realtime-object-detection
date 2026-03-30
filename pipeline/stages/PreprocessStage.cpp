#include "pipeline/stages/PreprocessStage.h"
#include "pipeline/Logger.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <ncnn/mat.h>

namespace yolo_pipeline {

PreprocessStage::PreprocessStage(SPSCQueue<std::shared_ptr<Frame>>* input_queue,
                                 SPSCQueue<std::shared_ptr<Frame>>* output_queue,
                                 int input_size)
    : input_queue_(input_queue), output_queue_(output_queue),
      input_size_(input_size) {}

void PreprocessStage::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&PreprocessStage::run, this);
}

void PreprocessStage::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void PreprocessStage::run() {
    while (running_.load(std::memory_order_acquire)) {
        std::shared_ptr<Frame> frame;
        if (!input_queue_->try_pop(frame)) {
           
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        const int iw = frame->width;
        const int ih = frame->height;
        const float scale = std::min((float)input_size_ / iw, (float)input_size_ / ih);
        const int new_w = static_cast<int>(iw * scale);
        const int new_h = static_cast<int>(ih * scale);
        const int pad_x = (input_size_ - new_w) / 2;
        const int pad_y = (input_size_ - new_h) / 2;

        const size_t resized_sz = static_cast<size_t>(new_w) * new_h * 3;
        if (resized_buf_.size() < resized_sz) resized_buf_.resize(resized_sz);
        ncnn::resize_bilinear_c3(frame->data, iw, ih, resized_buf_.data(), new_w, new_h);

        const size_t padded_sz = static_cast<size_t>(input_size_) * input_size_ * 3;
        if (padded_buf_.size() < padded_sz) padded_buf_.resize(padded_sz);
        std::fill(padded_buf_.begin(), padded_buf_.end(), static_cast<uint8_t>(114));
        for (int row = 0; row < new_h; ++row) {
            const uint8_t* src_row = resized_buf_.data() + row * new_w * 3;
            uint8_t* dst_row = padded_buf_.data() + ((row + pad_y) * input_size_ + pad_x) * 3;
            std::memcpy(dst_row, src_row, static_cast<size_t>(new_w) * 3);
        }

        frame->letterbox_scale = scale;
        frame->pad_x = pad_x;
        frame->pad_y = pad_y;

        float* dst = reinterpret_cast<float*>(frame->preprocessed_data);
        const uint8_t* src_u8 = padded_buf_.data();
        const int hw = input_size_ * input_size_;

        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < hw; ++i) {
                dst[c * hw + i] = src_u8[i * 3 + c] / 255.0f;
            }
        }

        static int pp_count = 0;
        if (pp_count < 3) {
            LOGI("frame#%d: input=%dx%d scale=%.3f pad=(%d,%d)",
                 pp_count, frame->width, frame->height, scale, pad_x, pad_y);
        }
        ++pp_count;

        frame->preprocessed_w = input_size_;
        frame->preprocessed_h = input_size_;

        while (running_.load(std::memory_order_acquire)) {
            if (output_queue_->try_push(frame)) {
                break;
            }
           
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
}

} // namespace yolo_pipeline