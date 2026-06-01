#include "pipeline/utils/FramePool.h"

#include <chrono>
#include <cstddef>

namespace yolo_pipeline {

FramePool::FramePool(int pool_size, int width, int height, int channels, int input_size)
    : pool_size_(pool_size),
      width_(width),
      height_(height),
      channels_(channels),
      input_size_(input_size),
      slots_(static_cast<size_t>(pool_size)),
      frames_(static_cast<size_t>(pool_size)),
      free_list_(static_cast<size_t>(pool_size))
{
    const int prep_count = input_size * input_size * 3;

    for (int i = 0; i < pool_size_; ++i) {
        auto pixel_size = static_cast<size_t>(width) * height * channels;
        auto prep_size  = static_cast<size_t>(prep_count) * sizeof(float);

        slots_[i].buffer.resize(pixel_size);
        slots_[i].prep_buffer.resize(prep_size);

        frames_[i].data             = slots_[i].buffer.data();
        frames_[i].width            = width;
        frames_[i].height           = height;
        frames_[i].channels         = channels;
        frames_[i].preprocessed_data = slots_[i].prep_buffer.data();
        frames_[i].preprocessed_w   = input_size;
        frames_[i].preprocessed_h   = input_size;
    }

    for (int i = 0; i < pool_size_ - 1; ++i) {
        free_list_[i].next = i + 1;
    }
    free_list_[pool_size_ - 1].next = -1;

    free_head_.store(0, std::memory_order_release);
}

std::shared_ptr<Frame> FramePool::acquire() {
    int head = free_head_.load(std::memory_order_acquire);
    while (head != -1) {
        int next = free_list_[head].next;
        if (free_head_.compare_exchange_weak(head, next,
                                             std::memory_order_acq_rel,
                                             std::memory_order_acquire)) {
            Frame& f = frames_[head];
            f.frame_id     = frame_counter_.fetch_add(1, std::memory_order_relaxed);
            f.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();

            return std::shared_ptr<Frame>(&f,
                [this, head](Frame*) { this->release(head); });
        }
    }
    return nullptr;
}

void FramePool::release(int index) {
    int head = free_head_.load(std::memory_order_acquire);
    do {
        free_list_[index].next = head;
    } while (!free_head_.compare_exchange_weak(head, index,
                                               std::memory_order_acq_rel,
                                               std::memory_order_acquire));
}

}  // namespace yolo_pipeline
