#pragma once

#include "pipeline/types.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace yolo_pipeline {

struct PoolSlot {
    std::vector<uint8_t> buffer;
    std::vector<uint8_t> prep_buffer;
};

struct FreeNode {
    int next;  // -1表示链尾
};

// 预分配的帧内存池，无锁free list，acquire/release可跨线程调用
class FramePool {
public:
    FramePool(int pool_size, int width, int height, int channels, int input_size);
    ~FramePool() = default;

    // 不能拷贝移动，因为shared_ptr的custom deleter捕获了this
    FramePool(const FramePool&) = delete;
    FramePool& operator=(const FramePool&) = delete;
    FramePool(FramePool&&) = delete;
    FramePool& operator=(FramePool&&) = delete;

    // 返回带custom deleter的shared_ptr，析构时自动归还池中
    std::shared_ptr<Frame> acquire();

    int available() const; // O(n)遍历，给诊断用的
    int frameWidth() const { return width_; }
    int frameHeight() const { return height_; }

private:
    void release(int index);

    const int pool_size_;
    const int width_;
    const int height_;
    const int channels_;
    const int input_size_;

    std::vector<PoolSlot> slots_;
    std::vector<Frame> frames_;
    std::vector<FreeNode> free_list_;

    alignas(64) std::atomic<int> free_head_{-1};
    std::atomic<int64_t> frame_counter_{0};
};

}  // namespace yolo_pipeline
