#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace yolo_pipeline {

// 无锁环形队列，读写指针分开避免缓存冲突
template <typename T>
class SPSCQueue {
public:
    explicit SPSCQueue(size_t capacity)
        : capacity_(next_power_of_two(capacity)),
          mask_(capacity_ - 1),
          buffer_(new Slot[capacity_])
    {
        for (size_t i = 0; i < capacity_; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~SPSCQueue() {
        delete[] buffer_;
    }

    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;
    SPSCQueue(SPSCQueue&&) = delete;
    SPSCQueue& operator=(SPSCQueue&&) = delete;

    bool try_push(const T& item) {
        size_t pos = write_idx_.load(std::memory_order_relaxed);
        Slot& slot = buffer_[pos & mask_];
        size_t seq = slot.sequence.load(std::memory_order_acquire);
        if (seq != pos) {
            return false;  // full
        }
        slot.data = item;
        slot.sequence.store(pos + 1, std::memory_order_release);
        write_idx_.store(pos + 1, std::memory_order_relaxed);
        return true;
    }

    bool try_pop(T& item) {
        size_t pos = read_idx_.load(std::memory_order_relaxed);
        Slot& slot = buffer_[pos & mask_];
        size_t seq = slot.sequence.load(std::memory_order_acquire);
        if (seq != pos + 1) {
            return false;  // empty
        }
        item = std::move(slot.data);
        slot.sequence.store(pos + capacity_, std::memory_order_release);
        read_idx_.store(pos + 1, std::memory_order_relaxed);
        return true;
    }

    bool empty() const {
        size_t r = read_idx_.load(std::memory_order_relaxed);
        size_t w = write_idx_.load(std::memory_order_relaxed);
        return r == w;
    }

    bool full() const {
        size_t r = read_idx_.load(std::memory_order_relaxed);
        size_t w = write_idx_.load(std::memory_order_relaxed);
        return (w - r) == capacity_;
    }

    size_t size() const {
        size_t r = read_idx_.load(std::memory_order_relaxed);
        size_t w = write_idx_.load(std::memory_order_relaxed);
        return w - r;
    }

    size_t capacity() const {
        return capacity_;
    }

private:
    struct alignas(64) Slot {
        T data;
        std::atomic<size_t> sequence;
    };

    static size_t next_power_of_two(size_t v) {
        if (v == 0) return 1;
        --v;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        if (sizeof(size_t) > 4) {
            v |= v >> 32;
        }
        return v + 1;
    }

    const size_t capacity_;
    const size_t mask_;

    alignas(64) std::atomic<size_t> write_idx_{0};
    alignas(64) std::atomic<size_t> read_idx_{0};

    Slot* buffer_;
};

}  // namespace yolo_pipeline
