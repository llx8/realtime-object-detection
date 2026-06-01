#pragma once

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace yolo_pipeline {

struct V4l2Buffer {
    uint32_t index;
    uint32_t bytesused;
    void* data;
};

class V4l2Exception : public std::runtime_error {
public:
    V4l2Exception(const std::string& device, const std::string& op)
        : std::runtime_error("V4L2 error on " + device + " during " + op) {}
};

class V4l2Camera {
public:
    explicit V4l2Camera(const std::string& device);
    ~V4l2Camera();

    V4l2Camera(const V4l2Camera&) = delete;
    V4l2Camera& operator=(const V4l2Camera&) = delete;

    void setFormat(uint32_t width, uint32_t height, uint32_t pixelformat);
    void requestBuffers(uint32_t count);
    void queueAllBuffers();
    void startStream();
    void stopStream();

    V4l2Buffer dequeueBuffer();
    void queueBuffer(const V4l2Buffer& buf);

    int fd() const { return fd_; }

private:
    void closeDevice();

    int fd_ = -1;
    bool streaming_ = false;
    std::vector<void*> buffers_;
    std::vector<size_t> buffer_lengths_; // munmap时需要真实长度
};

} // namespace yolo_pipeline