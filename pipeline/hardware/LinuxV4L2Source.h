#pragma once
#include "pipeline/types.h"
#include "v4l2_camera.hpp"
#include <memory>
#include <string>

namespace yolo_pipeline {

class LinuxV4L2Source {
public:
    explicit LinuxV4L2Source(const std::string& dev_name = "/dev/video0");
    ~LinuxV4L2Source();

    bool open(int width, int height, int fps);
    bool read(Frame* frame);
    void close();

    int getWidth() const { return width_; }
    int getHeight() const { return height_; }
    bool isOpen() const { return camera_ != nullptr; }

private:
    std::string device_name_;
    std::unique_ptr<V4l2Camera> camera_;
    int width_ = 0;
    int height_ = 0;
};

} // namespace yolo_pipeline