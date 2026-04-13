#include "pipeline/hardware/LinuxV4L2Source.h"
#include "pipeline/Logger.h"
#include <linux/videodev2.h>
#include <algorithm>

namespace yolo_pipeline {

LinuxV4L2Source::LinuxV4L2Source(const std::string& dev_name) 
    : device_name_(dev_name) {}

LinuxV4L2Source::~LinuxV4L2Source() {
    close();
}

bool LinuxV4L2Source::open(int width, int height, int fps) {
    try {
        camera_ = std::make_unique<V4l2Camera>(device_name_);
        // YUYV大部分摄像头都支持，也好转RGB
        camera_->setFormat(width, height, V4L2_PIX_FMT_YUYV);
        camera_->requestBuffers(4);
        camera_->queueAllBuffers();
        camera_->startStream();
        
        width_ = width;
        height_ = height;
        LOGI("LinuxV4L2Source opened successfully: %dx%d", width, height);
        return true;
    } catch (const std::exception& e) {
        LOGE("LinuxV4L2Source open failed: %s", e.what());
        camera_.reset();
        return false;
    }
}

bool LinuxV4L2Source::read(Frame* frame) {
    if (!camera_) return false;

    try {
        V4l2Buffer buf = camera_->dequeueBuffer();

        const uint8_t* yuyv = static_cast<const uint8_t*>(buf.data);
        uint8_t* rgb = frame->data;
        int pixels = width_ * height_;
        
        for (int i = 0, j = 0; i < pixels * 2; i += 4, j += 6) {
            int y0 = yuyv[i + 0];
            int u  = yuyv[i + 1] - 128;
            int y1 = yuyv[i + 2];
            int v  = yuyv[i + 3] - 128;

            int ruv = 409 * v + 128;
            int guv = -100 * u - 208 * v + 128;
            int buv = 516 * u + 128;

            int y00 = 298 * (y0 - 16);
            rgb[j + 0] = std::clamp((y00 + ruv) >> 8, 0, 255);
            rgb[j + 1] = std::clamp((y00 + guv) >> 8, 0, 255);
            rgb[j + 2] = std::clamp((y00 + buv) >> 8, 0, 255);

            int y11 = 298 * (y1 - 16);
            rgb[j + 3] = std::clamp((y11 + ruv) >> 8, 0, 255);
            rgb[j + 4] = std::clamp((y11 + guv) >> 8, 0, 255);
            rgb[j + 5] = std::clamp((y11 + buv) >> 8, 0, 255);
        }

        frame->format = PixelFormat::RGB;
        camera_->queueBuffer(buf);
        return true;
    } catch (...) {
        return false;
    }
}

void LinuxV4L2Source::close() {
    camera_.reset(); 
}

} // namespace yolo_pipeline