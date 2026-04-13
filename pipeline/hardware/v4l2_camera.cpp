#include "v4l2_camera.hpp"
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <linux/videodev2.h>
#include <stdexcept>
#include <cstdio>

namespace yolo_pipeline {

V4l2Camera::V4l2Camera(const std::string& device) {
    fd_ = open(device.c_str(), O_RDWR);
    if (fd_ < 0) {
        throw V4l2Exception(device, "open");
    }

    struct v4l2_capability cap;
    std::memset(&cap, 0, sizeof(cap));

    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) != 0) {
        closeDevice();
        throw V4l2Exception(device, "VIDIOC_QUERYCAP");
    }

    if ((cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0) {
        closeDevice();
        throw V4l2Exception(device, "no VIDEO_CAPTURE capability");
    }

    if ((cap.capabilities & V4L2_CAP_STREAMING) == 0) {
        closeDevice();
        throw V4l2Exception(device, "no STREAMING capability");
    }
}

V4l2Camera::~V4l2Camera() {
    try {
        stopStream();
    } catch (...) {}

    
    for (size_t i = 0; i < buffers_.size(); ++i) {
        if (buffers_[i] != nullptr && buffers_[i] != MAP_FAILED) {
            munmap(buffers_[i], buffer_lengths_[i]); 
        }
    }

    closeDevice();
}

void V4l2Camera::closeDevice() {
    if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
    }
}

void V4l2Camera::enumFormats() {
    int fmt_index = 0;
    while (true) {
        struct v4l2_fmtdesc fmtdesc;
        std::memset(&fmtdesc, 0, sizeof(fmtdesc));
        fmtdesc.index = fmt_index;
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        if (ioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc) != 0) break;

        int frame_index = 0;
        while (true) {
            struct v4l2_frmsizeenum fsenum;
            std::memset(&fsenum, 0, sizeof(fsenum));
            fsenum.index = frame_index;
            fsenum.pixel_format = fmtdesc.pixelformat;

            if (ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &fsenum) != 0) break;

            if (fsenum.type == V4L2_FRMSIZE_TYPE_DISCRETE) {
                printf("format %s (0x%08X), framesize %u x %u\n",
                       fmtdesc.description, fmtdesc.pixelformat,
                       fsenum.discrete.width, fsenum.discrete.height);
            }
            ++frame_index;
        }
        ++fmt_index;
    }
}

void V4l2Camera::setFormat(uint32_t width, uint32_t height, uint32_t pixelformat) {
    struct v4l2_format fmt;
    std::memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_ANY;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) != 0) {
        throw V4l2Exception("device", "VIDIOC_S_FMT");
    }
    printf("set format ok: %ux%u\n", fmt.fmt.pix.width, fmt.fmt.pix.height);
}

void V4l2Camera::requestBuffers(uint32_t count) {
    struct v4l2_requestbuffers rb;
    std::memset(&rb, 0, sizeof(rb));
    rb.count = count;
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &rb) != 0) {
        throw V4l2Exception("device", "VIDIOC_REQBUFS");
    }

    printf("allocated %u buffers\n", rb.count);

    buffers_.reserve(rb.count);
    buffer_lengths_.reserve(rb.count);

    for (uint32_t i = 0; i < rb.count; ++i) {
        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) != 0) {
            throw V4l2Exception("device", "VIDIOC_QUERYBUF");
        }

        void* ptr = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd_, buf.m.offset);

        if (ptr == MAP_FAILED) {
            throw V4l2Exception("device", "mmap");
        }

        buffers_.push_back(ptr);
        buffer_lengths_.push_back(buf.length);
    }
}

void V4l2Camera::queueAllBuffers() {
    for (uint32_t i = 0; i < buffers_.size(); ++i) {
        struct v4l2_buffer buf;
        std::memset(&buf, 0, sizeof(buf));
        buf.index = i;
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) != 0) {
            throw V4l2Exception("device", "VIDIOC_QBUF");
        }
    }
    printf("queued %zu buffers ok\n", buffers_.size());
}

void V4l2Camera::startStream() {
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) != 0) {
        throw V4l2Exception("device", "VIDIOC_STREAMON");
    }
    streaming_ = true;
    printf("start capture ok\n");
}

void V4l2Camera::stopStream() {
    if (!streaming_) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type) != 0) {
        throw V4l2Exception("device", "VIDIOC_STREAMOFF");
    }
    streaming_ = false;
    printf("stop capture ok\n");
}

V4l2Buffer V4l2Camera::dequeueBuffer() {
    struct v4l2_buffer buf;
    std::memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_DQBUF, &buf) != 0) {
        throw V4l2Exception("device", "VIDIOC_DQBUF");
    }

    return V4l2Buffer{
        .index = buf.index,
        .bytesused = buf.bytesused,
        .data = buffers_.at(buf.index)
    };
}

void V4l2Camera::queueBuffer(const V4l2Buffer& buf) {
    struct v4l2_buffer vb;
    std::memset(&vb, 0, sizeof(vb));
    vb.index = buf.index;
    vb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vb.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_QBUF, &vb) != 0) {
        throw V4l2Exception("device", "VIDIOC_QBUF");
    }
}

} // namespace yolo_pipeline