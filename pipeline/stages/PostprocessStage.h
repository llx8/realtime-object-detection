#pragma once

#include "pipeline/types.h"
#include "pipeline/utils/SPSCQueue.h"

#include <ncnn/mat.h>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

namespace yolo_pipeline {

class PostprocessStage {
public:
    PostprocessStage(SPSCQueue<InferenceFrame>* input_queue,
                     float conf_threshold, float iou_threshold,
                     int input_size);
    void start();
    void stop();
    void setCallback(DetectionCallback cb);
    void run();

private:
    struct TrackedDet {
        DetectionResult raw;
        DetectionResult smoothed;  // EMA平滑后的bbox
        int missed = 0;
    };

    std::vector<DetectionResult> decode(const ncnn::Mat& output, const Frame& frame);
    std::vector<DetectionResult> smooth(const std::vector<DetectionResult>& detections);

    static float iou(const DetectionResult& a, const DetectionResult& b);
    static std::vector<DetectionResult> nms(std::vector<DetectionResult>& dets, float iou_thresh);

    SPSCQueue<InferenceFrame>* input_queue_;
    float conf_threshold_;
    float iou_threshold_;
    int input_size_;
    DetectionCallback callback_;
    std::atomic<bool> running_{false};
    std::thread thread_;

    std::vector<TrackedDet> tracked_;
    static constexpr float kEmaAlpha = 0.25f;
    static constexpr float kMatchIou = 0.3f;
    static constexpr int kMaxMisses = 6;
};

} // namespace yolo_pipeline
