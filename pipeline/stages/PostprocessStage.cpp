#include "pipeline/stages/PostprocessStage.h"
#include "pipeline/Logger.h"
#include <algorithm>
#include <cmath>

namespace yolo_pipeline {

PostprocessStage::PostprocessStage(SPSCQueue<InferenceFrame>* input_queue,
                                   float conf_threshold, float iou_threshold,
                                   int input_size)
    : input_queue_(input_queue), conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold), input_size_(input_size) {}

void PostprocessStage::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&PostprocessStage::run, this);
}

void PostprocessStage::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) {
        thread_.join();
    }
}

void PostprocessStage::setCallback(DetectionCallback cb) {
    callback_ = std::move(cb);
}

std::vector<DetectionResult> PostprocessStage::decode(const ncnn::Mat& output,
                                                      const Frame& frame) {
    const int cam_w = frame.width;
    const int cam_h = frame.height;
    const int rotation = frame.rotation;
    std::vector<DetectionResult> detections;
    const int num_candidates = output.w;
    const int num_classes = output.h - 4;

    for (int i = 0; i < num_candidates; ++i) {
        float max_score = -1.0f;
        int max_cls_id = -1;
        for (int c = 0; c < num_classes; ++c) {
            float s = output.row(c + 4)[i];
            if (s > max_score) { max_score = s; max_cls_id = c; }
        }
        if (max_score < conf_threshold_) continue;

        float cx_n = (output.row(0)[i] - frame.pad_x) / (frame.letterbox_scale * cam_w);
        float cy_n = (output.row(1)[i] - frame.pad_y) / (frame.letterbox_scale * cam_h);
        float w_n = output.row(2)[i] / (frame.letterbox_scale * cam_w);
        float h_n = output.row(3)[i] / (frame.letterbox_scale * cam_h);

        float x1n = std::max(0.0f, std::min(1.0f, cx_n - w_n * 0.5f));
        float y1n = std::max(0.0f, std::min(1.0f, cy_n - h_n * 0.5f));
        float x2n = std::max(0.0f, std::min(1.0f, cx_n + w_n * 0.5f));
        float y2n = std::max(0.0f, std::min(1.0f, cy_n + h_n * 0.5f));

        DetectionResult det;
        det.x = x1n; det.y = y1n;
        det.w = x2n - x1n; det.h = y2n - y1n;
        det.confidence = max_score;
        det.class_id = max_cls_id;
        det.class_name = (max_cls_id >= 0 && max_cls_id < NUM_COCO_CLASSES)
                         ? COCO_NAMES[max_cls_id] : "unknown";

        if (det.w > 0.95f || det.h > 0.95f) continue;
        if (det.w < 0.003f || det.h < 0.003f) continue;
        detections.push_back(det);
    }
    return detections;
}

float PostprocessStage::iou(const DetectionResult& a, const DetectionResult& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float inter_w = std::max(0.0f, x2 - x1);
    float inter_h = std::max(0.0f, y2 - y1);
    float inter_area = inter_w * inter_h;
    float union_area = a.w * a.h + b.w * b.h - inter_area;
    return (union_area <= 0.0f) ? 0.0f : inter_area / union_area;
}

std::vector<DetectionResult> PostprocessStage::nms(std::vector<DetectionResult>& dets,
                                                   float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  return a.confidence > b.confidence;
              });

    std::vector<DetectionResult> result;
    std::vector<bool> suppressed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (suppressed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue;
            if (iou(dets[i], dets[j]) > iou_thresh) suppressed[j] = true;
        }
    }
    return result;
}

std::vector<DetectionResult> PostprocessStage::smooth(
    const std::vector<DetectionResult>& detections) {
    for (auto& t : tracked_) t.missed++;

    std::vector<bool> track_matched(tracked_.size(), false);
    std::vector<bool> det_matched(detections.size(), false);

    for (size_t di = 0; di < detections.size(); ++di) {
        const auto& det = detections[di];
        float best_iou = kMatchIou;
        int best_idx = -1;
        for (size_t ti = 0; ti < tracked_.size(); ++ti) {
            if (track_matched[ti]) continue;
            if (tracked_[ti].raw.class_id != det.class_id) continue;
            float v = iou(tracked_[ti].raw, det);
            if (v > best_iou) { best_iou = v; best_idx = (int)ti; }
        }

        if (best_idx >= 0) {
            track_matched[best_idx] = true;
            tracked_[best_idx].raw = det;
            auto& s = tracked_[best_idx].smoothed;
            s.x = kEmaAlpha * det.x + (1.0f - kEmaAlpha) * s.x;
            s.y = kEmaAlpha * det.y + (1.0f - kEmaAlpha) * s.y;
            s.w = kEmaAlpha * det.w + (1.0f - kEmaAlpha) * s.w;
            s.h = kEmaAlpha * det.h + (1.0f - kEmaAlpha) * s.h;
            s.confidence = det.confidence;
            tracked_[best_idx].missed = 0;
            det_matched[di] = true;
        }
    }

    for (size_t di = 0; di < detections.size(); ++di) {
        if (!det_matched[di])
            tracked_.push_back({detections[di], detections[di], 0});
    }

    tracked_.erase(
        std::remove_if(tracked_.begin(), tracked_.end(),
                       [](const TrackedDet& t) { return t.missed > kMaxMisses; }),
        tracked_.end());

    std::vector<DetectionResult> result;
    for (const auto& t : tracked_) {
        if (t.missed == 0) result.push_back(t.smoothed);
    }
    return result;
}

void PostprocessStage::run() {
    while (running_.load(std::memory_order_acquire)) {
        InferenceFrame inf_frame;
        if (!input_queue_->try_pop(inf_frame)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        std::vector<DetectionResult> detections = decode(inf_frame.output, *inf_frame.frame);

        if ((int)detections.size() > MAX_PRE_NMS) {
            std::partial_sort(detections.begin(), detections.begin() + MAX_PRE_NMS, detections.end(),
                              [](const DetectionResult& a, const DetectionResult& b) {
                                  return a.confidence > b.confidence;
                              });
            detections.resize(MAX_PRE_NMS);
        }

        std::vector<DetectionResult> nms_results = nms(detections, iou_threshold_);

        nms_results.erase(
            std::remove_if(nms_results.begin(), nms_results.end(),
                           [](const DetectionResult& d) { return d.confidence < POST_NMS_CONF; }),
            nms_results.end());

        std::vector<DetectionResult> results = smooth(nms_results);

        if ((int)results.size() > MAX_OUTPUT_DETECTIONS) {
            std::partial_sort(results.begin(), results.begin() + MAX_OUTPUT_DETECTIONS, results.end(),
                              [](const DetectionResult& a, const DetectionResult& b) {
                                  return a.confidence > b.confidence;
                              });
            results.resize(MAX_OUTPUT_DETECTIONS);
        }

        if (callback_) callback_(results);
    }
}

} // namespace yolo_pipeline