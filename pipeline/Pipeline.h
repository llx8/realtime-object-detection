#pragma once
#include "pipeline/types.h"
#include "pipeline/utils/SPSCQueue.h"
#include "pipeline/utils/FramePool.h"
#include "pipeline/stages/CaptureStage.h"
#include "pipeline/stages/PreprocessStage.h"
#include "pipeline/stages/InferenceStage.h"
#include "pipeline/stages/PostprocessStage.h"
#include <ncnn/net.h>
#include <memory>
#include <atomic>

namespace yolo_pipeline {

class LinuxV4L2Source;

class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);
    ~Pipeline();

    void setCamera(std::unique_ptr<LinuxV4L2Source> camera);
    void start();
    void stop();
    bool isRunning() const;
    void setCallback(DetectionCallback callback);

private:
    PipelineConfig config_;
    std::unique_ptr<LinuxV4L2Source> camera_;
    ncnn::Net net_;

    std::unique_ptr<FramePool> frame_pool_;

    SPSCQueue<std::shared_ptr<Frame>> queue_capture_to_preprocess_;
    SPSCQueue<std::shared_ptr<Frame>> queue_preprocess_to_inference_;
    SPSCQueue<InferenceFrame> queue_inference_to_postprocess_;

    std::unique_ptr<CaptureStage> capture_stage_;
    std::unique_ptr<PreprocessStage> preprocess_stage_;
    std::unique_ptr<InferenceStage> inference_stage_;
    std::unique_ptr<PostprocessStage> postprocess_stage_;

    DetectionCallback pending_callback_;
    std::atomic<bool> running_{false};
};

} // namespace yolo_pipeline
