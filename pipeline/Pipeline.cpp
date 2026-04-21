#include "pipeline/Pipeline.h"
#include "pipeline/hardware/LinuxV4L2Source.h"
#include "pipeline/Logger.h"
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace yolo_pipeline {

Pipeline::Pipeline(const PipelineConfig& config)
    : config_(config),
      queue_capture_to_preprocess_(config.queue_capacity),
      queue_preprocess_to_inference_(config.queue_capacity),
      queue_inference_to_postprocess_(config.queue_capacity)
{
    int ret = net_.load_param(config_.model_param_path.c_str());
    if (ret != 0) {
        throw std::runtime_error("Failed to load model param: " + config_.model_param_path);
    }
    ret = net_.load_model(config_.model_bin_path.c_str());
    if (ret != 0) {
        throw std::runtime_error("Failed to load model bin: " + config_.model_bin_path);
    }
    net_.opt.num_threads = 1;
}

Pipeline::~Pipeline() {
    if (running_.load()) stop();
}

void Pipeline::setCamera(std::unique_ptr<LinuxV4L2Source> camera) {
    camera_ = std::move(camera);
}

void Pipeline::start() {
    if (!camera_ || !camera_->isOpen()) {
        throw std::runtime_error("Camera not set or not opened");
    }

    int frame_w = camera_->getWidth();
    int frame_h = camera_->getHeight();
    frame_pool_ = std::unique_ptr<FramePool>(
        new FramePool(config_.frame_pool_size, frame_w, frame_h, 3, config_.input_size));
    capture_stage_ = std::unique_ptr<CaptureStage>(
        new CaptureStage(camera_.get(), frame_pool_.get(), &queue_capture_to_preprocess_));

    preprocess_stage_ = std::unique_ptr<PreprocessStage>(
        new PreprocessStage(&queue_capture_to_preprocess_,
                            &queue_preprocess_to_inference_, config_.input_size));

    inference_stage_ = std::unique_ptr<InferenceStage>(
        new InferenceStage(&queue_preprocess_to_inference_,
                           &queue_inference_to_postprocess_, &net_, config_.input_size));

    postprocess_stage_ = std::unique_ptr<PostprocessStage>(
        new PostprocessStage(&queue_inference_to_postprocess_, config_.conf_threshold,
                             config_.iou_threshold, config_.input_size));

    if (pending_callback_) postprocess_stage_->setCallback(std::move(pending_callback_));

    running_.store(true);
    postprocess_stage_->start();
    inference_stage_->start();
    preprocess_stage_->start();
    capture_stage_->start();
}

void Pipeline::stop() {
    running_.store(false);
    capture_stage_->stop();
    preprocess_stage_->stop();
    inference_stage_->stop();
    postprocess_stage_->stop();
}

bool Pipeline::isRunning() const {
    return running_.load();
}

void Pipeline::setCallback(DetectionCallback callback) {
    if (postprocess_stage_) {
        postprocess_stage_->setCallback(std::move(callback));
    } else {
        pending_callback_ = std::move(callback);
    }
}

} // namespace yolo_pipeline