#include "pipeline/Pipeline.h"
#include "pipeline/hardware/LinuxV4L2Source.h"
#include "pipeline/Logger.h"
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>

std::atomic<bool> g_running{true};

void signal_handler(int signum) {
    LOGI("Received signal %d, shutting down...", signum);
    g_running = false;
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    yolo_pipeline::PipelineConfig config;
    config.model_param_path = "model/yolov8n_int8.param";
    config.model_bin_path = "model/yolov8n_int8.bin";
    config.camera_width = 640;
    config.camera_height = 480;
    config.camera_fps = 30;
    config.input_size = 640;

    try {
        yolo_pipeline::Pipeline pipeline(config);

        auto camera = std::make_unique<yolo_pipeline::LinuxV4L2Source>("/dev/video0");
        if (!camera->open(config.camera_width, config.camera_height, config.camera_fps)) {
            LOGE("Failed to open /dev/video0.");
            return -1;
        }
        pipeline.setCamera(std::move(camera));

        pipeline.setCallback([](const std::vector<yolo_pipeline::DetectionResult>& results) {
            if (results.empty()) return;
            LOGI("==== Detected %zu objects ====", results.size());
            for (const auto& res : results) {
                LOGI("[%s] conf: %.2f, box: (%.2f, %.2f, %.2f, %.2f)",
                     res.class_name.c_str(), res.confidence, res.x, res.y, res.w, res.h);
            }
        });

        LOGI("Starting YOLO Pipeline Daemon...");
        pipeline.start();

        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        pipeline.stop();
        LOGI("Pipeline stopped safely.");

    } catch (const std::exception& e) {
        LOGE("Fatal error: %s", e.what());
        return -1;
    }

    return 0;
}
