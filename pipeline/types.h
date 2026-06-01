#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <ncnn/mat.h>

namespace yolo_pipeline {

enum class PixelFormat {
    RGB,
    BGR
};

struct Frame {
    uint8_t* data = nullptr;
    int width = 0;
    int height = 0;
    int channels = 3;
    PixelFormat format = PixelFormat::RGB;
    int64_t timestamp_us = 0;
    int64_t frame_id = 0;
    uint8_t* preprocessed_data = nullptr;
    int preprocessed_w = 0;
    int preprocessed_h = 0;
    int rotation = 0;
    float letterbox_scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
};

struct DetectionResult {
    float x, y, w, h;
    float confidence;
    int class_id;
    std::string class_name;
};

struct PipelineConfig {
    std::string model_param_path;
    std::string model_bin_path;
    int camera_width = 640;
    int camera_height = 480;
    int camera_fps = 30;
    int input_size = 640;
    float conf_threshold = 0.7f;
    float iou_threshold = 0.3f;
    int queue_capacity = 4;
    int frame_pool_size = 6;
};

using DetectionCallback = std::function<void(const std::vector<DetectionResult>&)>;

struct InferenceFrame {
    std::shared_ptr<Frame> frame;
    ncnn::Mat output;
};

constexpr int NUM_COCO_CLASSES = 80;

constexpr int MAX_PRE_NMS = 1000;       // 控制NMS前的候选数量
constexpr float POST_NMS_CONF = 0.7f;
constexpr int MAX_OUTPUT_DETECTIONS = 50;

// static是怕多个编译单元重复链接
static const char* COCO_NAMES[] = {
    "人", "自行车", "车", "摩托车", "飞机",
    "公交车", "火车", "卡车", "船", "红绿灯",
    "消防栓", "停车标志", "停车计时器", "长椅", "鸟",
    "猫", "狗", "马", "羊", "牛",
    "大象", "熊", "斑马", "长颈鹿", "背包",
    "雨伞", "手提包", "领带", "行李箱", "飞盘",
    "滑雪板", "单板", "球", "风筝", "棒球棒",
    "棒球手套", "滑板", "冲浪板", "网球拍", "瓶子",
    "酒杯", "杯子", "叉子", "刀", "勺子",
    "碗", "香蕉", "苹果", "三明治", "橙子",
    "西兰花", "胡萝卜", "热狗", "披萨", "甜甜圈",
    "蛋糕", "椅子", "沙发", "盆栽", "床",
    "餐桌", "马桶", "电视", "笔记本", "鼠标",
    "遥控器", "键盘", "手机", "微波炉", "烤箱",
    "烤面包机", "水槽", "冰箱", "书", "时钟",
    "花瓶", "剪刀", "泰迪熊", "吹风机", "牙刷"
};

}  // namespace yolo_pipeline
