#include "postprocess.h"
#include <algorithm>

// 后处理：解析输出 + 置信度过滤
std::vector<Box> ParseOutput(const float* data) {
    constexpr int kCandidates = 8400;
    constexpr int kNumClasses = 80;
    const float CONF_THRESHOLD = 0.25f;   // 必须和 Python 端一致

    std::vector<Box> boxes;
    for (int i = 0; i < kCandidates; i++) {
        float best = 0.0f; int bestCls = -1;
        for (int c = 0; c < kNumClasses; c++) {
            // 布局：第 i 个候选框的第 k 个属性在 data[k * 8400 + i]（不是 i*84+k）
            float s = data[(4 + c) * kCandidates + i];
            if (s > best) { best = s; bestCls = c; }
        }
        if (best < CONF_THRESHOLD) continue;
        float cx = data[0 * kCandidates + i];
        float cy = data[1 * kCandidates + i];
        float w  = data[2 * kCandidates + i];
        float h  = data[3 * kCandidates + i];
        boxes.push_back({cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, best, bestCls});
    }
    return boxes;
}

// 后处理：IoU + NMS
float ComputeIoU(const Box& a, const Box& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    float areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter);
}

// 按得分降序逐个保留，删掉与保留框 IoU 超过阈值的
std::vector<Box> NMS(std::vector<Box> boxes, float iouThreshold) {
    std::sort(boxes.begin(), boxes.end(),
              [](const Box& a, const Box& b) { return a.score > b.score; });
    std::vector<Box> keep;
    while (!boxes.empty()) {
        keep.push_back(boxes[0]);
        std::vector<Box> rest;
        for (size_t i = 1; i < boxes.size(); i++)
            if (ComputeIoU(boxes[0], boxes[i]) < iouThreshold)
                rest.push_back(boxes[i]);
        boxes = std::move(rest);
    }
    return keep;
}