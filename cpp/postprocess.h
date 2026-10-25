#pragma once
#include <vector>

// Box：一个检测结果（此时还是 640 坐标系）
struct Box {
    float x1, y1, x2, y2;
    float score;
    int cls;
};

std::vector<Box> ParseOutput(const float* data);
float ComputeIoU(const Box& a, const Box& b);
std::vector<Box> NMS(std::vector<Box> boxes, float iouThreshold = 0.45f);