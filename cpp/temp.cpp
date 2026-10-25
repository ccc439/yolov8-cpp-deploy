// 前处理.h
#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat Letterbox(const cv::Mat& src, int size, float& ratio, int& padLeft, int& padTop);
std::vector<float> ToBlob(const cv::Mat& img);
// 前处理.cpp
// src 输入图；size 目标大小为size*size；ratio 缩放比；padLeft 左侧填充灰边宽度；padTop 上侧填充灰边宽度。
cv::Mat Letterbox(const cv::Mat& src, int size, float& ratio, int& padLeft, int& padTop){
    int w = src.cols;
    int h = src.rows;
    ratio = std::min((float)size / w, (float)size / h);
    int new_w = (int)w * ratio;
    int new_h = (int)h * ratio;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));
    padLeft = (size - new_w) / 2;
    padTop = (size - new_h) / 2;
    // 填充灰边
    cv::Mat out;
    cv::copyMakeBorder(resized, out, padTop, size - new_h - padTop, padLeft, size - new_w - padLeft, 
        cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    return out;
}
// ToBlob：把 Mat 转成模型输入格式（RGB / CHW / 归一化）
// img 输入图（Letterbox 处理完的 640×640 图）
// 返回：3×640×640 个 float 的一维数组（CHW 布局），可直接包成模型输入张量
std::vector<float> ToBlob(const cv::Mat& img){
    int c = img.channels();
    int h = img.rows;
    int w = img.cols;
    std::vector<float> blob(c * h * w);
    // 遍历cv::Mat
    for(int i = 0;i < h;i++){
        for(int j = 0;j < w;j++){
            // 获取像素指针
            const uchar* p = img.ptr<uchar>(i) + j*c;
            // （RGB / CHW / 归一化）
            for(int k = 0;k < c;k++){
                blob[k * h * w + i * w + j] = p[c - k - 1] / 255.0f;
            }
            
        }
    }
    return blob;
}

// 后处理.h
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

// 后处理.cpp
std::vector<Box> ParseOutput(const float* data){
    int kCandidates = 8400;
    int kNumClasses = 80;
    float CONF_THRESHOLD = 0.25f;

    std::vector<Box> out;

    for(int i = 0;i < kCandidates; i++){
        // 找得分最高的类别
        int cls = -1;float score = 0.0f;
        for(int j = 0; j < kNumClasses;j++){
            float curScore = data[(j + 4) * kCandidates + i];
            if(curScore > score){
                score = curScore;
                cls = j;
            }
        }
        // 置信度阈值过滤
        if(score < CONF_THRESHOLD)continue;
        // 若有效，加入结果
        float cx = data[0 * kCandidates + i];
        float cy = data[1 * kCandidates + i];
        float w = data[2 * kCandidates + i];
        float h = data[3 * kCandidates + i];

        float x1 = cx - w/2;
        float y1 = cy - h/2;
        float x2 = cx + w/2;
        float y2 = cy + h/2;
        out.push_back({x1,y1,x2,y2,score,cls});
    }
    return out;
}
//计算IoU
float ComputeIoU(const Box& a, const Box& b){
    float x1 = std::max(a.x1,b.x1);
    float x2 = std::min(a.x2,b.x2);
    float y1 = std::max(a.y1,b.y1);
    float y2 = std::min(a.y2,b.y2);
    // 交集
    float i = std::max(0.0f,x2 - x1) * std::max(0.0f,y2 - y1);
    // 并集
    float u = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - i;
    return i / u;
}
// NMS
std::vector<Box> NMS(std::vector<Box> boxes, float iouThreshold = 0.45f){
    // 按置信度降序排序
    std::sort(boxes.begin(),boxes.end(),[](Box a,Box b){
        return a.score > b.score;
    });
    std::vector<Box> out; // 结果
    
    while(!boxes.empty()){
        std::vector<Box> temp; // 下一轮保留的
        Box cur = boxes[0];
        out.push_back(cur);
        for(int i = 1;i < boxes.size(); i++){ // 与cur过于重叠的全部舍弃
            if(ComputeIoU(cur,boxes[i]) < iouThreshold){
                // 符合要求，保留
                temp.push_back(boxes[i]);
            }
        }
        // 下一轮
        boxes = temp;
    }
    return out;
}