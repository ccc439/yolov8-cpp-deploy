#include "preprocess.h"
#include <cmath>

// 前处理：Letterbox（缩放 + 灰边填充）
// ratio 和 padding 要保存下来，后处理还原坐标用
cv::Mat Letterbox(const cv::Mat& src, int size, float& ratio, int& padLeft, int& padTop) {
    int h = src.rows, w = src.cols;
    ratio = std::min((float)size / h, (float)size / w);
    int newW = (int)std::round(w * ratio);
    int newH = (int)std::round(h * ratio);
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
    padLeft = (size - newW) / 2;
    padTop  = (size - newH) / 2;
    cv::Mat out;
    cv::copyMakeBorder(resized, out, padTop, size - newH - padTop,
                       padLeft, size - newW - padLeft,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

// 前处理：Mat 转 blob（RGB / CHW / 归一化）
// BGR 倒序变 RGB、HWC 变 CHW、除以 255 归一化（忘了除会全错）
std::vector<float> ToBlob(const cv::Mat& img) {
    int c = img.channels(), h = img.rows, w = img.cols;
    std::vector<float> blob((size_t)c * h * w);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            const uchar* p = img.ptr<uchar>(y) + x * c;
            for (int k = 0; k < c; k++)
                blob[k * h * w + y * w + x] = p[c - 1 - k] / 255.0f;
        }
    }
    return blob;
}