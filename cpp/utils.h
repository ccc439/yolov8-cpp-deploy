#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include "postprocess.h"

void DrawBoxes(cv::Mat& img, const std::vector<Box>& boxes,
               const std::vector<std::string>& classNames,
               float ratio, int padLeft, int padTop);
void WriteBaseline(const std::string& path, const std::vector<Box>& boxes, float ratio, int padLeft, int padTop);
std::vector<std::string> LoadClassNames(const std::string& path);

// 计时工具：包住一段代码，返回执行耗时（毫秒）
template <typename F>
double TimeIt(F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}