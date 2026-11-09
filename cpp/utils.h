#pragma once
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <vector>
#include "postprocess.h"
void DrawBoxes(cv::Mat& img,const std::vector<Box>& boxes,const std::vector<std::string>& names);
void WriteBaseline(const std::string& path,const std::vector<Box>& boxes);
std::vector<std::string> LoadClassNames(const std::string& path);
template<typename F> double TimeIt(F&& f) {
    auto start=std::chrono::steady_clock::now(); f();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
