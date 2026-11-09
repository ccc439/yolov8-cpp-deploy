#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
cv::Mat Letterbox(const cv::Mat& src, int width, int height, float& ratio, int& left, int& top);
void ToBlob(const cv::Mat& img, std::vector<float>& blob);
