#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat Letterbox(const cv::Mat& src, int size, float& ratio, int& padLeft, int& padTop);
std::vector<float> ToBlob(const cv::Mat& img);