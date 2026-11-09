#include "preprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
cv::Mat Letterbox(const cv::Mat& src, int width, int height, float& ratio, int& left, int& top) {
    if (src.empty() || src.type()!=CV_8UC3 || width<=0 || height<=0)
        throw std::invalid_argument("Expected nonempty BGR image and positive input size");
    ratio=std::min(float(width)/src.cols,float(height)/src.rows);
    int w=std::clamp(int(std::round(src.cols*ratio)),1,width);
    int h=std::clamp(int(std::round(src.rows*ratio)),1,height);
    left=(width-w)/2; top=(height-h)/2;
    cv::Mat resized,padded;
    cv::resize(src,resized,cv::Size(w,h),0,0,cv::INTER_LINEAR);
    cv::copyMakeBorder(resized,padded,top,height-h-top,left,width-w-left,
                      cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
    return padded;
}
// Reuse buffer capacity. Each channel plane is written contiguously.
void ToBlob(const cv::Mat& img, std::vector<float>& blob) {
    if (img.empty() || img.type()!=CV_8UC3) throw std::invalid_argument("Expected BGR image");
    size_t plane=size_t(img.rows)*img.cols;
    blob.resize(3*plane);
    constexpr float scale=1.0f/255.0f;
    cv::parallel_for_(cv::Range(0,img.rows),[&](const cv::Range& rows) {
        for(int y=rows.start;y<rows.end;++y) {
            const auto* src=img.ptr<cv::Vec3b>(y);
            float* r=blob.data()+size_t(y)*img.cols;
            float* g=r+plane; float* b=g+plane;
            for(int x=0;x<img.cols;++x) {
                r[x]=src[x][2]*scale; g[x]=src[x][1]*scale; b[x]=src[x][0]*scale;
            }
        }
    });
}
