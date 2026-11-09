#include "postprocess.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
std::vector<Box> ParseOutput(const float* data,size_t candidates,size_t classes,float conf) {
    if(!data || !candidates || !classes || !std::isfinite(conf) || conf<0 || conf>1)
        throw std::invalid_argument("Invalid output layout or confidence threshold");
    std::vector<Box> boxes;
    for(size_t i=0;i<candidates;++i) {
        float best=-1; int cls=-1;
        for(size_t c=0;c<classes;++c) {
            float s=data[(4+c)*candidates+i];
            if(s>best) { best=s; cls=int(c); }
        }
        if(cls<0 || !std::isfinite(best) || best<conf) continue;
        float cx=data[i],cy=data[candidates+i],w=data[2*candidates+i],h=data[3*candidates+i];
        if(!std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(w)||!std::isfinite(h)||w<=0||h<=0) continue;
        Box b{cx-w/2,cy-h/2,cx+w/2,cy+h/2,best,cls};
        if(std::isfinite(b.x1)&&std::isfinite(b.y1)&&std::isfinite(b.x2)&&std::isfinite(b.y2)) boxes.push_back(b);
    }
    return boxes;
}
float ComputeIoU(const Box& a,const Box& b) {
    float inter=std::max(0.0f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*
                std::max(0.0f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    float areaA=std::max(0.0f,a.x2-a.x1)*std::max(0.0f,a.y2-a.y1);
    float areaB=std::max(0.0f,b.x2-b.x1)*std::max(0.0f,b.y2-b.y1);
    float denom=areaA+areaB-inter;
    return denom>0 && std::isfinite(denom) ? inter/denom : 0.0f;
}
std::vector<Box> NMS(std::vector<Box> boxes,float threshold,bool agnostic) {
    if(!std::isfinite(threshold)||threshold<0||threshold>1) throw std::invalid_argument("Invalid IoU threshold");
    std::stable_sort(boxes.begin(),boxes.end(),[](const Box& a,const Box& b){return a.score>b.score;});
    std::vector<unsigned char> suppressed(boxes.size(),0);
    std::vector<Box> keep; keep.reserve(boxes.size());
    for(size_t i=0;i<boxes.size();++i) {
        if(suppressed[i]) continue;
        keep.push_back(boxes[i]);
        for(size_t j=i+1;j<boxes.size();++j)
            if(!suppressed[j] && (agnostic || boxes[i].cls==boxes[j].cls) &&
               ComputeIoU(boxes[i],boxes[j])>=threshold) suppressed[j]=1;
    }
    return keep;
}
std::vector<Box> RestoreBoxes(const std::vector<Box>& boxes,float ratio,int left,int top,int width,int height) {
    if(!std::isfinite(ratio)||ratio<=0||width<=0||height<=0) throw std::invalid_argument("Invalid transform");
    std::vector<Box> result; result.reserve(boxes.size());
    for(Box b:boxes) {
        b.x1=std::clamp((b.x1-left)/ratio,0.0f,float(width));
        b.x2=std::clamp((b.x2-left)/ratio,0.0f,float(width));
        b.y1=std::clamp((b.y1-top)/ratio,0.0f,float(height));
        b.y2=std::clamp((b.y2-top)/ratio,0.0f,float(height));
        if(b.x2>b.x1 && b.y2>b.y1) result.push_back(b);
    }
    return result;
}
