#pragma once
#include <cstddef>
#include <vector>
struct Box { float x1,y1,x2,y2,score; int cls; };
std::vector<Box> ParseOutput(const float* data,size_t candidates,size_t classes,float conf);
float ComputeIoU(const Box& a,const Box& b);
std::vector<Box> NMS(std::vector<Box> boxes,float threshold=0.45f,bool agnostic=false);
std::vector<Box> RestoreBoxes(const std::vector<Box>& boxes,float ratio,int left,int top,int width,int height);
