#include "preprocess.h"
#include "postprocess.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
void Check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
int main() {
    try {
        Box a{0,0,10,10,.9f,0}, same{0,0,10,10,.8f,0}, other{0,0,10,10,.7f,1};
        Check(NMS({a,same,other}).size()==2,"Class-aware NMS");
        Check(NMS({a,same,other},.45f,true).size()==1,"Agnostic NMS");
        Check(ComputeIoU(a,a)==1,"Identical box IoU");
        Check(ComputeIoU({0,0,0,0,0,0},{0,0,0,0,0,0})==0,"Degenerate IoU");
        // Two candidates, two classes: shape [1,6,2], not COCO-80.
        float output[]={5,15,5,15,10,10,10,10,.9f,.1f,.1f,.8f};
        auto parsed=ParseOutput(output,2,2,.25f);
        Check(parsed.size()==2 && parsed[1].cls==1,"Shape-driven parsing");
        output[0]=std::numeric_limits<float>::quiet_NaN();
        Check(ParseOutput(output,2,2,.25f).size()==1,"Invalid boxes rejected");
        auto clipped=RestoreBoxes({{-10,-10,110,110,.9f,0}},1,0,0,100,100);
        Check(clipped.size()==1 && clipped[0].x1==0 && clipped[0].x2==100,"Coordinate clipping");
        cv::Mat image(3,5,CV_8UC3,cv::Scalar(10,20,30));
        float ratio; int left,top;
        auto padded=Letterbox(image,8,8,ratio,left,top);
        Check(padded.cols==8 && padded.rows==8 && top==1,"Odd letterbox padding");
        std::vector<float> blob;
        ToBlob(image,blob); auto* ptr=blob.data();
        Check(std::abs(blob[0]-30.0f/255)<1e-6 && std::abs(blob[15]-20.0f/255)<1e-6,"RGB CHW values");
        ToBlob(image,blob); Check(ptr==blob.data(),"Input buffer reused");
        bool rejected=false;
        try { Letterbox(cv::Mat(),8,8,ratio,left,top); } catch(const std::invalid_argument&) {rejected=true;}
        Check(rejected,"Empty image rejected");
        std::cout<<"All core tests passed\n"; return 0;
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
