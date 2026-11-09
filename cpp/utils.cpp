#include "utils.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
void DrawBoxes(cv::Mat& img,const std::vector<Box>& boxes,const std::vector<std::string>& names) {
    for(const auto& b:boxes) {
        cv::rectangle(img,cv::Point(int(b.x1),int(b.y1)),cv::Point(int(b.x2),int(b.y2)),cv::Scalar(0,255,0),2);
        std::ostringstream label;
        label << (b.cls>=0 && size_t(b.cls)<names.size()?names[b.cls]:std::to_string(b.cls))
              << " " << std::fixed << std::setprecision(2) << b.score;
        cv::putText(img,label.str(),cv::Point(int(b.x1),std::max(15,int(b.y1)-5)),
                    cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(0,255,0),1);
    }
}
void WriteBaseline(const std::string& path,const std::vector<Box>& boxes) {
    std::ofstream f(path);
    if(!f) throw std::runtime_error("Cannot write baseline: "+path);
    for(const auto& b:boxes)
        f << b.cls << " " << std::fixed << std::setprecision(2)
          << b.x1 << " " << b.y1 << " " << b.x2 << " " << b.y2 << " "
          << std::setprecision(4) << b.score << "\n";
    if(!f) throw std::runtime_error("Failed writing baseline: "+path);
}
std::vector<std::string> LoadClassNames(const std::string& path) {
    std::ifstream f(path); std::vector<std::string> names; std::string line;
    while(std::getline(f,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back();
        if(!line.empty()) names.push_back(line);
    }
    return names;
}
