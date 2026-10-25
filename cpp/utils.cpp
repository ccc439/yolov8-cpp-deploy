#include "utils.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

// 画框：坐标还原 + 画框 + 标类别（只改图，不写文件）
void DrawBoxes(cv::Mat& img, const std::vector<Box>& boxes,
               const std::vector<std::string>& classNames,
               float ratio, int padLeft, int padTop) {
    for (const Box& b : boxes) {
        // 减灰边 = 把 0 点挪回真实图像，除 ratio = 放大回原尺寸
        float x1 = (b.x1 - padLeft) / ratio;
        float y1 = (b.y1 - padTop)  / ratio;
        float x2 = (b.x2 - padLeft) / ratio;
        float y2 = (b.y2 - padTop)  / ratio;
        cv::rectangle(img, cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2),
                      cv::Scalar(0, 255, 0), 2);
        // 标签：名字 + 置信度（2 位小数）；表空退回纯编号
        std::string label = std::to_string(b.cls);
        if ((size_t)b.cls < classNames.size()) {
            std::ostringstream oss;
            oss << classNames[b.cls] << " " << std::fixed << std::setprecision(2) << b.score;
            label = oss.str();
        }
        cv::putText(img, label, cv::Point((int)x1, (int)y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

// 写基准文本：格式和 Python 端 baseline 一致（cls x1 y1 x2 y2 score，坐标 2 位小数、得分 4 位）
void WriteBaseline(const std::string& path, const std::vector<Box>& boxes, float ratio, int padLeft, int padTop) {
    std::ofstream f(path);
    for (const Box& b : boxes) {
        float x1 = (b.x1 - padLeft) / ratio;
        float y1 = (b.y1 - padTop)  / ratio;
        float x2 = (b.x2 - padLeft) / ratio;
        float y2 = (b.y2 - padTop)  / ratio;
        f << b.cls << " " << std::fixed << std::setprecision(2)
          << x1 << " " << y1 << " " << x2 << " " << y2 << " "
          << std::setprecision(4) << b.score << "\n";
    }
}

// 读类名表：每行一个名字，跳过空行；文件读不到返回空表（画框退回编号）
std::vector<std::string> LoadClassNames(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream f(path);
    if (!f) return names;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) names.push_back(line);
    }
    return names;
}