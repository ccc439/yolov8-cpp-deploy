#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"

// 单帧检测：前处理、推理、后处理；ratio / padLeft / padTop 是输出参数（还原坐标用）
std::vector<Box> Detect(const cv::Mat& img, Ort::Session& session,
                        float& ratio, int& padLeft, int& padTop,
                        double* preMs = nullptr, double* inferMs = nullptr, double* postMs = nullptr) {
    // 前处理
    cv::Mat padded;
    std::vector<float> blob;
    auto pre = [&] {
        padded = Letterbox(img, 640, ratio, padLeft, padTop);
        blob = ToBlob(padded);
    };
    if (preMs) *preMs = TimeIt(pre);
    else pre();

    // 输入张量：借用 blob 内存不拷贝，blob 在 Run 结束前不能销毁
    std::array<int64_t, 4> shape = {1, 3, 640, 640};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, blob.data(), blob.size(), shape.data(), shape.size());
    const char* inputNames[]  = {"images"};
    const char* outputNames[] = {"output0"};

    // 推理：outputs 声明在 lambda 外，data 指向它持有的张量内存
    std::vector<Ort::Value> outputs;
    float* data = nullptr;
    auto run = [&] {
        outputs = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        data = outputs[0].GetTensorMutableData<float>();
    };
    if (inferMs) *inferMs = TimeIt(run);
    else run();

    // 后处理
    std::vector<Box> boxes;
    auto post = [&] { boxes = NMS(ParseOutput(data)); };
    if (postMs) *postMs = TimeIt(post);
    else post();
    return boxes;
}

// 单图模式：一张图、三段计时、基准文本（和 Python 对齐验证用）
void RunSingleImage(Ort::Session& session, bool useTensorRT,
                    const std::vector<std::string>& classNames) {
    cv::Mat img = cv::imread("images/zidane.jpg");
    float ratio = 1.0f; int padLeft = 0, padTop = 0;
    // GPU 热身：首次 Run 含引擎构建，白跑一次不计入计时
    if (useTensorRT)
        Detect(img, session, ratio, padLeft, padTop);

    double preMs = 0, inferMs = 0, postMs = 0;
    std::vector<Box> boxes = Detect(img, session, ratio, padLeft, padTop, &preMs, &inferMs, &postMs);

    DrawBoxes(img, boxes, classNames, ratio, padLeft, padTop);
    WriteBaseline("output/baseline_zidane_cpp.txt", boxes, ratio, padLeft, padTop);
    cv::imwrite("output/zidane_result.jpg", img);

    std::cout << "preprocess: " << preMs << " ms" << std::endl;
    std::cout << "inference:  " << inferMs << " ms" << std::endl;
    std::cout << "postprocess: " << postMs << " ms" << std::endl;
}

// 批量模式：遍历文件夹，每张图画框存 output/<原名>_result.jpg
void RunBatch(Ort::Session& session, const std::string& folder,
              const std::vector<std::string>& classNames) {
    namespace fs = std::filesystem;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".png") continue;
        cv::Mat img = cv::imread(entry.path().string());
        if (img.empty()) continue;
        float ratio = 1.0f; int padLeft = 0, padTop = 0;
        std::vector<Box> boxes = Detect(img, session, ratio, padLeft, padTop);
        DrawBoxes(img, boxes, classNames, ratio, padLeft, padTop);
        std::string name = entry.path().stem().string();
        cv::imwrite("output/" + name + "_result.jpg", img);
        std::cout << "saved output/" << name << "_result.jpg (" << boxes.size() << " boxes)" << std::endl;
    }
}

// 摄像头模式：实时检测 + imshow，ESC 退出，退出打印平均耗时和 FPS
void RunCamera(Ort::Session& session, bool useTensorRT,
               const std::vector<std::string>& classNames) {
    cv::VideoCapture cap(0);   // 0 = 默认摄像头
    if (!cap.isOpened()) {
        std::cout << "cannot open camera" << std::endl;
        return;
    }
    cv::Mat frame;
    float ratio = 1.0f; int padLeft = 0, padTop = 0;
    // GPU 热身：先白跑一帧，引擎构建不计入 FPS
    cap.read(frame);
    if (!frame.empty() && useTensorRT)
        Detect(frame, session, ratio, padLeft, padTop);

    double totalMs = 0;
    int frames = 0;
    while (cap.read(frame)) {
        double preMs = 0, inferMs = 0, postMs = 0;
        std::vector<Box> boxes = Detect(frame, session, ratio, padLeft, padTop, &preMs, &inferMs, &postMs);
        totalMs += preMs + inferMs + postMs;
        frames++;
        DrawBoxes(frame, boxes, classNames, ratio, padLeft, padTop);
        cv::imshow("yolo", frame);
        if (cv::waitKey(1) == 27) break;   // ESC 退出
    }
    if (frames > 0)
        std::cout << "avg " << totalMs / frames << " ms/frame, "
                  << frames / (totalMs / 1000.0) << " FPS" << std::endl;
}

int main(int argc, char** argv) {
    // 参数解析：可选 gpu + 模式（默认 = 单图，文件夹路径 = 批量，"cam" = 摄像头）
    bool useTensorRT = false;
    int modeArg = 1;
    if (argc > 1 && std::string(argv[1]) == "gpu") { useTensorRT = true; modeArg = 2; }
    std::string mode = (argc > modeArg) ? argv[modeArg] : "";

    // 加载模型
    // Env 必须比 Session 活得久：env 先声明、session 后声明（析构顺序相反，session 先析构）
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo-demo");
    Ort::SessionOptions opts;
    // 线程数取 CPU 逻辑核数；获取失败（返回 0）时退回 4
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;
    opts.SetIntraOpNumThreads(numThreads);

    // EP 诊断：打印 ORT 当前能提供哪些 EP（有没有 TensorRT 一眼可见）
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    char** providerNames = nullptr;
    int numProviders = 0;
    api->GetAvailableProviders(&providerNames, &numProviders);
    std::cout << "available providers:";
    for (int p = 0; p < numProviders; p++)
        std::cout << " " << providerNames[p];
    std::cout << std::endl;

    // GPU 模式：挂 TensorRT EP
    if (useTensorRT) {
        std::cout << "appending TRT EP..." << std::endl;
        OrtTensorRTProviderOptions trtOpts{};
        trtOpts.device_id = 0;
        trtOpts.trt_fp16_enable = 1;
        trtOpts.trt_max_partition_iterations = 1000;
        trtOpts.trt_min_subgraph_size = 1;
        trtOpts.trt_engine_cache_enable = 1;
        trtOpts.trt_engine_cache_path = "trt_cache";
        OrtStatus* status = api->SessionOptionsAppendExecutionProvider_TensorRT(opts, &trtOpts);
        if (status != nullptr) {
            std::cout << "append TRT EP failed: " << api->GetErrorMessage(status) << std::endl;
            api->ReleaseStatus(status);
        } else {
            std::cout << "append TRT EP ok" << std::endl;
        }
    }

    // Session 创建包 try/catch：ORT 的 C++ 接口出错时抛异常，不接住就是静默闪退
    std::unique_ptr<Ort::Session> sessionPtr;
    try {
        sessionPtr = std::make_unique<Ort::Session>(env, L"models/yolov8n.onnx", opts);
    } catch (const Ort::Exception& e) {
        std::cout << "CreateSession failed: " << e.what() << std::endl;
        return 1;
    }
    Ort::Session& session = *sessionPtr;

    // 类名表
    std::vector<std::string> classNames = LoadClassNames("models/classes.txt");

    // 模式分发
    if (mode == "cam") {
        RunCamera(session, useTensorRT, classNames);
    } else if (!mode.empty()) {
        RunBatch(session, mode, classNames);
    } else {
        RunSingleImage(session, useTensorRT, classNames);
    }
    return 0;
}