#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"
namespace fs=std::filesystem;

struct Config {
    fs::path model="models/yolov8n.onnx", classes="models/classes.txt", output="output";
    std::string source="images/zidane.jpg";
    bool gpu=false, agnostic=false, help=false;
    float conf=0.25f, iou=0.45f;
    int size=640, threads=0, warmup=3, benchmark=0;
};
int Integer(const std::string& value) {
    size_t end=0; int n=std::stoi(value,&end);
    if(end!=value.size()) throw std::invalid_argument("Invalid integer: "+value);
    return n;
}
float Number(const std::string& value) {
    size_t end=0; float n=std::stof(value,&end);
    if(end!=value.size() || !std::isfinite(n)) throw std::invalid_argument("Invalid number: "+value);
    return n;
}
Config Arguments(int argc,char** argv) {
    Config c; bool sourceSet=false;
    for(int i=1;i<argc;++i) {
        std::string a=argv[i];
        auto value=[&]() { if(++i>=argc) throw std::invalid_argument("Missing value for "+a); return std::string(argv[i]); };
        if(a=="--help" || a=="-h") c.help=true;
        else if(a=="gpu" || a=="--gpu") c.gpu=true;
        else if(a=="--agnostic") c.agnostic=true;
        else if(a=="--model") c.model=value();
        else if(a=="--classes") c.classes=value();
        else if(a=="--output") c.output=value();
        else if(a=="--conf") c.conf=Number(value());
        else if(a=="--iou") c.iou=Number(value());
        else if(a=="--size") c.size=Integer(value());
        else if(a=="--threads") c.threads=Integer(value());
        else if(a=="--warmup") c.warmup=Integer(value());
        else if(a=="--benchmark") c.benchmark=Integer(value());
        else if(a.rfind("--",0)==0) throw std::invalid_argument("Unknown option: "+a);
        else { if(sourceSet) throw std::invalid_argument("Only one image/folder/cam source is allowed"); c.source=a; sourceSet=true; }
    }
    if(c.conf<0||c.conf>1||c.iou<0||c.iou>1||c.size<=0||c.threads<0||c.warmup<0||c.benchmark<0)
        throw std::invalid_argument("Invalid threshold, size, thread count or iteration count");
    if(c.benchmark && (c.source=="cam" || fs::is_directory(c.source)))
        throw std::invalid_argument("--benchmark requires one image");
    return c;
}
struct Timing { double pre=0,infer=0,post=0,total=0; };
class Detector {
    // Destruction order: tensor, buffers, session, then environment.
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING,"yolo-demo"};
    Ort::Session session{nullptr};
    std::string inputName,outputName;
    std::array<int64_t,4> shape{};
    int width=0,height=0;
    size_t expectedClasses=0;
    std::vector<float> blob;
    Ort::Value tensor{nullptr};
    const Config& config;
public:
    Detector(const Config& c):config(c) {
        if(!fs::is_regular_file(c.model)) throw std::runtime_error("Model not found: "+c.model.string());
        Ort::SessionOptions options;
        if(c.threads>0) options.SetIntraOpNumThreads(c.threads);
        if(c.gpu) {
            auto providers=Ort::GetAvailableProviders();
            if(std::find(providers.begin(),providers.end(),"TensorrtExecutionProvider")==providers.end())
                throw std::runtime_error("TensorRT EP unavailable; use matching ORT GPU package and DLLs");
            fs::create_directories("trt_cache");
            OrtTensorRTProviderOptions trt{};
            trt.device_id=0; trt.trt_fp16_enable=1;
            trt.trt_max_partition_iterations=1000; trt.trt_min_subgraph_size=1;
            trt.trt_engine_cache_enable=1; trt.trt_engine_cache_path="trt_cache";
            Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_TensorRT(options,&trt));
            std::cout<<"requested backend: TensorRT FP16 (unsupported nodes may use CPU)\n";
        } else std::cout<<"requested backend: CPU\n";
        session=Ort::Session(env,c.model.c_str(),options);
        if(session.GetInputCount()!=1 || session.GetOutputCount()!=1)
            throw std::runtime_error("Expected one input and one raw YOLO detection output");
        Ort::AllocatorWithDefaultOptions allocator;
        inputName=session.GetInputNameAllocated(0,allocator).get();
        outputName=session.GetOutputNameAllocated(0,allocator).get();
        auto inInfo=session.GetInputTypeInfo(0);
        auto in=inInfo.GetTensorTypeAndShapeInfo();
        auto dims=in.GetShape();
        if(in.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || dims.size()!=4 ||
           (dims[0]!=1 && dims[0]!=-1) || dims[1]!=3)
            throw std::runtime_error("Expected float32 NCHW input with batch=1 and channels=3");
        height=dims[2]>0 ? int(dims[2]) : c.size;
        width=dims[3]>0 ? int(dims[3]) : c.size;
        shape={1,3,height,width};
        auto outInfo=session.GetOutputTypeInfo(0);
        auto out=outInfo.GetTensorTypeAndShapeInfo();
        auto od=out.GetShape();
        if(out.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || od.size()!=3 ||
           (od[0]!=1 && od[0]!=-1) || (od[1]>0 && od[1]<=4))
            throw std::runtime_error("Expected raw float32 YOLOv8 output [1,4+classes,candidates]");
        if(od[1]>4) expectedClasses=size_t(od[1]-4);
        blob.resize(size_t(3)*height*width);
        auto memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        tensor=Ort::Value::CreateTensor<float>(memory,blob.data(),blob.size(),shape.data(),shape.size());
        std::cout<<"input "<<inputName<<" [1,3,"<<height<<","<<width<<"], output "<<outputName<<"\n";
    }
    size_t Classes() const { return expectedClasses; }
    std::vector<Box> Detect(const cv::Mat& image,Timing& t) {
        std::vector<Box> boxes;
        t.total=TimeIt([&] {
            float ratio=1; int left=0,top=0;
            t.pre=TimeIt([&] {
                cv::Mat padded=Letterbox(image,width,height,ratio,left,top);
                ToBlob(padded,blob); // Fixed shape: capacity/address stays unchanged.
            });
            std::vector<Ort::Value> outputs;
            const char* inputs[]={inputName.c_str()}; const char* names[]={outputName.c_str()};
            t.infer=TimeIt([&] { outputs=session.Run(Ort::RunOptions{nullptr},inputs,&tensor,1,names,1); });
            t.post=TimeIt([&] {
                auto info=outputs[0].GetTensorTypeAndShapeInfo(); auto dims=info.GetShape();
                if(dims.size()!=3 || dims[0]!=1 || dims[1]<=4 || dims[2]<=0 ||
                   info.GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                    throw std::runtime_error("Unsupported runtime output shape");
                boxes=RestoreBoxes(NMS(ParseOutput(outputs[0].GetTensorData<float>(),
                    size_t(dims[2]),size_t(dims[1]-4),config.conf),config.iou,config.agnostic),
                    ratio,left,top,image.cols,image.rows);
            });
        });
        return boxes;
    }
};
cv::Mat ReadImage(const fs::path& path) {
    cv::Mat image=cv::imread(path.string());
    if(image.empty()) throw std::runtime_error("Cannot read image: "+path.string());
    return image;
}
void Warmup(Detector& detector,const cv::Mat& image,int iterations) {
    Timing t;
    for(int i=0;i<iterations;++i) detector.Detect(image,t);
}
void Save(const Config& c,const fs::path& input,cv::Mat image,
          const std::vector<Box>& boxes,const std::vector<std::string>& names,bool baseline) {
    std::string stem=input.stem().string();
    if(baseline) WriteBaseline((c.output/("baseline_"+stem+"_cpp.txt")).string(),boxes);
    DrawBoxes(image,boxes,names);
    if(!cv::imwrite((c.output/(stem+"_result.jpg")).string(),image))
        throw std::runtime_error("Cannot save result image");
}
double Percentile(std::vector<double> values,double fraction) {
    std::sort(values.begin(),values.end());
    return values.at(size_t(std::ceil(fraction*values.size()))-1);
}
void RunImage(Detector& detector,const Config& c,const std::vector<std::string>& names) {
    cv::Mat image=ReadImage(c.source);
    Warmup(detector,image,c.warmup);
    Timing t; std::vector<Box> boxes; std::vector<double> total;
    double pre=0,infer=0,post=0;
    int iterations=std::max(1,c.benchmark);
    for(int i=0;i<iterations;++i) {
        boxes=detector.Detect(image,t); total.push_back(t.total);
        pre+=t.pre; infer+=t.infer; post+=t.post;
    }
    double mean=std::accumulate(total.begin(),total.end(),0.0)/iterations;
    std::cout<<std::fixed<<std::setprecision(3)<<"preprocess mean: "<<pre/iterations
             <<" ms, inference mean: "<<infer/iterations<<" ms, postprocess mean: "<<post/iterations
             <<" ms\ndetect mean: "<<mean<<" ms, p50: "<<Percentile(total,0.5)
             <<" ms, p95: "<<Percentile(total,0.95)<<" ms, detect FPS: "<<1000/mean<<"\n";
    Save(c,c.source,image,boxes,names,true);
    if(c.benchmark) {
        std::ofstream csv(c.output/"benchmark_cpp.csv");
        if(!csv) throw std::runtime_error("Cannot write benchmark CSV");
        csv<<"backend,iterations,warmup,pre_ms,infer_ms,post_ms,detect_mean_ms,p50_ms,p95_ms\n"
           <<(c.gpu?"tensorrt_fp16":"cpu")<<","<<iterations<<","<<c.warmup<<","
           <<pre/iterations<<","<<infer/iterations<<","<<post/iterations<<","<<mean<<","
           <<Percentile(total,0.5)<<","<<Percentile(total,0.95)<<"\n";
        if(!csv) throw std::runtime_error("Failed writing benchmark CSV");
    }
}
void RunBatch(Detector& detector,const Config& c,const std::vector<std::string>& names) {
    std::vector<fs::path> paths;
    for(const auto& entry:fs::directory_iterator(c.source)) {
        if(!entry.is_regular_file()) continue;
        std::string ext=entry.path().extension().string();
        std::transform(ext.begin(),ext.end(),ext.begin(),[](unsigned char x){return char(std::tolower(x));});
        if(ext==".jpg"||ext==".jpeg"||ext==".png"||ext==".bmp") paths.push_back(entry.path());
    }
    std::sort(paths.begin(),paths.end());
    if(paths.empty()) throw std::runtime_error("No supported images in folder");
    for(size_t i=0;i<paths.size();++i) {
        cv::Mat image=ReadImage(paths[i]); if(i==0) Warmup(detector,image,c.warmup);
        Timing t; auto boxes=detector.Detect(image,t);
        Save(c,paths[i],image,boxes,names,false);
        std::cout<<paths[i].filename().string()<<": "<<boxes.size()<<" boxes\n";
    }
}
void RunCamera(Detector& detector,const Config& c,const std::vector<std::string>& names) {
    cv::VideoCapture cap(0);
    if(!cap.isOpened()) throw std::runtime_error("Cannot open camera");
    cv::Mat frame;
    if(!cap.read(frame)) throw std::runtime_error("Cannot read camera frame");
    Warmup(detector,frame,c.warmup);
    size_t frames=0; double detectMs=0;
    double wallMs=TimeIt([&] {
        while(cap.read(frame)) {
            Timing t; auto boxes=detector.Detect(frame,t); detectMs+=t.total; ++frames;
            DrawBoxes(frame,boxes,names); cv::imshow("yolo",frame);
            if(cv::waitKey(1)==27) break;
        }
    });
    if(frames) std::cout<<"detect avg: "<<detectMs/frames<<" ms, detect FPS: "<<frames*1000/detectMs
                        <<", camera loop FPS: "<<frames*1000/wallMs<<"\n";
}
int main(int argc,char** argv) {
    try {
        Config c=Arguments(argc,argv);
        if(c.help) {
            std::cout<<"yolo_demo [gpu] [image|folder|cam] [options]\n"
                     <<"--model PATH --classes PATH --output DIR --conf 0.25 --iou 0.45\n"
                     <<"--size 640 (dynamic dimensions only) --threads N (0=ORT default)\n"
                     <<"--warmup 3 --benchmark N (image only) --agnostic --gpu\n"; return 0;
        }
        if(c.source!="cam" && !fs::exists(c.source)) throw std::runtime_error("Source not found: "+c.source);
        fs::create_directories(c.output);
        Detector detector(c); auto names=LoadClassNames(c.classes.string());
        if(!names.empty() && detector.Classes() && names.size()!=detector.Classes())
            throw std::runtime_error("Class-name count does not match model output");
        if(names.empty()) std::cerr<<"warning: class names unavailable, using numeric labels\n";
        if(c.source=="cam") RunCamera(detector,c,names);
        else if(fs::is_directory(c.source)) RunBatch(detector,c,names);
        else RunImage(detector,c,names);
        return 0;
    } catch(const std::exception& e) { std::cerr<<"error: "<<e.what()<<"\n"; return 1; }
}
