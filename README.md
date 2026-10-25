# YOLOv8 C++ 部署（ONNX Runtime）

用 C++ 和 ONNX Runtime 部署 YOLOv8 目标检测模型，前后处理（letterbox、IoU、NMS）全部手写实现。支持单图、批量图片、摄像头实时检测三种模式，后端可切换 CPU 和 TensorRT GPU。

## 环境要求

- Windows 10/11，VS 2022（MSVC，C++17）
- [OpenCV](https://opencv.org/releases/) 4.10 Windows 预编译包
- [ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) 1.19.2：CPU 模式用 onnxruntime-win-x64 包；TensorRT 模式用 onnxruntime-win-x64-gpu 包，另需 CUDA 12、cuDNN、TensorRT 10 的运行时 DLL
- Python 3.12 + ultralytics（仅用于导出模型和结果对照，运行 demo 不需要）

## 模型准备

仓库不含 `models/` 下的任何文件，两个文件都自行准备后放到 `models/`：

**1. yolov8n.onnx**（约 12MB）

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640 simplify=True
```

**2. classes.txt**（COCO 80 类名，每行一个，按模型输出下标 0~79 对应，画框标签用）

```text
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush
```

换自定义模型时，类名表换成对应类别即可；程序读不到 classes.txt 时标签退回纯编号。

## 构建

1. 打开 `cpp\build.bat`，把 `OPENCV_DIR`、`ORT_DIR` 改成本机 SDK 路径
2. 运行 `cpp\build.bat`，编译产物输出到 `bin\`（exe 和运行时 DLL）
3. 在项目根目录运行 `bin\yolo_demo.exe`（程序按相对路径读 images/、models/）

## 运行

| 命令 | 说明 |
| :--- | :--- |
| `bin\yolo_demo.exe` | 单图模式：检测 images/zidane.jpg，打印三段耗时 |
| `bin\yolo_demo.exe images` | 批量模式：遍历文件夹，结果存 output/ |
| `bin\yolo_demo.exe cam` | 摄像头模式：实时显示，ESC 退出并打印平均耗时和 FPS |

以上命令加 `gpu` 前缀走 TensorRT，如 `bin\yolo_demo.exe gpu cam`。

## 检测流程

1. 前处理：Letterbox 等比缩放 + 灰边填充（114）到 640×640，BGR 转 RGB、HWC 转 CHW、除 255 归一化
2. 推理：ONNX Runtime 前向，输出 [1, 84, 8400]（84 = 框坐标 4 + 类别得分 80，8400 = 候选框数）
3. 后处理：按置信度 0.25 过滤，手写 IoU + NMS（阈值 0.45）去重
4. 坐标还原：减灰边、除缩放比还原到原图坐标，画框标注"类别名 + 置信度"

单图模式会在 output/ 生成基准文本（baseline_*_cpp.txt），和 python/infer.py 的输出逐框对比，可验证 C++ 端与 Python 端结果一致。

## 性能参考

2026-10 实测（GPU：RTX 4050 Laptop，TensorRT fp16），单位 ms：

| 方案 | 预处理 | 推理 | 后处理 | 总耗时 | FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Python + ONNX Runtime | 13.18 | 152.04 | 16.47 | 181.69 | ~5.5 |
| C++ + ONNX Runtime (CPU) | 7.86 | 28.56 | 0.51 | 36.93 | ~27 |
| C++ + TensorRT (GPU fp16) | 7.32 | 4.54 | 0.51 | 12.36 | ~81 |

## 目录结构

```text
├── cpp\          # C++ 源码 + build.bat
├── models\       # 运行前自行准备：yolov8n.onnx + classes.txt（仓库不含）
├── images\       # 测试图片
├── output\       # 运行生成：结果图 + 基准文本
└── python\       # infer.py 对照脚本
```
