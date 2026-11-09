# YOLOv8 C++ 目标检测部署

基于 **C++17、OpenCV 和 ONNX Runtime** 部署 YOLOv8 目标检测模型，支持单张图片、文件夹图片和摄像头实时检测，并提供可选的 TensorRT FP16 加速。

前后处理自行实现，包括 Letterbox、BGR → RGB、HWC → CHW、归一化、输出解析、IoU、按类别 NMS 和坐标还原。Python 脚本用于与 C++ 检测结果对照。

## 功能

- **三种输入方式**：单图、文件夹逐图处理、默认摄像头。
- **两种执行方式**：CPU、TensorRT FP16。
- **模型适配**：读取模型输入输出名称、固定输入尺寸、类别数和候选数，支持相同输出布局的自定义类别模型。
- **运行参数**：可指定模型、类名、输出目录、阈值、线程数和动态输入尺寸。
- **性能统计**：预热后重复测量，输出均值、P50、P95 和 CSV。
- **结果验证**：提供核心算法测试和 Python/C++ 逐框匹配脚本。

## 环境要求

当前构建脚本面向 **Windows x64 + MSVC**，以下为已验证的 C++ 环境：

| 依赖 | 版本 / 要求 |
| :--- | :--- |
| Visual Studio | 2022，安装“使用 C++ 的桌面开发”工作负载 |
| C++ 标准 | C++17 |
| OpenCV | 4.10.0 Windows 预编译 SDK |
| ONNX Runtime | 1.19.2 Windows x64 C/C++ SDK |
| Python | 3.12，用于模型导出和结果对照；C++ 程序运行时不需要 Python |

SDK 下载：[OpenCV 4.10.0](https://github.com/opencv/opencv/releases/tag/4.10.0)、[ONNX Runtime 1.19.2](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2)。

先运行 CPU 版本只需准备 OpenCV 和 ORT CPU 包；启用 GPU 的附加条件见下文“TensorRT 加速”。模型、SDK、运行时 DLL 和编译产物均不随仓库提供。

## 快速开始：CPU 检测

以下命令使用 **PowerShell**，均从仓库根目录执行。

### 1. 获取代码

```powershell
git clone https://github.com/ccc439/yolov8-cpp-deploy.git
cd yolov8-cpp-deploy
```

### 2. 准备模型与类别文件

本例使用官方 YOLOv8n COCO 预训练模型。创建 Python 虚拟环境并安装导出工具：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ultralytics onnx
```

在同一个 PowerShell 窗口执行下面的代码，将导出的模型和类别名保存到 `models/`。首次执行会下载 `yolov8n.pt`，导出工具可能还需下载其依赖：

```powershell
@'
from pathlib import Path
from ultralytics import YOLO

target = Path("models")
target.mkdir(exist_ok=True)
model = YOLO(str(target / "yolov8n.pt"))
exported = Path(model.export(
    format="onnx", imgsz=640, batch=1, dynamic=False,
    opset=17, simplify=True, nms=False,
))
destination = target / "yolov8n.onnx"
if exported.resolve() != destination.resolve():
    exported.replace(destination)
names = [model.names[i] for i in range(len(model.names))]
(target / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
print("Ready:", destination, "and", target / "classes.txt")
'@ | .\.venv\Scripts\python.exe -
```

也可以自行提供兼容的 ONNX 模型和类别表。`classes.txt` 每行一个类别名，顺序必须与模型类别编号一致。文件缺失时程序显示类别编号；提供了非空类别表但数量与已知模型类别数不一致时会报错。

### 3. 设置 SDK 路径并编译

解压下载的 SDK，按实际安装位置修改下面的路径。`OPENCV_DIR` 指向包含 `include/` 和 `x64/` 的 OpenCV `build` 目录；`ORT_DIR` 指向包含 `include/` 和 `lib/` 的 ORT SDK 根目录：

```powershell
$env:OPENCV_DIR = "C:\deps\opencv\build"
$env:ORT_DIR = "C:\deps\onnxruntime-win-x64-1.19.2"
$env:VS_VCVARS = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

.\cpp\build.bat --test
```

`VS_VCVARS` 按 Visual Studio 版本和安装位置调整。脚本编译主程序与核心测试，并将 OpenCV、ORT 的运行库复制到 `bin/`。成功后可看到 `All core tests passed` 和 `Build OK`，主程序位于 `bin/yolo_demo.exe`。

脚本中的 OpenCV 库名固定为 `opencv_world4100.lib/.dll`，使用其他 OpenCV 版本时还需同步修改库名和库目录。编译失败会返回非零退出码。

### 4. 运行

```powershell
.\bin\yolo_demo.exe
```

默认检测仓库中的 `images/zidane.jpg`，输出：

```text
output/
├── zidane_result.jpg          # 检测结果图
└── baseline_zidane_cpp.txt    # 类别、原图坐标与得分
```

控制台打印预处理、推理、后处理和总检测耗时。请从仓库根目录运行，默认模型与图片路径相对于当前工作目录。

## 使用方式

```powershell
# 指定一张图片
.\bin\yolo_demo.exe images\bus.jpg

# 文件夹逐图检测，结果保存到 output/
.\bin\yolo_demo.exe images

# 默认摄像头，按 ESC 退出
.\bin\yolo_demo.exe cam

# 指定模型、类别表和输出目录
.\bin\yolo_demo.exe images\bus.jpg --model models\custom.onnx --classes models\custom.txt --output output\custom

# 调整置信度与 NMS 阈值
.\bin\yolo_demo.exe images\bus.jpg --conf 0.3 --iou 0.5

# 预热 5 次，再对单图测量 100 次
.\bin\yolo_demo.exe --benchmark 100 --warmup 5
```

文件夹模式不递归，支持 `.jpg`、`.jpeg`、`.png`、`.bmp`，扩展名不区分大小写；每张图片独立推理，模型 batch 仍为 1。同名、不同扩展名的图片会使用相同输出文件名，请避免在同一次任务中使用相同文件主名。

### 命令行参数

语法：`yolo_demo [gpu] [图片路径 | 文件夹路径 | cam] [选项]`。

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| 输入源 | `images/zidane.jpg` | 单图、文件夹或 `cam` |
| `gpu` / `--gpu` | 关闭 | 请求 TensorRT FP16 |
| `--model PATH` | `models/yolov8n.onnx` | ONNX 模型 |
| `--classes PATH` | `models/classes.txt` | 按类别编号排序的类名表 |
| `--output DIR` | `output` | 输出目录，自动创建 |
| `--conf FLOAT` | `0.25` | 置信度阈值，范围 0～1 |
| `--iou FLOAT` | `0.45` | NMS IoU 阈值，范围 0～1 |
| `--agnostic` | 关闭 | 使用跨类别 NMS；默认按类别抑制 |
| `--size INT` | `640` | 动态输入高宽的取值；固定维度以模型为准 |
| `--threads INT` | `0` | ORT intra-op 线程数；0 使用 ORT 默认策略 |
| `--warmup INT` | `3` | 正式检测前的预热次数 |
| `--benchmark INT` | `0` | 对单图重复测量 N 次；0 为普通检测 |
| `--help` / `-h` | — | 显示帮助 |

## TensorRT 加速（可选）

需要 NVIDIA GPU、驱动，以及 **ORT GPU SDK 与其匹配的 TensorRT、CUDA、cuDNN 运行库**。参考 [ORT TensorRT EP 官方兼容说明](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)，按所选 ORT 版本准备依赖。

使用 GPU 包重新构建，例如：

```powershell
$env:ORT_DIR = "C:\deps\onnxruntime-win-x64-gpu-1.19.2"
.\cpp\build.bat --test

.\bin\yolo_demo.exe gpu images\bus.jpg
.\bin\yolo_demo.exe gpu cam
.\bin\yolo_demo.exe gpu --benchmark 100 --warmup 5
```

构建脚本只复制 OpenCV 和 ORT 的 DLL，TensorRT/CUDA/cuDNN 及其间接依赖仍需放在 `bin/` 或加入当前进程的 `PATH`，保持 DLL 原名。首次执行可能较慢，TensorRT 引擎缓存位于 `trt_cache/`。

程序通过 ORT TensorRT Execution Provider 启用 FP16，不直接加载用户预制的 `.engine` 文件。显式请求 GPU 后，TensorRT 不可用或注册失败会报错；未由 TensorRT 接管的节点仍可能在 CPU 执行。

## 性能参考

RTX 4050 Laptop 测试机器，Windows、MSVC `/O2`、OpenCV 4.10、ORT 1.19.2；YOLOv8n、640×640 输入、`zidane.jpg`，预热 5 次后测量 30 次。记录日期：2026-11-09。

| 后端 | 预处理均值 | 推理均值 | 后处理均值 | 检测总均值 | P50 | P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | 0.701 ms | 18.897 ms | 0.777 ms | 20.377 ms | 20.131 ms | 22.588 ms |
| TensorRT FP16 | 0.722 ms | 2.729 ms | 0.647 ms | 4.099 ms | 4.051 ms | 4.346 ms |

总检测耗时包含前后处理、`Session.Run` 和调用管理开销，**不包含模型加载、预热、读图、画框和文件保存**。表格来自一次本机测量，硬件、功耗和运行负载会影响结果；CPU 行不代表所有 CPU 的性能。

开启 `--benchmark` 后会保存 `output/benchmark_cpp.csv`，可在自己的机器上重复测量。控制台 `detect FPS` 由检测均值换算；摄像头另有包含采集、绘制和显示的 `camera loop FPS`，两者口径不同。

## 测试与 Python 对照

核心测试随以下命令构建运行：

```powershell
.\cpp\build.bat --test
```

覆盖按类别 / 跨类别 NMS、退化框 IoU、两类别输出解析、无效坐标、坐标裁剪、Letterbox、RGB/CHW 值和输入 buffer 复用。

在同一模型、图片和阈值下，分别运行 C++ 和 Python，再匹配基准：

```powershell
.\bin\yolo_demo.exe images\zidane.jpg
.\.venv\Scripts\python.exe python\infer.py images\zidane.jpg
.\.venv\Scripts\python.exe tests\compare_baseline.py output\baseline_zidane.txt output\baseline_zidane_cpp.txt
```

匹配脚本不依赖输出行序，要求类别一致、坐标误差不超过 0.5 像素、得分误差不超过 0.001。Python 对照默认使用 CPU，加载模型、预热和文件输出不计入检测耗时。已验证 CPU 下的 `bus.jpg`、`zidane.jpg`，以及两类别动态输入测试模型；这类一致性检查不等同于数据集 mAP 评估。

## 模型兼容范围

支持单输入、单输出的 **YOLOv8 检测模型原始输出**：

- 输入：float32、NCHW、batch=1、三通道 RGB。
- 输出：float32、`[1, 4 + 类别数, 候选数]`，前四项为模型输入坐标系的 `cx, cy, w, h`。
- 每个候选框选最高分类别，默认执行按类别 NMS，再将坐标还原、裁剪到原图。

不支持多 batch、转置输出布局、带独立 objectness 的 YOLOv5 输出、内置 NMS 输出，以及分割或姿态模型。TensorRT FP16 是内部计算精度，当前模型外部接口仍要求 float32。

## 常见问题

| 问题 | 检查方式 |
| :--- | :--- |
| 找不到模型或图片 | 从仓库根目录运行，或通过参数传入正确路径 |
| 找不到 OpenCV / ORT 库 | 检查 SDK 根目录、x64 架构和脚本中的库名 |
| 请求 API 19，但运行库只支持较低版本 | 头文件和实际加载的 ORT DLL 不是同一版本；核对 `bin/onnxruntime.dll` |
| TensorRT 不可用或 error 126 | 检查 ORT 是否为 GPU 包、版本兼容性，以及目标 DLL 的间接依赖 |
| 类别数量不匹配 | 类名文件必须与模型输出类别数一致 |
| 摄像头无法打开 | 当前使用设备索引 0，检查设备连接及相机访问权限 |

## 目录结构

```text
├── cpp/                  # C++ 检测实现与 Windows 构建脚本
├── python/infer.py       # Python 参考实现
├── tests/                # 核心测试、结果匹配与测试模型生成
├── images/               # 示例图片
├── datasets/NEU-DET/     # 储备数据，不参与当前示例运行
├── requirements.txt      # Python 对照依赖
├── models/               # 自行准备的模型与类名（不入库）
├── bin/                  # 编译产物与运行库（不入库）
├── output/               # 检测结果与统计（不入库）
└── trt_cache/            # TensorRT 缓存（不入库）
```

模型导出参数的更多说明见 [Ultralytics 导出文档](https://docs.ultralytics.com/modes/export/)。
