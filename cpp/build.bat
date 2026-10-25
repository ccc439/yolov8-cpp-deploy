@echo off
rem ============ YOLOv8 部署项目编译脚本 ============

rem 先切到脚本所在目录（防止双击运行时工作目录不对）
cd /d "%~dp0"

rem 加载 MSVC 编译环境（提供 cl 命令）
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

rem SDK 路径（按本机实际位置改；GPU 包同时支持 CPU 和 TensorRT 两种模式）
set OPENCV_DIR=D:\YOLOTR\CppOpenCV\opencv\build
set ORT_DIR=D:\YOLOTR\CppOpenCV\onnxruntime\onnxruntime-win-x64-gpu-1.19.2

rem 产物目录：exe、obj、dll 全部放 bin，不污染源码目录
if not exist ..\bin mkdir ..\bin

rem 编译：/O2 优化、/MD 动态运行库、/EHsc 异常处理、C++17、/Fo 中间文件进 bin
cl /nologo /utf-8 /O2 /MD /EHsc /std:c++17 /Fo..\bin\ /I"%OPENCV_DIR%\include" /I"%ORT_DIR%\include" main.cpp preprocess.cpp postprocess.cpp utils.cpp /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4100.lib /LIBPATH:"%ORT_DIR%\lib" onnxruntime.lib /OUT:..\bin\yolo_demo.exe

rem 拷贝运行所需的 DLL 到 exe 旁边
copy /Y "%OPENCV_DIR%\x64\vc16\bin\opencv_world4100.dll" ..\bin\ >nul
copy /Y "%ORT_DIR%\lib\onnxruntime.dll" ..\bin\ >nul
copy /Y "%ORT_DIR%\lib\onnxruntime_providers_shared.dll" ..\bin\ >nul
copy /Y "%ORT_DIR%\lib\onnxruntime_providers_cuda.dll" ..\bin\ >nul
copy /Y "%ORT_DIR%\lib\onnxruntime_providers_tensorrt.dll" ..\bin\ >nul

echo Build OK. Run from project root: bin\yolo_demo.exe
pause
