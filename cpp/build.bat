@echo off
setlocal
cd /d "%~dp0"
if not defined OPENCV_DIR set "OPENCV_DIR=D:\YOLOTR\CppOpenCV\opencv\build"
if not defined ORT_DIR set "ORT_DIR=D:\YOLOTR\CppOpenCV\onnxruntime\onnxruntime-win-x64-gpu-1.19.2"
if not defined VS_VCVARS set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call "%VS_VCVARS%"
if errorlevel 1 exit /b 1
if not exist ..\bin mkdir ..\bin
cl /nologo /utf-8 /O2 /MD /EHsc /W4 /std:c++17 /Fo..\bin\ /I"%OPENCV_DIR%\include" /I"%ORT_DIR%\include" main.cpp preprocess.cpp postprocess.cpp utils.cpp /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4100.lib /LIBPATH:"%ORT_DIR%\lib" onnxruntime.lib /OUT:..\bin\yolo_demo.exe
if errorlevel 1 exit /b 1
copy /Y "%OPENCV_DIR%\x64\vc16\bin\opencv_world4100.dll" ..\bin\ >nul
if errorlevel 1 exit /b 1
copy /Y "%ORT_DIR%\lib\onnxruntime.dll" ..\bin\ >nul
if errorlevel 1 exit /b 1
for %%F in (onnxruntime_providers_shared.dll onnxruntime_providers_cuda.dll onnxruntime_providers_tensorrt.dll) do if exist "%ORT_DIR%\lib\%%F" copy /Y "%ORT_DIR%\lib\%%F" ..\bin\ >nul
if not "%~1"=="--test" goto done
cl /nologo /utf-8 /O2 /MD /EHsc /W4 /std:c++17 /Fo..\bin\ /I. /I"%OPENCV_DIR%\include" ..\tests\core_test.cpp preprocess.cpp postprocess.cpp /link /LIBPATH:"%OPENCV_DIR%\x64\vc16\lib" opencv_world4100.lib /OUT:..\bin\core_test.exe
if errorlevel 1 exit /b 1
..\bin\core_test.exe
if errorlevel 1 exit /b 1
:done
echo Build OK. Run from project root: bin\yolo_demo.exe
exit /b 0
