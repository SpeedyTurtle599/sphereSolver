@echo off
setlocal enabledelayedexpansion

:: CUDA settings for RTX 4090 (Ada Lovelace)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set ARCH=90
set SMS=90

:: Compilation flags
set NVCC_FLAGS=^
    -o sphere_flow.exe ^
    -arch=compute_%ARCH% ^
    -code=sm_%SMS% ^
    -use_fast_math ^
    -O3 ^
    -lineinfo ^
    --ptxas-options=-v ^
    --default-stream per-thread ^
    -Xcompiler "/O2 /MD /arch:AVX2"

:: Check CUDA compiler
if exist "%CUDA_PATH%\bin\nvcc.exe" (
    echo Found CUDA 12.6 compiler
) else (
    echo Cannot find CUDA compiler
    exit /b 1
)

:: Compile
echo Compiling for compute_%ARCH% and sm_%SMS%...
"%CUDA_PATH%\bin\nvcc.exe" %NVCC_FLAGS% main.cu

if %ERRORLEVEL% EQU 0 (
    echo Compilation successful
) else (
    echo Compilation failed
    exit /b 1
)

endlocal