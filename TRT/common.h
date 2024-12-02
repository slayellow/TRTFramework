#ifndef COMMON_H
#define COMMON_H

#include <iostream>

#include <cuda_runtime.h>
#include <NvInfer.h>

struct TRTDestroyer
{
    template <typename T>
    void operator()(T* obj) const
    {
        delete obj;
    }
};

static bool platformSupprotFP16()
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        // 에러 처리
        return false;
    }

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    // Compute Capability 5.3 이상부터 FP16 지원
    if (major > 5 || (major == 5 && minor >= 3))
    {
        return true;
    }
    else
    {
        return false;
    }
}

static bool platformSupportInt8()
{
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
        // 에러 처리
        return false;
    }

    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

    // Compute Capability 6.1 이상부터 INT8 지원
    if (major > 6 || (major == 6 && minor >= 1))
    {
        return true;
    }
    else
    {
        return false;
    }
}

class CTRTLogger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        std::string start;
        std::string colorCode;
        const std::string resetColor = "\033[0m";

        switch(severity)
        {
        case nvinfer1::ILogger::Severity::kERROR:
            start = "ERROR          : ";
            colorCode = "\033[1;31m";
            break;
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            start = "INTERNAL_ERROR : ";
            colorCode = "\033[1;35m";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            start = "INFO           : ";
            colorCode = "\033[1;32m";
            break;
        case nvinfer1::ILogger::Severity::kVERBOSE:
            start = "VERBOSE        : ";
            colorCode = "\033[1;34m";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            start = "WARNING        : ";
            colorCode = "\033[1;33m";
            break;
        }
        if(severity <= nvinfer1::ILogger::Severity::kINFO)
        {
            std::cerr << colorCode << "[TRT] " << start << msg << resetColor << std::endl;
        }
    }
};



#endif // COMMON_H
