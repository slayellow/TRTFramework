#ifndef CTRTENGINE_H
#define CTRTENGINE_H

#include <memory>
#include <algorithm>
#include <fstream>

#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include "common.h"

class CTRTEngine
{
public:
    CTRTEngine();
    ~CTRTEngine();

    bool Create(std::string& path);

private:
    bool BuildONNX(std::string& savePath);
    bool LoadEngine();
    void CreateDeviceIOBuffer();

    std::unique_ptr<CTRTLogger>                                     mLogger;

    // Build Variable
    std::unique_ptr<nvinfer1::IBuilder, TRTDestroyer>               mBuilder;
    std::unique_ptr<nvinfer1::INetworkDefinition, TRTDestroyer>     mNetwork;
    std::unique_ptr<nvonnxparser::IParser, TRTDestroyer>            mParser;
    std::unique_ptr<nvinfer1::IBuilderConfig, TRTDestroyer>         mConfig;
    std::unique_ptr<nvinfer1::IHostMemory, TRTDestroyer>            mSerializedNetwork;

    // Runtime Variable
    std::unique_ptr<nvinfer1::IRuntime, TRTDestroyer>               mRuntime;
    std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroyer>            mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroyer>      mContext;

    std::vector<void*> m_bindings; // pointers to input and output data
    std::vector<size_t> m_binding_sizes; // sizes of input and output data
    std::vector<nvinfer1::Dims> m_binding_dims; // dimensions of input and output data
    std::vector<nvinfer1::DataType> m_binding_types; // data types of input and output data
    std::vector<std::string> m_binding_names; // names of input and output data

    // .onnx, .engine File Path
    std::string                                                     mPath;
};

#endif // CTRTENGINE_H
