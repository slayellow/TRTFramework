#include "CTRTEngine.h"

CTRTEngine::CTRTEngine()
{
    mLogger = std::make_unique<CTRTLogger>();
}

CTRTEngine::~CTRTEngine()
{

}

bool CTRTEngine::Create(std::string& path)
{
    mPath = path;

    std::string::size_type extIdx = mPath.rfind('.');
    if(extIdx == std::string::npos)
    {
        return false;
    }

    std::string extension = mPath.substr(extIdx + 1);
    std::string save_path = mPath.substr(0, extIdx);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if(extension == "onnx")
    {
        return BuildONNX(save_path);
    }
    else if(extension == "trt")
    {
        return LoadEngine();
    }
    else
    {
        std::string error = extension + " is not supported!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }
}

bool CTRTEngine::BuildONNX(std::string& savePath)
{
    mBuilder = std::unique_ptr<nvinfer1::IBuilder, TRTDestroyer>(nvinfer1::createInferBuilder(*mLogger));
    if(!mBuilder)
    {
        std::string error = "1. nvinfer1::IBuilder is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mNetwork = std::unique_ptr<nvinfer1::INetworkDefinition, TRTDestroyer>(mBuilder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if(!mNetwork)
    {
        std::string error = "2. nvinfer1::INetworkDefinition is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mParser = std::unique_ptr<nvonnxparser::IParser, TRTDestroyer>(nvonnxparser::createParser(*mNetwork, *mLogger));
    if(!mParser)
    {
        std::string error = "3. nvonnxparser::IParser is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    auto loaded = mParser->parseFromFile(mPath.c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO));
    if(!loaded)
    {
        std::string error = "4. ONNX File is not loaded!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mConfig = std::unique_ptr<nvinfer1::IBuilderConfig, TRTDestroyer>(mBuilder->createBuilderConfig());
    if(!mConfig)
    {
        std::string error = "5. nvinfer1::IBuilderConfig is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);
    if(platformSupprotFP16())
    {
        std::cout << "FP16 is supported!" << std::endl;
        mConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if(platformSupportInt8())
    {
        std::cout << "INT8 is supported!" << std::endl;
        mConfig->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

    mSerializedNetwork = std::unique_ptr<nvinfer1::IHostMemory, TRTDestroyer>(mBuilder->buildSerializedNetwork(*mNetwork, *mConfig));
    if(!mSerializedNetwork)
    {
        std::string error = "6. Network is not Serialized!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mRuntime = std::unique_ptr<nvinfer1::IRuntime, TRTDestroyer>(nvinfer1::createInferRuntime(*mLogger));
    if(!mRuntime)
    {
        std::string error = "7. nvinfer1::IRuntime is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mEngine = std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroyer>(mRuntime->deserializeCudaEngine(mSerializedNetwork->data(), mSerializedNetwork->size()));
    if(!mEngine)
    {
        std::string error = "8. Network is not Deserialized!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    savePath = savePath + ".trt";
    std::ofstream engineFile(savePath, std::ios::binary);
    if(!engineFile)
    {
        std::string error = "Cannot Open Engine File : " + savePath;
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    engineFile.write(static_cast<char*>(mSerializedNetwork->data()), mSerializedNetwork->size());

    if(!engineFile.fail())
    {
        mLogger->log(nvinfer1::ILogger::Severity::kINFO, "Engine file is Saved!!");
    }
    else
    {
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, "Engine file is not Saved!!");
        return false;
    }

    mContext = std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroyer>(mEngine->createExecutionContext());
    if (!mContext)
    {
        std::string error = "9. nvinfer1::IExecutionContext is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    std::cout << float(mEngine->getDeviceMemorySizeV2() / 1024.0 / 1024.0) <<"MB Memory" << std::endl;

    return true;
}

bool CTRTEngine::LoadEngine()
{
    std::ifstream engineFile(mPath, std::ifstream::binary);
    if(!engineFile.is_open())
    {
        std::string error = "1. Cannot Open Engine File : " + mPath;
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    auto const start_pos = engineFile.tellg();

    engineFile.ignore(std::numeric_limits<std::streamsize>::max());
    size_t filesize = engineFile.gcount();
    engineFile.seekg(start_pos);
    std::unique_ptr<char[]> engineBuf(new char[filesize]);
    engineFile.read(engineBuf.get(), filesize);

    mRuntime.reset(nvinfer1::createInferRuntime(*mLogger));
    if(!mRuntime)
    {
        std::string error = "2. nvinfer1::IRuntime is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mEngine.reset(mRuntime->deserializeCudaEngine((void*)engineBuf.get(), filesize));
    if(!mEngine)
    {
        std::string error = "3. Network is not Deserialized!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    mContext = std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroyer>(mEngine->createExecutionContext());
    if (!mContext)
    {
        std::string error = "4. nvinfer1::IExecutionContext is not created!!";
        mLogger->log(nvinfer1::ILogger::Severity::kERROR, error.c_str());
        return false;
    }

    std::cout << float(mEngine->getDeviceMemorySizeV2() / 1024.0 / 1024.0) <<"MB Memory" << std::endl;

    return true;
}

void CTRTEngine::CreateDeviceIOBuffer()
{

}
