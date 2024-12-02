#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <optional>
#include <opencv2/opencv.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>
#define ASSERT(condition) assert(condition)
using namespace nvinfer1;
using namespace std;

void setAllDynamicRanges(nvinfer1::INetworkDefinition* network, float inRange, float outRange)
{
    // Ensure that all layer inputs have a scale.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbInputs(); j++)
        {
            nvinfer1::ITensor* input{layer->getInput(j)};
            // Optional inputs are nullptr here and are from RNN layers.
            if (input != nullptr && !input->dynamicRangeIsSet())
            {
                ASSERT(input->setDynamicRange(-inRange, inRange));
            }
        }
    }

    // Ensure that all layer outputs have a scale.
    // Tensors that are also inputs to layers are ingored here
    // since the previous loop nest assigned scales to them.
    for (int i = 0; i < network->getNbLayers(); i++)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); j++)
        {
            nvinfer1::ITensor* output{layer->getOutput(j)};
            // Optional outputs are nullptr here and are from RNN layers.
            if (output != nullptr && !output->dynamicRangeIsSet())
            {
                // Pooling must have the same input and output scales.
                if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                {
                    ASSERT(output->setDynamicRange(-inRange, inRange));
                }
                else
                {
                    ASSERT(output->setDynamicRange(-outRange, outRange));
                }
            }
        }
    }
}


class OutputAllocator : public nvinfer1::IOutputAllocator {
public:
    void* reallocateOutputAsync(const char* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override
    {
        if (currentMemory != nullptr)
        {
            cudaFree(currentMemory);
        }

        void* ptr = nullptr;
        cudaMalloc(&ptr, size);
        return ptr;
    }

    void notifyShape(const char* tensorName, const nvinfer1::Dims& dims) noexcept override
    {
        // 텐서의 새로운 shape에 대한 알림 처리
    }
};

// Logger for TensorRT
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
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
} gLogger;

// Helper function to read file contents
std::vector<char> readFile(const std::string& filepath)
{
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file)
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

// Save engine to file
void saveEngine(const std::string& filepath, IHostMemory* engine)
{
    std::ofstream file(filepath, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Failed to save engine to file: " + filepath);
    }
    file.write(reinterpret_cast<const char*>(engine->data()), engine->size());
}

std::pair<double, double> benchmarkEngine(ICudaEngine* engine, int batchSize = 1, int warmupRuns = 10, int benchmarkRuns = 50) {
    // Execution Context 생성
    IExecutionContext* context = engine->createExecutionContext();
    if (!context)
    {
        throw std::runtime_error("Execution Context 생성에 실패했습니다.");
    }

    // 텐서 이름을 기반으로 입력과 출력 텐서 식별
    std::vector<void*> bindings(engine->getNbIOTensors(), nullptr);
    std::vector<size_t> bindingSizes(engine->getNbIOTensors());

    for (int i = 0; i < engine->getNbIOTensors(); ++i)
    {
        const char* tensorName = engine->getIOTensorName(i);
        Dims dims = engine->getTensorShape(tensorName);
        size_t size = batchSize;
        for (int j = 0; j < dims.nbDims; ++j)
        {
            size *= dims.d[j];
        }
        size *= sizeof(float); // 입력과 출력을 float로 가정
        bindingSizes[i] = size;

        void* buffer;
        cudaMalloc(&buffer, size);
        bindings[i] = buffer;

        // 텐서 주소 설정
        if (!context->setTensorAddress(tensorName, buffer))
        {
            throw std::runtime_error(std::string("텐서 주소를 설정하는 데 실패했습니다: ") + tensorName);
        }
    }

    // CUDA 스트림 생성
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Output Allocator 생성
    OutputAllocator outputAllocator;

    // 워밍업 실행
    for (int i = 0; i < warmupRuns; ++i)
    {
        if (!context->enqueueV3(stream))
        {
            throw std::runtime_error("워밍업 실행 중 오류가 발생했습니다.");
        }
    }

    // 스트림 동기화
    cudaStreamSynchronize(stream);

    // 벤치마크 실행
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchmarkRuns; ++i)
    {
        if (!context->enqueueV3(stream))
        {
            throw std::runtime_error("벤치마크 실행 중 오류가 발생했습니다.");
        }
    }
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();

    // 성능 계산
    double totalTime = std::chrono::duration<double, std::milli>(end - start).count();
    double avgLatency = totalTime / benchmarkRuns;
    double throughput = (1000.0 / avgLatency) * batchSize;

    // 할당된 메모리 해제
    for (void* buffer : bindings)
    {
        cudaFree(buffer);
    }
    delete context;
    cudaStreamDestroy(stream);

    return {avgLatency, throughput};
}

std::vector<std::string> getImagePathsFromFolder(const std::string& folderPath) {
    std::vector<std::string> imagePaths;
    namespace fs = std::filesystem;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                const auto& path = entry.path();
                // 이미지 파일 확장자 필터링
                if (path.extension() == ".jpg" || path.extension() == ".png" || path.extension() == ".bmp") {
                    imagePaths.push_back(path.string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading folder: " << folderPath << ". " << e.what() << std::endl;
    }

    return imagePaths;
}

// Convert ONNX to TensorRT Engine
std::pair<double, double> convertAndBenchmark(const std::string& onnxFile, const std::string& precision, bool bDLA=false)
{
    // Create builder, network, and config
    std::string dla = "DLA";
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    INetworkDefinition* network = builder->createNetworkV2(0);

    // Parse ONNX model
    auto parser = nvonnxparser::createParser(*network, gLogger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(ILogger::Severity::kERROR)))
    {
        throw std::runtime_error("Failed to parse ONNX model.");
    }

    // Set optimization profiles
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB
    if (precision == "FP16" && builder->platformHasFastFp16())
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision == "INT8" && builder->platformHasFastInt8())
    {
        config->setFlag(BuilderFlag::kINT8);
        Dims inputDims = network->getInput(0)->getDimensions();
        setAllDynamicRanges(network, 127.0F, 127.0F);
    }

    // Configure DLA if available
    int dlaCore = -1;
    if (bDLA && builder->getNbDLACores() > 0)
    {
        dlaCore = 0; // 첫 번째 DLA 코어 사용
        config->setFlag(BuilderFlag::kGPU_FALLBACK);
        config->setDLACore(dlaCore);
    }
    else
    {
        dla = "";
        if(bDLA)
        {
            std::cout << "DLA is not supported." << std::endl;
            return {-1, -1};
        }
    }

    // Build engine
    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    if (!engine)
    {
        throw std::runtime_error("Failed to build TensorRT engine.");
    }

    // Save engine
    std::string outputEngineFile = onnxFile.substr(0, onnxFile.find_last_of('.')) + "_" + dla + precision + ".engine";
    saveEngine(outputEngineFile, engine);
    std::cout << "Saved TensorRT engine to " << outputEngineFile << std::endl;

    // Benchmark engine
    auto runtime = createInferRuntime(gLogger);
    auto cudaEngine = runtime->deserializeCudaEngine(engine->data(), engine->size());
    auto result = benchmarkEngine(cudaEngine);

    // Cleanup
    delete engine;
    delete runtime;
    delete config;
    delete network;
    delete builder;

    return result;
}

int main(int argc, char** argv)
{
    if (argc < 1)
    {
        std::cerr << "Usage: " << argv[0] << " <onnx_file>" << std::endl;
        return 1;
    }

    std::string onnxFile = argv[1];
    // std::string onnxFile = "/home/a2mind/Data/Code/CCTI/gitlab/ai-dev/tensorrt/tensorrt/yolov5m.onnx";

    try
    {
        std::cout << "====================================================================" << std::endl;
        auto result_32 = convertAndBenchmark(onnxFile, "FP32", false);
        std::cout << "====================================================================" << std::endl;
        auto result_16 = convertAndBenchmark(onnxFile, "FP16", false);
        std::cout << "====================================================================" << std::endl;
        auto result_8 = convertAndBenchmark(onnxFile, "INT8", false);
        std::cout << "====================================================================" << std::endl;



        std::cout << "====================================================================" << std::endl;
        auto dla_result_32 = convertAndBenchmark(onnxFile, "FP32", true);
        std::cout << "====================================================================" << std::endl;
        auto dla_result_16 = convertAndBenchmark(onnxFile, "FP16", true);
        std::cout << "====================================================================" << std::endl;
        auto dla_result_8 = convertAndBenchmark(onnxFile, "INT8", true);
        std::cout << "====================================================================" << std::endl;

        std::cout << "[FP32] Latency(Avg): " << result_32.first << " ms" << std::endl;
        std::cout << "[FP32] Throughout: " << result_32.second << " 추론/초" << std::endl;

        std::cout << "[FP16] Latency(Avg): " << result_16.first << " ms" << std::endl;
        std::cout << "[FP16] Throughout: " << result_16.second << " 추론/초" << std::endl;

        std::cout << "[INT8] Latency(Avg): " << result_8.first << " ms" << std::endl;
        std::cout << "[INT8] Throughout: " << result_8.second << " 추론/초" << std::endl;

        std::cout << "[DLA][FP32] Latency(Avg): " << dla_result_32.first << " ms" << std::endl;
        std::cout << "[DLA][FP32] Throughout: " << dla_result_32.second << " 추론/초" << std::endl;

        std::cout << "[DLA][FP16] Latency(Avg): " << dla_result_16.first << " ms" << std::endl;
        std::cout << "[DLA][FP16] Throughout: " << dla_result_16.second << " 추론/초" << std::endl;

        std::cout << "[DLA][INT8] Latency(Avg): " << dla_result_8.first << " ms" << std::endl;
        std::cout << "[DLA][INT8] Throughout: " << dla_result_8.second << " 추론/초" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

