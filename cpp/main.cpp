#include <iostream>
#include <fstream>
#include <vector>

#include "TRT.h"

#include "WAV.h"
#include "STFT.h"
#include "mel.h"

int main() {

  std::string model_path("../../test.onnx");
  int batch_size = 1;

  // initialize TensorRT engine and parse ONNX model
  TRTUniquePtr<nvinfer1::ICudaEngine> engine{ nullptr };
  TRTUniquePtr<nvinfer1::IExecutionContext> context{ nullptr };
  parseOnnxModel(model_path, engine, context);

  // get sizes of input and output and allocate memory required for input data and for output data
  std::vector<nvinfer1::Dims> input_dims; // we expect only one input
  std::vector<nvinfer1::Dims> output_dims; // and one output
  std::vector<void*> buffers(engine->getNbBindings()); // buffers for input and output data

  std::cout << "engine->getNbBindings() : " << engine->getNbBindings()<<std::endl;


  // allocate data
  for (size_t i = 0; i < engine->getNbBindings(); ++i)
  {
    std::cout << "engine->getBindingDimensions("<<i<<").nbDims : " << engine->getBindingDimensions(i).nbDims<<std::endl;

    auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * batch_size * sizeof(float);  

    cudaMalloc(&buffers[i], binding_size);

    if (engine->bindingIsInput(i))
    {
      input_dims.emplace_back(engine->getBindingDimensions(i));
    }
    else
    {
      output_dims.emplace_back(engine->getBindingDimensions(i));
    }
  }
  if (input_dims.empty() || output_dims.empty())
  {
    std::cerr << "Expect at least one input and one output for network\n";
    return -1;
  }

  // inference
  context->enqueue(batch_size, buffers.data(), 0, nullptr);

  //void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size)
  // copy results from GPU to CPU
  std::vector<float> cpu_output(getSizeByDim(output_dims[0]) * batch_size);
  cudaMemcpy(cpu_output.data(), (float*)buffers[1], cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

  std::cout << "output : "<<std::endl;
  std::cout << cpu_output[0]<<std::endl;
  std::cout << cpu_output[1]<<std::endl;
  std::cout << cpu_output[2]<<std::endl;

  // free
  for (void* buf : buffers)
  {
    cudaFree(buf);
  }

  return 0;
}