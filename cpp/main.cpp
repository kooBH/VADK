#include <torch/torch.h>
#include <torch/script.h>

#include <memory>
#include <vector>

int main(){
    torch::jit::script::Module module;

      try {
    module = torch::jit::load("../model.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n"<<e.what();
    return -1;
  }

    // Check for cuda
  if (torch::cuda::is_available()) {
    std::cout << "Moving model to GPU" << std::endl;
    module.to(at::kCUDA);
  }

  at::Tensor tensor_input;

  // load data

  // inference
  at::Tensor result = module.forward({tensor_input}).toTensor();


    return 0;    
}