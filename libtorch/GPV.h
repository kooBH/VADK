#ifndef _H_GPV_
#define _H_GPV_

/*
GPV module 

mel 40 -> GPV -> prob

*/


#include <torch/torch.h>
#include <torch/script.h>
#include <string>


class GPV {
private :

  // Var
  int n_ch = 3; // mel, delta, delta-delta
  int n_mels;
  int n_unit;
  // [C, F, T];
  float * data; 



  torch::jit::script::Module module;


  // Aux var
  int n_mel_unit;


public :
  inline GPV(std::string path, int n_mels, int n_unit);
  inline ~GPV();

  // input[n_unit][n_mels], prob[n_unit]
  inline void process(double** input, double* prob);

};


GPV::GPV(std::string path, int n_mels_, int n_unit_) {
  n_mels = n_mels_;
  n_unit = n_unit_;
  n_mel_unit = n_mels * n_unit;

  /* Model Loading*/
  try {
    std::cout << "GPV::loading : " << path << std::endl;
    module = torch::jit::load("C:/workplace/VADK/libtorch/build/GPV.pt");

    //for(auto x : module.attributes())
    //  std::cout << x<<std::endl;
 //   module.to(torch::kCPU);
 //   module.eval();
  }
  catch (const c10::Error& e) {
    std::cerr << "ERROR::GPV:failed to load model\n" << e.what();
    exit(-1);
  }

  /* Data alloc */
  data = new float[n_ch * n_mels * n_unit];  

}

GPV::~GPV(){
  delete[] data;
}


// input[n_unit][n_mels], prob[n_unit]
void GPV::process(double** input, double* prob) {
  memset(data, 0, sizeof(float) * n_ch * n_mels * n_unit);

 // printf("GPV::input\n");
  for(int i = 0; i < n_ch; i++){
    switch (i) {
      // mel
    case 0 :
      for (int j = 0; j < n_unit; j++) {
        for (int k = 0; k < n_mels; k++) {
          data[k * n_unit + j] = static_cast<float>(input[j][k]);
        }
      }
      break;

    // delta
    case 1:
      for (int j = 0; j < n_unit; j++) {
        for (int k = 0; k < n_mels - 1; k++) {
          data[n_mel_unit + k * n_unit + j] = input[j][k + 1] - input[j][k];
        }
      }
      break;

    // delta-delta
    case 2:
      for (int j = 0; j < n_unit; j++) {
        for (int k = 0; k < n_mels - 2; k++) {
          data[2 * n_mel_unit + k * n_unit + j]
            = data[n_mel_unit + (k + 1) * n_unit + j] = data[n_mel_unit + (k)*n_unit + j];
        }
      }
      break;

    default:
      printf("ERROR::GPV:n_ch > 3\n");
      exit(-1);
    }
  }
  try {
  //  printf("GPV::from_blob\n");
    //auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor_data = torch::from_blob(data, { 1, n_ch, n_mels, n_unit }, torch::kFloat32);

    //at::Tensor tensor_data;


    //std::cout << tensor_data.data() << std::endl;


   // printf("GPV::forward\n");
    //tensor_result = module.forward({ tensor_data }).toTensor();
    torch::Tensor tensor_result = module.forward({ tensor_data }).toTensor();

    //std::cout << "tensor shape: " << tensor_result.sizes() << std::endl;

    // [1, 1, 50]
    float* result = tensor_result.data<float>();


    for (int x = 0; x < tensor_result.sizes()[2]; ++x){
      //std::cout << (*result) << " ";
      prob[x] = (*result++);
    }
    //std::cout << std::endl;
  }
  catch (const c10::Error& e) {
    std::cerr << "ERROR::GPV\n" << e.what();
    exit(-1);
  
  }



  /* ex
  auto maxResult = result.max(1);
    auto maxIndex = std::get<1>(maxResult).item<float>();
    auto maxOut = std::get<0>(maxResult).item<float>();
  */

}

#endif