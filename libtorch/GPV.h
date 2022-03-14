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
  inline GPV(std::string path, int n_mels);
  inline ~GPV();

  // input[n_unit][n_mels], prob[n_unit]
  inline void process(double** input, double* prob, int n_unit);
};


GPV::GPV(std::string path, int n_mels_) {
  n_mels = n_mels_;

  /* Model Loading*/
  try {
    std::cout << "GPV::loading : " << path << std::endl;
    module = torch::jit::load("C:/workplace/VADK/libtorch/build/GPV.pt");
    module.eval();

    //for(auto x : module.attributes())
    //  std::cout << x<<std::endl;
 //   module.to(torch::kCPU);
 //   module.eval();
  }
  catch (const c10::Error& e) {
    std::cerr << "ERROR::GPV:failed to load model\n" << e.what();
    exit(-1);
  }


}

GPV::~GPV(){
}


// input[n_unit][n_mels], prob[n_unit]
void GPV::process(double** input, double* prob,int n_unit_) {

  n_unit = n_unit_;
  n_mel_unit = n_mels * n_unit;
  /* Data alloc */
  data = new float[n_ch * n_mels * n_unit];  
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
      for (int j = 0; j < n_unit-1; j++) {
        for (int k = 0; k < n_mels; k++) {
          data[n_mel_unit + k * n_unit + j] = input[j+1][k] - input[j][k];
        }
      }
      break;

    // delta-delta
    case 2:
      for (int j = 0; j < n_unit-1; j++) {
        for (int k = 0; k < n_mels; k++) {
          data[2 * n_mel_unit + k * n_unit + j]
            = data[n_mel_unit + k * n_unit + j+1] - data[n_mel_unit + k*n_unit + j];
        }
      }
      break;

    default:
      printf("ERROR::GPV:n_ch > 3\n");
      exit(-1);
    }
  }

  // Debugging
  /*
  for (int t = 0; t < 2; t++) {
    printf("==== d %d ====\n",t);
    for (int c = 0; c < n_ch; c++) {
      for (int m = 0; m < 5; m++)
        printf("%.4e ",data[c*n_mel_unit+m*n_unit + t]);
      printf("\n");
    }
    printf("\n");
  }
  */

  try {
  //  printf("GPV::from_blob\n");
    //auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor_data = torch::from_blob(data, { 1, n_ch, n_mels, n_unit }, torch::kFloat32);
   //torch::Tensor tensor_data = torch::ones({ 1, n_ch, n_mels, n_unit }, torch::kFloat32);
   // std::cout << tensor_data[0][0][0][1] << std::endl;

    //std::cout << "tensor data: " << tensor_data.sizes() << std::endl;

   // printf("GPV::forward\n");
    torch::Tensor tensor_result = module.forward({ tensor_data }).toTensor();

   // std::cout << "tensor shape: " << tensor_result.sizes() << std::endl;

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

  delete[] data;

}


void _TEST_BLOB_DIM() {
  
  float* data;
  data = new float [2 * 3];

  int cnt = 0;
  printf("== host==\n");
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%d ",cnt);
      data[3 * i + j] = cnt++;
    }
    printf("\n");
  }

  torch::Tensor tensor_data = torch::from_blob(data, { 2,3 }, torch::kFloat32);

  printf("== tensor ==\n");
  std::cout << tensor_data << std::endl;
}

#endif