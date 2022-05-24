#ifndef _H_GPV_
#define _H_GPV_

/*
GPV module 

mel 40 -> GPV -> prob

*/

//#define _USE_ONNX_
#define _USE_TORCH_

#ifdef _USE_TORCH_
#include <torch/torch.h>
#include <torch/script.h>
#endif 

#ifdef _USE_ONNX_
#include <onnxruntime_cxx_api.h>
#endif

#include <string>

class GPV {
private :

  // Var
  int n_ch = 3; // mel, delta, delta-delta
  int n_mels;
  int n_unit;
  // [C, F, T];
  float * data; 

#ifdef _USE_TORCH_
  torch::jit::script::Module module;
#endif
#ifdef _USE_ONNX_
  Ort::Env env;
  Ort::SessionOptions session_options;
  Ort::Session session;

  Ort::AllocatorWithDefaultOptions allocator;
#endif

  // Aux var
  int n_mel_unit;

public :
#ifdef _USE_TORCH_
  inline GPV(std::string path, int n_mels);
#endif
#ifdef _USE_ONNX_
  std::vector<int64_t> shape;
  inline GPV(wchar_t* path, int n_mels);
#endif
  inline ~GPV();

  // input[n_unit][n_mels], prob[n_unit]
  inline void process(double** input, double* prob, int n_unit);
};


#ifdef _USE_TORCH_
GPV::GPV(std::string path, int n_mels_) 
{
  n_mels = n_mels_;

  /* Model Loading*/

  try {
    std::cout << "GPV::loading : " << path << std::endl;
    module = torch::jit::load(path);
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
#endif

#ifdef _USE_ONNX_
GPV::GPV(wchar_t* path, int n_mels_)
:session(env,path, session_options){

  n_mels = n_mels_;

}

#endif

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
#ifdef _USE_TORCH_
  try {
  //  printf("GPV::from_blob\n");
    //auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor tensor_data = torch::from_blob(data, { 1, n_ch, n_mels, n_unit }, torch::kFloat32);
    torch::Tensor tensor_result = module.forward({ tensor_data }).toTensor();


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
#endif
#ifdef _USE_ONNX_
  shape.push_back(1);
  shape.push_back(n_ch);
  shape.push_back(n_mels);
  shape.push_back(n_unit);

  printf("GPV::n_unit:%d | %d %d\n",n_unit,n_ch,n_mels);

  const char* input_names[] = { "input" };
  const char* output_names[] = { "output" };

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, data, (1*n_ch*n_mels*n_unit), shape.data(), shape.size());

  auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

  float* result = output_tensors.front().GetTensorMutableData<float>();

  for (int x = 0; x < n_unit; ++x) {
    //std::cout << (*result) << " ";
    prob[x] = (*result++);
  }

  shape.clear();

#endif

  


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
#ifdef _USE_TORCH_
  torch::Tensor tensor_data = torch::from_blob(data, { 2,3 }, torch::kFloat32);
  printf("== tensor ==\n");
  std::cout << tensor_data << std::endl;
#endif
}

#endif