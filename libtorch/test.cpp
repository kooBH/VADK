
#include "STFT.h"
#include "WAV.h"
#include "mel.h"
#include "GPV.h"

#include <string>
#include <filesystem>

/* Set Parameter of Input */
constexpr int rate = 16000;
constexpr int frame = 512;
constexpr int shift = 128;
constexpr int n_mels = 40;
constexpr int n_unit = 50;

constexpr double threshold = 0.5;

//#define TEST

#ifdef TEST
int main() {
  _TEST_BLOB_DIM();
  /*
  double** data;
  data = new double*[n_unit];
  for (int i = 0; i < n_unit; i++) {
    data[i] = new double[n_mels];

    for (int j = 0; j < n_mels; j++)
      data[i][j] = 1.0;
  }

  double* prob;
  prob = new double[n_unit];
  memset(prob, 0, sizeof(double) * (n_unit));

  GPV vad("GPV.pt",n_mels,n_unit);
  vad.process(data, prob);

  for (int i = 0; i < n_unit; i++)
    printf("%lf ",prob[i]);
  printf("\n");
  */

  return 0;
}
#endif

#ifndef TEST

int main() {
  /* Define Algorithm Class here */
  int length;
  STFT process(1, frame, shift);
  mel mel_filter(rate, frame, n_mels);
  GPV vad("GPV.pt",n_mels,n_unit);

  int nhfft = int(frame / 2 + 1);

  short* buf_in;
  buf_in = new short[shift];

  double* spec;
  spec = new double[frame + 2];
  memset(spec, 0, sizeof(double) * (frame + 2));

  double* mag;
  mag = new double[frame/2 + 1];
  memset(mag, 0, sizeof(double) * (frame/2 + 1));

  double* mel;
  mel = new double[n_mels];
  memset(mel, 0, sizeof(double) * (n_mels));

  double** data;
  data = new double*[n_unit];
  for (int i = 0; i < n_unit; i++) {
    data[i] = new double[n_mels];
    memset(data[i], 0, sizeof(double) * n_mels);
  }

  double* prob;
  prob = new double[n_unit];
  memset(prob, 0, sizeof(double) * (n_unit));

  for (auto path : std::filesystem::directory_iterator{ "../input" }) {
    int cnt_time=0;
    std::string target(path.path().string());
    printf("processing : %s\n", target.c_str());
    WAV input;
    input.OpenFile(target.c_str());
    
    int ch = input.GetChannels();
    short* buf_tmp = new short[ch*shift];
    printf("channels : %d\n",ch);

    short *buf_wav = new short[ch * shift * n_unit];
    memset(buf_wav, 0, sizeof(short) * ch * shift * n_unit);

    WAV output(ch,rate);
  
    //output path
    std::string path_o= "../output/" + target.substr(9, target.length() - 9);
    std::cout << path_o << std::endl;
    output.NewFile(path_o);

    int cnt_crop = 0;
    
    while (!input.IsEOF()) {
      length = input.ReadUnit(buf_tmp, shift * ch);

      // extract first channel only
      for (int i = 0; i < shift; i++)buf_in[i] = buf_tmp[ch * i];
      //for (int i = 0; i < shift; i++)buf_in[i] = 1000;



      // stft
      process.stft(buf_in, spec);
      //mag
      for (int i = 0; i < nhfft; i++) {
        mag[i] = std::sqrt(spec[2 * i]*spec[2*i] + spec[2 * i + 1]*spec[2*i+1]);
      }


      //mel
      mel_filter.filter(mag, mel);


      /*
      printf("===== %d =====\n",cnt_time);
      for (int i = 0; i < 20; i++)printf("%lf ", mel[i]);
      printf("\n");
      if (cnt_time > 10)
        exit(0);
      */


      //if (cnt_time > 5) {
      if (false) {
        printf("raw : "); for (int i = 0; i < shift; i++)printf("%d ", buf_in[i]); printf("\n");
        printf("spec : "); for (int i = 0; i < frame + 2; i++)printf("%lf ", spec[i]); printf("\n");
        printf("mag : "); for (int i = 0; i < nhfft; i++)printf("%lf ", mag[i]); printf("\n");
        printf("mel : "); for (int i = 0; i < n_mels; i++)printf("%lf ", mel[i]); printf("\n");
        exit(-1);
      }

      // accumlate 
      memcpy(data[cnt_time],mel,sizeof(double)*n_mels);
      memcpy(&buf_wav[cnt_time * ch * shift], buf_tmp, sizeof(short) * ch * shift);

     // printf("cnt_time : %d\n",cnt_time);
      cnt_time++;
      if (cnt_time == n_unit) {
        cnt_time = 0;

        /*
        cnt_crop++;
        if (cnt_crop == 2)
          break;
        */

       // printf("vad process\n");
        /* Run Process here */
        vad.process(data, prob);

        /* Crop wav */

        for (int i = 0; i < n_unit; i++) {
          if (prob[i] <= threshold)
            memset(&buf_wav[i * shift * ch], 0, sizeof(short) * shift * ch);
        }
        output.Append(buf_wav, shift * ch*n_unit);
      }
    }
    printf("Done\n");
    delete[] buf_wav;
    delete[] buf_tmp;
    input.Finish();
    output.Finish();
  }

  delete[] buf_in;
  delete[] spec;
  delete[] mag;
  delete[] mel;
  for(int i=0;i<n_unit;i++)
    delete[] data[i];
  delete[] data;
  delete[] prob;
  

  return 0;
}

#endif
