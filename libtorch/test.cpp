
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

constexpr double threshold = 0.7;

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
  int n_unit = -1;
  STFT process(1, frame, shift);
  mel mel_filter(rate, frame, n_mels);
  GPV vad("GPV.pt",n_mels);

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
  double* prob;

  for (auto path : std::filesystem::directory_iterator{ "../input" }) {
    int cnt_frame;

    std::string target(path.path().string());
    printf("processing : %s\n", target.c_str());
    WAV input;
    input.OpenFile(target.c_str());

    // Get param
    int ch = input.GetChannels();
    int n_sample = input.GetNumOfSamples();

    int n_unit = int(n_sample / shift + (n_sample%shift?1:0));
    printf("n_unit : %d\n",n_unit);

    // buffer alloc
    short* buf_tmp = new short[ch*shift];
    printf("channels : %d\n",ch);

    data = new double* [n_unit];
    for (int i = 0; i < n_unit; i++) {
      data[i] = new double[n_mels];
      memset(data[i], 0, sizeof(double) * n_mels);
    }

    prob = new double[n_unit];
    memset(prob, 0, sizeof(double) * (n_unit));

    int cnt_crop = 0;
    cnt_frame = 0;
    
    /* Crate Data Buffer */
    while (!input.IsEOF()) {
      length = input.ReadUnit(buf_tmp, shift * ch);

      // extract first channel only
      for (int i = 0; i < shift; i++)
        buf_in[i] = buf_tmp[ch * i];

      // stft
      process.stft(buf_in, spec);

      // mag
      for (int i = 0; i < nhfft; i++)
        mag[i] = std::sqrt(spec[2 * i]*spec[2*i] + spec[2 * i + 1]*spec[2*i+1]);

      // mel
      mel_filter.filter(mag, mel);

      // accumlate 
      memcpy(data[cnt_frame],mel,sizeof(double)*n_mels);
      cnt_frame++;
    }
    printf("INFO::feature extraction done. cnt_frame : %d\n",cnt_frame);

    vad.process(data, prob,n_unit);

    /* Post Process */



    /* Process output */
    input.Rewind();

    WAV output(ch, rate);
    //output path
    std::string path_o= "../output/" + target.substr(9, target.length() - 9);
    std::cout << path_o << std::endl;
    output.NewFile(path_o);


    cnt_frame = 0;
    while (!input.IsEOF()) {
      length = input.ReadUnit(buf_tmp, shift * ch);
      if (prob[cnt_frame] < threshold)
        memset(buf_tmp, 0, sizeof(short) * ch * shift);
      output.Append(buf_tmp, shift * ch);
      cnt_frame++;
      }
    input.Finish();
    output.Finish();

    for(int i=0;i<n_unit;i++)
      delete[] data[i];
    delete[] data;
    delete[] prob;
    delete[] buf_tmp;

    }
  printf("Done\n");

  delete[] buf_in;
  delete[] spec;
  delete[] mag;
  delete[] mel;

  return 0;


  }
#endif
