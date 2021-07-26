import librosa
import soundfile as sf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')
import os,glob

sr = 16000

root_input = '/home/data/kbh/AVTR/hospital_noise/'
root_output = '/home/data/kbh/AVTR/hospital_16kHz/'

list_target = [x for x in glob.glob(os.path.join(root_input,'*.wav'))]

def resample(idx):
    path_target = list_target[idx]
    name_target = path_target.split('/')[-1]

    data, _ = librosa.load(path_target,sr=sr)
    sf.write(os.path.join(root_output,name_target),data,sr)

if __name__=='__main__': 
    os.makedirs(root_output,exist_ok=True)

    cpu_num = cpu_count()

    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(resample, arr), total=len(arr),ascii=True,desc='Convert to 16kHz'))

