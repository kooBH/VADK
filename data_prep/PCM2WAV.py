import os,glob 
import tqdm
import wave

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


data_root = '/home/data/kbh/AVTR/Kspon'
output_root = '/home/data/kbh/AVTR/Kspon_WAV'

target_list = [x for x in glob.glob(os.path.join(data_root, '*','*.pcm'))]

ch = 1
samplebytes = 2
samplerates = 16000

def convert(idx):
    target_path = target_list[idx]

    target_name = target_path.split('/')[-1]
    target_name = target_name.split('.')[0]

    # read
    with open(target_path,'rb') as pcm :
        data = pcm.read()

    # save        
    with wave.open(output_root + '/' + target_name + '.wav','wb' ) as wav :
        wav.setparams((ch,samplebytes,samplerates, 0, 'NONE','NONE'))
        wav.writeframes(data)

os.makedirs(output_root,exist_ok=True)

cpu_num = cpu_count()

arr = list(range(len(target_list)))
with Pool(cpu_num) as p:
    r = list(tqdm(p.imap(convert, arr), total=len(arr),ascii=True,desc='PCM2WAV'))