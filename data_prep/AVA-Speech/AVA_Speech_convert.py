from pydub import AudioSegment
import numpy as np
import os,glob 
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

sr = 16000

input_root =  '/home/data/kbh/AVTR/AVA-Speech-id/'
output_root = '/home/data/kbh/AVTR/AVA-Speech-wav/'

target_list = [x for x in glob.glob(os.path.join(input_root,'*.mp3'))]

def convert(idx):
    target_path = target_list[idx]
    target_name = target_path.split('/')[-1]
    target_id = target_name.split('.')[0]

    sound = AudioSegment.from_mp3(target_path)
    sound = sound.set_frame_rate(sr)
    sound = sound.set_channels(1)

    # 15 min ~ 30 min
    sound_arr = sound.get_array_of_samples()
    sound_arr = sound_arr[15*60*sr:30*60*sr]

    sound = sound._spawn(sound_arr)
    sound.export(output_root+target_id+'.wav', format="wav")

cpu_num = cpu_count()

os.makedirs(output_root,exist_ok=True)

arr = list(range(len(target_list)))
with Pool(cpu_num) as p:
    r = list(tqdm(p.imap(convert, arr), total=len(arr),ascii=True,desc='mp3 to 16kHz mono wav'))