import pandas as pd
import numpy as np
import librosa as rosa
import soundfile as sf
import os

from tqdm import tqdm

import shutil

path_csv = 'unbalanced_keyboard.csv'

dir_input  = '/home/data/kbh/AVTR/Audioset_keyboard/'
dir_output = '/home/data/kbh/AVTR/Audioset_keyboard_typing/'


if __name__ == '__main__' :
    data = pd.read_csv(path_csv, names=['id','class1','class2','class3'])
    print(len(data))

    os.makedirs(dir_output , exist_ok=True)

    for idx in tqdm(data.index):
        item = data.loc[idx]
        if os.path.exists(dir_input + '/' + item.id[1:12]+'.wav')  \
        and type(item.class3) is float  \
        and item.class2 == 'Typing' \
        and pd.isna(item.class3) : 
            shutil.copy(dir_input + '/'+ item.id[1:12] + '.wav' ,dir_output)

        