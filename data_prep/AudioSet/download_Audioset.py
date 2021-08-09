from __future__ import unicode_literals

import os
import pandas as pd
import youtube_dl
import librosa
import numpy as np
import soundfile as sf

import logging

from tqdm import tqdm

prefix='https://youtu.be/'
output_root = '/home/data/kbh/AVTR/Audioset_keyboard/'

class MyLogger(object):
    def __init__(self):
        super(MyLogger,self)
        logging.basicConfig(filename='download.log')

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        logging.error(msg + ' : ' + id)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')
    elif d['status'] == 'downloading' :
        if 'total_bytes' in d :
            pass#print("Downloading : " + str(round( (d['downloaded_bytes'] / d['total_bytes'] )* 100 , 2) ) + "%")
        else :
            pass#print("Downloading : But couldn't get total_bytes" )

ydl_opts = {
    'prefer_ffmpeg':True,
	'format': 'bestaudio/best',
     #파일이름 default %(title)s-%(id)s.%(ext)s
     #'outtmpl':'%(title)s.',
    #'outtmpl':'downloads/%(title)s.%(ext)s',
    #'outtmpl':'/home/data/kbh/AVTR/AVA-Speech/%(title)s.%(ext)s',
    'outtmpl': output_root+'/%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        #'preferredcodec': 'mp3',
        'preferredquality': '192'
    }],
    'quiet':True,
    'ignoreerrors':True,
    'logger': MyLogger(),
    #iusername':username,
    #'password':password,
    'age_limit': 40
    #'progress_hooks': [my_hook],
}

if __name__ == '__main__':
    os.makedirs(output_root,exist_ok = True)

    data = pd.read_csv('unbalanced_keyboard.csv',names=['id','class1','class2','class3'])

    list_name = data['id'].tolist()
    list_id = [x[1:12] for x in list_name]

    list_start = [x.split('_')[-2] for x in list_name]
    list_end = [x.split('_')[-1] for x in list_name]

    #for id in list_id.head() :
    for idx in tqdm(range(len(list_id))):
        id = list_id[idx]
        start = int(list_start[idx])*16000
        end   = int(list_end[idx])*16000

        # down_load id  as 16kHz Single, WAV
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([prefix + id])
        
        # resample as 16kHz
        if os.path.exists(output_root+'/'+id+'.wav') :
            raw,_ = librosa.load(output_root+'/'+id+'.wav',sr=16000)
            sample = raw[start:end]
            # normalization
            sample = sample/np.max(np.abs(sample))
            sf.write(output_root+'/'+id+'.wav',sample,16000)
