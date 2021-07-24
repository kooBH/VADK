from __future__ import unicode_literals

import os

import pandas as pd
import youtube_dl

import logging

from tqdm import tqdm

prefix='https://youtu.be/'
output_root = '/home/data/kbh/AVTR/AVA-Speech-id/'

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

username = None
password = None

with open('../../key/youtube_account.txt') as f:
    username = f.readline().split()[0]
    password = f.readline().split()[0]
    print('username : ' + username)
    print('password : ' + password)

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
        #'preferredcodec': 'wav',
        'preferredcodec': 'mp3',
        'preferredquality': '192'
    }],
    'quiet':True,
    'ignoreerrors':True,
    'logger': MyLogger(),
    'username':username,
    'password':password,
    'age_limit': 40
    #'progress_hooks': [my_hook],
}

failed=False

if __name__ == '__main__':
    os.makedirs(output_root,exist_ok = True)

    if not failed :
        ava = pd.read_csv('ava_speech_labels_v1.csv',names=['id','start','end','label'])

        list_id = ava['id'].drop_duplicates()

        #for id in list_id.head() :
        for id in tqdm(list_id):
            # down_load id  as 16kHz Single, WAV
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([prefix + id])

    # run for failed data
    else :
        with open('retry.txt') as f :
            lines = f.readlines()

            for i in tqdm(lines) :
                id = i.split()[1]
                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([prefix + id])

