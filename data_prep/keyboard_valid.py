import shutil
import os,glob



root_in = '/home/data/kbh/AVTR/Audioset_keyboard_typing/'
root_out = '/home/data/kbh/AVTR/Audioset_keyboard_valid'

list_target = [x for x in glob.glob(os.path.join(root_in,'*.wav'))] 

for path in list_target : 
    if not '@' in path:
        shutil.copy(path,root_out+'/'+path.split('/')[-1])