import glob
import shutil
import os
fileList = glob.glob('./attachments/texts/*.txt')

os.makedirs('./database/', exist_ok=True)

for file in fileList:
    fpath = file.split('_')[0].replace('./attachments/texts\\','./database/')
    newfilename = '_'.join(file.split('_')[1:])
    os.makedirs(fpath, exist_ok=True)
    shutil.copy2(file, os.path.join(fpath, newfilename))