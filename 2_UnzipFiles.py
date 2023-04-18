import os
import pathlib
import time
import zipfile
from natsort import natsorted, ns

while True:
    ForRemove = []
    for file in os.listdir("./attachments"):
        if pathlib.Path(file).suffix == '.zip':
            filepath = os.path.join("./attachments", file)
            ForRemove.append(filepath)
            with zipfile.ZipFile(filepath, 'r') as zf:
                zipInfo = zf.infolist()
                for member in zipInfo:
                    try:
                        member.filename = member.filename.encode('cp437').decode('euc-kr')
                    except:
                        pass
                zipInfo = [i for i in zipInfo if i.filename.split('.')[-1] != 'lnk']
                zipInfo = natsorted(zipInfo, key=lambda member:member.filename)
                for i, member in enumerate(zipInfo):
                    member.filename = f"{file.split('.')[0]}_{str(i).zfill(2)}.{member.filename.split('.')[-1]}"
                    zf.extract(member, './attachments')

    time.sleep(1)
    for file in ForRemove:
        os.remove(file)
    if ForRemove == []:
        break