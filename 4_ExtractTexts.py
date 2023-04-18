import os
import pandas as pd
from src.imgs_extractor import imgs_extractor
from src.pdf_extractor import pdf_extractor
from src.WebCrawling import WebCrawling
import time
import multiprocessing
import re

if not os.path.exists(os.path.join("./attachments", 'texts')):
    os.makedirs(os.path.join("./attachments", 'texts'))



fullList = os.listdir("./attachments")
fullList = [i for i in fullList if os.path.splitext(os.path.basename(i))[0]+'.txt' not in os.listdir("./attachments/texts")]

imgsList = [i for i in fullList if i.split('.')[-1] in ['jpg', 'png', 'gif', 'jpeg', 'bmp']]
pdfList = [i for i in fullList if i.split('.')[-1] == 'pdf']
excelList = [i for i in fullList if i.split('.')[-1] in ['xls', 'xlsx']]

errorList = list(set(fullList)-set(imgsList)-set(pdfList)-set(excelList)-{'texts'})

imgsList = list(map(lambda x : os.path.join("./attachments", x), imgsList))
pdfList = list(map(lambda x : os.path.join("./attachments", x), pdfList))
excelList = list(map(lambda x : os.path.join("./attachments", x), excelList))






if errorList:
    print(*errorList, sep='\n')
    print('이상의 파일은 처리할 수 없습니다.')
    

    
print('='*10)
print('텍스트 추출을 시작합니다.')


if imgsList:
    print('이미지 파일에서 추출 중...', end = ' ')
    imgs_extractor(imgsList)
    print('완료!')

if excelList:
    print('엑셀 파일에서 추출 중...', end = ' ')

    for file in excelList:
        file_basename = os.path.splitext(os.path.basename(file))[0]
        table = pd.read_excel(file).to_csv(sep = ';', header=False, index=False)

        savetxt = os.path.join('./attachments','texts',f"{file_basename}.txt".split('/')[-1])
        f = open(savetxt,'w', encoding='utf-8')
        f.write(table)
        f.close()
    
    print('완료!')

if pdfList:
    print('PDF 파일에서 추출 중...', end = ' ')
    for file in pdfList:
        pdf_extractor(file)
    print('완료!')
    

print('공고문에서 metadata 추출 중...', end = ' ')

with open('./urls.txt') as f:
    urls = [i for i in f.read().split('\n') if i]
    
for url in urls:
    wc = WebCrawling(url)

    wc_dict = {key:val for key,val in wc.__dict__.items() if key not in ['soup', 'url']}

    wc_dict['카테고리'] = wc_dict.pop('category')
    wc_dict['공고 제목'] = wc_dict.pop('title')
    wc_dict['태그'] = wc_dict.pop('tags')

    wc_dict.update(wc.infos())
    wc_txt = '\n'.join(f"{key} : {val}" for key,val in wc_dict.items())

    wc_txt = re.sub('\t+',' ',wc_txt)
    wc_txt = re.sub('\n\(',' (',wc_txt)

    savetxt = os.path.join('./attachments','texts',f"{url[-6:]}_metadata.txt")
    f = open(savetxt,'w', encoding='utf-8')
    f.write(wc_txt)
    f.close()

    
print('완료!')