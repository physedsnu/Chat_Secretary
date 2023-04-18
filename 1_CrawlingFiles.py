from src.WebCrawling import WebCrawling
import re

with open('./urls.txt') as f:
    urls = [i for i in f.read().split('\n') if i]

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

import os
import time
from urllib.request import urlretrieve

cdriver = "./misc/chromedriver.exe"
driver = webdriver.Chrome(cdriver)

for url in urls:

    driver.get(url)
    time.sleep(0.5)

    html = driver.page_source

    soup = BeautifulSoup(html, 'html.parser')

    file_names=[i.getText() for i in soup.find('div', "attached_file_list").find_all('div', "file_name")]
    file_links=[i.get('href') for i in soup.find('div', "attached_file_list").find_all('a', target="_blank")]

    if len(file_names) != len(file_links):
        raise RuntimeError('something wrong')
    else:
        for name, link in zip(file_names, file_links):
            temp1 = [url[-6:], re.findall('FILE_\d*(\d{8})', link)[-1], re.findall('fileSn=\d*(\d{1})', link)[-1]]
            form = name.split('.')[-1]
            temp1 = f"{'_'.join(temp1)}.{form}"
            temp2 = 'https://www.bizinfo.go.kr'+link
            temp1 = os.path.join('./attachments/', temp1)
            print(f'download : {temp1}...', end=' ')
            urlretrieve(temp2, temp1)
            print(': complete')
driver.quit()
