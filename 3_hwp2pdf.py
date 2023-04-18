import os
import win32com.client as win32
import win32gui
import re
import glob
import time
from pyautogui import press, hotkey
import pyautogui






DIR = os.path.join(os.getcwd(), 'attachments')

if [f for f in os.listdir(DIR) if re.match('.*[.]hwp', f, re.I)]:
    
    print("한컴에서 pdf를 저장할 때 사용하는 설정은 사용자가 마지막으로 사용한 인쇄 설정과 동일합니다.")
    print("한페이지씩 컬러로 저장하고 싶을 때에는 실행 전 인쇄 설정을 확인해 주시기 바랍니다.")

    hwp = win32.Dispatch("hwpframe.hwpobject")
    hwp.XHwpWindows.Item(0).Visible = True
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
    hwnd = win32gui.FindWindow(None, '빈 문서 1 - 한글')
    if hwnd == 0:
        print("한글 프로그램이 열리지 않았습니다. 한글 버전을 확인하세요.")
        raise


    else:
        print("오류를 최소화하기 위해 변환되지 않은 경우 최대 2번 반복 시도될 수 있습니다.")
        for checkno in range(2):
            files = [f for f in os.listdir(DIR) if re.match('.*[.]hwp', f, re.I)]
            for i in files:
                hwp.Open(os.path.join(DIR, i), Format="hwp".upper(), arg="versionwarning=False")
                pre, ext = os.path.splitext(i)
                hwp.SaveAs(DIR+"\\"+pre+".pdf", "PDF")
            #win32gui.ShowWindow(hwnd, 5)
            hwp.Quit()
            time.sleep(1)

            for file in glob.glob("./attachments/*.pdf"):
                if os.path.exists(file.replace('.pdf', '.hwp')) or os.path.exists(file.replace('.pdf', '.HWP')):
                    if os.stat(file).st_size < 7000:
                        os.remove(file)
                    else:
                         os.remove(file.replace('.pdf', '.hwp'))
            if checkno < 2:
                hwp = win32.Dispatch("hwpframe.hwpobject")
                hwp.XHwpWindows.Item(0).Visible = True
                hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")






    temp1 = [i.split('.')[0] for i in os.listdir("./attachments") if i.split('.')[1]=='hwp']
    temp2 = [i.split('.')[0] for i in os.listdir("./attachments") if i.split('.')[1]=='pdf']

    leftover = [i+'.hwp' for i in (set(temp1)-set(temp2))]



    hwp = win32.Dispatch("hwpframe.hwpobject")
    hwp.XHwpWindows.Item(0).Visible = True
    hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")





    for i in leftover:
        ActiveDoc0 = win32gui.GetWindowText(win32gui.GetForegroundWindow())
        os.startfile(os.path.join(DIR, i))
        while True:
            ActiveDoc = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            if i in ActiveDoc:
                time.sleep(2)
                hotkey('ctrl', 'a')
                time.sleep(1)
                hotkey('ctrl', 'c')
                break
            pass

        hotkey('alt', 'f4')

        hwnd = win32gui.FindWindow(None, ActiveDoc0)

        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, 9)
        time.sleep(2)
        hotkey('ctrl', 'v')
        time.sleep(2)
        pre, ext = os.path.splitext(i)
        hwp.SaveAs(DIR+"\\"+pre+".hwp", "HWP")
        time.sleep(5)
        hwp.SaveAs(DIR+"\\"+pre+".pdf", "PDF")
        time.sleep(5)
        hotkey('ctrl', 'z')
        time.sleep(2)
    hwp.Quit()


    leftover = [i+'.hwp' for i in (set(temp1)-set(temp2))]


    while leftover:
        print(*leftover, sep='\n')
        print('위 파일은 보안 등의 이유로 열지 못하였습니다.')
        cont = input("새 문서를 열고 전체복사 후 파일을 덮어쓰기로 저장하십시오. .\n\
        다시 확인하려면 아무 텍스트나 입력하십시오.\n\
        Exit 를 입력시 강제 계속되지만 나머지 한글파일은 버려집니다.\n\
        ")
        print('\n'*3)
        if cont in ['Exit', 'exit']:
            break
        elif cont:
            hwp = win32.Dispatch("hwpframe.hwpobject")
            hwp.XHwpWindows.Item(0).Visible = True
            hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
            hwnd = win32gui.FindWindow(None, '빈 문서 1 - 한글')        

            files = [f for f in os.listdir(DIR) if re.match('.*[.]hwp', f, re.I)]
            for i in files:
                hwp.Open(os.path.join(DIR, i), Format="hwp".upper(), arg="")
                pre, ext = os.path.splitext(i)
                hwp.SaveAs(DIR+"\\"+pre+".pdf", "PDF")
            #win32gui.ShowWindow(hwnd, 5)
            hwp.Quit()
            time.sleep(1)


            for file in glob.glob("./attachments/*.pdf"):
                if os.path.exists(file.replace('.pdf', '.hwp')) or os.path.exists(file.replace('.pdf', '.HWP')):
                    if os.stat(file).st_size < 7000:
                        os.remove(file)
                    else:
                         os.remove(file.replace('.pdf', '.hwp'))

            temp1 = [i.split('.')[0] for i in os.listdir("./attachments") if i.split('.')[1]=='hwp']
            temp2 = [i.split('.')[0] for i in os.listdir("./attachments") if i.split('.')[1]=='pdf']
            leftover = [i+'.hwp' for i in (set(temp1)-set(temp2))]
        else:
            cont = input("다시 확인하려면 아무 텍스트나 입력하십시오.\n")
            if os.path.exists(file.replace('.pdf', '.hwp')) or os.path.exists(file.replace('.pdf', '.HWP')):
                if os.stat(file).st_size < 7000:
                    os.remove(file)
                else:
                     os.remove(file.replace('.pdf', '.hwp'))

    temp1 = [i.split('.')[0] for i in os.listdir("./attachments") if i.split('.')[1]=='hwp']

    for f in [os.path.join('./attachments/', i+'.hwp') for i in temp1]:
        os.remove(f)
else:
    pass