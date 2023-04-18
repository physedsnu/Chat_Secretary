import os
import openai
import tiktoken
from python_settings import settings
import pandas as pd
import numpy as np
import json
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

class GptForJson:

    def __init__(self, title, settings):
        embedding_path = os.path.join(settings.CRAWLER.OUTPUT_FOLDER, title, "processed/embeddings.csv")
        self.settings = settings
        openai.api_key = settings.OpenAI.API_KEY
        self.df = pd.read_csv(embedding_path)
        self.df['embeddings'] = self.df['embeddings'].apply(eval).apply(np.array)

    def create_context(self, question, max_len=500, size="ada"):
        """
        질문에 대한 맥락을 생성하기 위해 데이터프레임에서 가장 유사한 맥락을 찾습니다.
        """

        # 질문에 대한 임베딩을 가져옵니다.
        q_embeddings = openai.Embedding.create(input=question, engine=self.settings.OpenAI.EMBEDDING_MODEL)['data'][0]['embedding']
        print("q_embeddings:", q_embeddings)

        # 임베딩에서 거리를 구합니다.
        self.df['distances'] = distances_from_embeddings(q_embeddings, self.df['embeddings'].values, distance_metric='cosine')

        returns = []
        cur_len = 0

        # 거리순으로 정렬하고 맥락의 길이가 너무 길어질 때까지 텍스트를 추가합니다.
        for i, row in self.df.sort_values('distances', ascending=True).iterrows():
            
            # 텍스트의 길이를 현재 길이에 더합니다.
            cur_len += row['n_tokens'] + 4
            
            # 만약 맥락이 너무 길어지면 중지합니다.
            if cur_len > max_len:
                break
            # 그렇지 않으면 반환되는 텍스트에 추가합니다.
            else:
                returns.append(row["text"])

        # 맥락을 반환합니다.
        return "\n\n".join(returns) + "\n\n"

    def answer_question(self,
        # model="text-davinci-003", model="gpt-3.5-turbo", # 모델 선택
        model="gpt-3.5-turbo", # 모델 선택
        json_form = "잠온다...", # json 형식 
        q = 'What is your question?', # 질문 입력 
        max_len=3600, # 생성할 컨텍스트의 최대 길이
        size="ada", # 생성할 컨텍스트의 크기
        debug=False, # 디버그 여부
        max_tokens=500, # 최대 토큰 개수
        stop_sequence=None, # 멈춤 시퀀스
        temperature=0.0 # 모델 출력의 다양성 조절
    ):
        """
        데이터 프레임 텍스트의 가장 유사한 컨텍스트를 기반으로 질문에 답합니다
        """
        context = self.create_context( # 입력된 질문을 기반으로 컨텍스트 생성
            q,
            max_len=max_len,
            size=size,
        ) 

        # 디버깅하는 경우 원시 모델 응답을 출력합니다
        if debug:
            print("Context:\n" + context)
            print("\n\n")

        try:
            # 토큰수 알아내는법
            # import tiktoken
            # text = "토큰수를 알고싶은 문장을 넣으시오"
            # tokenizer = tiktoken.get_encoding("cl100k_base")
            # tokens = tokenizer.encode(text) 
            # token_count = len(tokens) 
            # print(f"토큰수: {token_count}")
            
            # 117토큰
            prompt = f"---\n{context} \n위 내용을 참고하여 {json_form}를 키값으로 하는 json파일을 작성해주세요. 값에대한 내용이 없다면 '없음'이라고 표시하고, 값에대한 내용안에 \" 문자가 있다면 반드시 ' 로 바꿔서 표시해주세요."
            
#             prompt = f"---\nContext: {context} \nCreate a json file with {json_form} as the key value by referring to the context\nBe sure to observe the following precautions\n\
# 1.Make sure to write in json format\n2.Mark 'None' if there is no content about the value\n3. f there is a \" character in the content of the value, it must be replaced with '" 
                        
                            
                        
            # prompt1 = f"Fill in the given json form based on the context below, \n \
            #             If the information does not exist, fill in 'null' \n \
            #             Does not generate new keys that are not shown in the example in json form \n \
            #             Context: {context} \
            #             \n\n---\n\n \
            #             {json_form}"

            print(prompt)
            print(f"max len: {max_len}")
            
            response = openai.ChatCompletion.create(
                model=self.settings.OpenAI.CHAT_MODEL,
                messages=[
                    {"role": "assistant", "content": prompt},
                ],
                temperature=temperature, # 모델 출력의 다양성 조절
            )
            print(response["choices"][0]["message"]["content"])
            return response["choices"][0]["message"]["content"]
            # return response
        except Exception as e:
            print(e)
            return ""
        
    def answer_to_json(self, title, answer):
        
        output_folder_path = os.path.join(settings.CRAWLER.OUTPUT_FOLDER, title, "json")
        
        # JSON 파일 이름 설정 및 경로 생성
        json_file_name = f"{title}.json"
        json_file_path = os.path.join(output_folder_path, json_file_name)
        
        # 출력 폴더가 없으면 생성
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            
        # answer를 파이썬 객체로 변환
        answer = json.loads(answer)
            
        # JSON 형식의 텍스트 파일로 answer 저장
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(answer, json_file, ensure_ascii=False, indent=4)
    
    def merge_answers(self, answers):
        
        merged_dict = {}
        
        # JSON 문자열을 Python 딕셔너리로 변환
        for dic in answers:
            dic = json.loads(dic)

            # 두 딕셔너리를 하나로 합치기
            merged_dict = {**merged_dict, **dic}

        # 합쳐진 딕셔너리를 JSON 문자열로 변환
        merged_json_data = json.dumps(merged_dict, ensure_ascii=False)
        
        return merged_json_data
    
        
    # 기본정보를 json 파일로 저장    
    def text_to_json(self, title, input_folder, output_folder):
        input_folder_path = os.path.join(settings.CRAWLER.OUTPUT_FOLDER, title, input_folder)
        output_folder_path = os.path.join(settings.CRAWLER.OUTPUT_FOLDER, title, output_folder)
        
        # txt 파일 이름 설정 및 경로 생성
        text_file_name = f"{title}.txt"
        text_file_path = os.path.join(input_folder_path, text_file_name)
        
        # JSON 파일 이름 설정 및 경로 생성
        json_file_name = f"{title}.json"
        json_file_path = os.path.join(output_folder_path, json_file_name)
        
        # 텍스트 파일에서 공고명(공고문 이름), 소관부처·지자체, 수행기관, 신청기간, 사업개요 정보 가져오기
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            lines = text_file.readlines()
            
        # 기존 JSON 파일에서 데이터 불러오기
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            existing_data = json.load(json_file)
        
        # 텍스트 파일에서 태그 정보 가져오기
        tags = []
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            is_tag_section = False
            for line in text_file:
                line = line.strip()
                if line == "-기본 태그(태깅) 정보-":
                    is_tag_section = True
                    continue
                
                if is_tag_section and line:
                    tags.append(line)
                elif is_tag_section and not line:
                    break
                
        # 태그 정보를 기존 JSON 데이터에 추가하기
        existing_data["태그(태깅)0"] = tags

        # 기존 정보를 기존 JSON 데이터에 추가하기
        for line in lines:
            if "공고명 :" in line:
                existing_data["공고명"] = line.replace("공고명 :", "").strip()
            elif "소관부처·지자체 :" in line:
                existing_data["소관부처·지자체"] = line.replace("소관부처·지자체 :", "").strip()
            elif "수행기관 :" in line:
                existing_data["수행기관"] = line.replace("수행기관 :", "").strip()
            elif "신청기간 :" in line:
                existing_data["신청기간"] = line.replace("신청기간 :", "").strip()
            elif "사업개요 :" in line:
                existing_data["사업개요"] = line.replace("사업개요 :", "").strip()
        
        # 딕셔너리의 키를 정렬하여 새로운 딕셔너리를 만듦
        sorted_data = {k: existing_data[k] for k in sorted(existing_data)}
        
        # 수정된 JSON 데이터를 파일에 저장하기
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(sorted_data, json_file, ensure_ascii=False, indent=4)


    def generate_keywords(): 
        for i in range(6):
            filename = f"json_form{i}.txt"

            with open(filename, 'r', encoding='utf-8') as f:
                json_form = f.read()

            gptforJson = GptForJson(title, settings)

            try:
                answer = gptforJson.answer_question(json_form)
            except Exception as e:
                print(e)
                answer = gptforJson.answer_question(json_form)
            
            gptforJson.answer_to_json(title, answer)
        
            return 
        
if __name__ == "__main__":
    title = "2023년 글로벌 기업 협업 프로그램 창업기업 모집 공고" # backup 폴더의 특이사항5
    
    # with open('json_form.txt', 'r', encoding='utf-8') as f:
    #     json_form = f.read()
    
    # gptforJson = GptForJson(title, settings)
    # try:
    #     answer = gptforJson.answer_question(json_form)
    # except Exception as e:
    #     print(e)
    #     answer = gptforJson.answer_question(json_form)
    # gptforJson.answer_to_json(title, answer)

    for i in range(3):
        jsonname = f"json_forms/json_form{i}.txt" 

        with open(jsonname, 'r', encoding='utf-8') as f:
            json_form = f.read()

        gptforJson = GptForJson(title, settings)

        try:
            for i in range(3):
                q_name = f"q_files/q{i}.txt"
                answer = gptforJson.answer_question(json_form=jsonname, q=q_name)

        except Exception as e:
            print(e)
            answer = gptforJson.answer_question(json_form=jsonname, q=q_name)
            
        gptforJson.answer_to_json(title, answer)
        

    
    # ret = helper.answer_question(question="[인천] 2023년 소상공인시장진흥자금 융자 계획 공고에 대해서 알려주세요", debug=True)
    # print(ret)
