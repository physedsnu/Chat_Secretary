import pandas as pd

df = pd.read_csv("./merged_data.csv")

for ii in range(len(df)):
    row = df.iloc[ii]
    row_text = eval(row.text)
    print(row_text)

full_df = pd.DataFrame()

for ii in range(len(df)):
    row = df.iloc[ii]
    row_text = eval(row.text)
    
    small_df = pd.DataFrame()
    for key, value in row_text.items():
        small_df = pd.concat([small_df, pd.DataFrame({'key':key, 'value':str(value)}, index=[0])])

    small_df = small_df.transpose()
    small_df.columns = small_df.iloc[0]
    small_df = small_df[1:]
    small_df.index=[row.title]
    small_df.columns.name = None
    full_df = pd.concat([full_df, small_df])
    
    
full_df.to_csv('./data_final.csv', encoding='utf-8-sig')
    
    









