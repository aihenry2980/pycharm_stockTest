import pandas as pd
import numpy as np


# display(predict(org_3))

# 해당 번호의 기업 주가 csv파일을 DataFrame 형태로 불러오는 함수입니다.
def load_file(org_idx):
    # return pd.read_csv(f"/mnt/elice/dataset/contest_data/org_{org_idx}.csv")
    return pd.read_csv(f"data/org_{org_idx}.csv")

# 15개의 기업 주가 파일을 DataFrame 형태로 하나씩 불러온 후 리스트에 저장하는 함수입니다.
def load_files():
    df_list = []
    for i in range(15):
        df_list.append(load_file(i+1))
    return df_list

org_3 = load_file(3)
print('3번 기업의 주가 데이터')
# display(org_3.head())
print(org_3)

# 한 기업의 주가 예측 결과 10일치를 리스트 형태로 반환하는 함수
def predict(df):
    # 간단한 예시를 위해 마지막 10일간의 주가를 거꾸로 뒤집은 결과를 얻는 함수를 만들었습니다.(주가는 생각했던 것과 거꾸로 가니까요...ㅠㅠ)
    # 실제로는 머신러닝 및 딥러닝 알고리즘을 사용하여 값을 예측해야 합니다.
    predicted = df['Close'][-1:-11:-1]
    return list(predicted)

print('3번 기업의 주가 예측 결과')
print(predict(org_3))



# result.csv로 저장할 결과를 result_df라는 dataframe으로 만들어서 저장하는 예시
result_df = pd.DataFrame(
    columns=[f"org_{org_idx}" for org_idx in range(1, 16)], # org는 총 15개
    index=[f"day{day_idx}" for day_idx in range(1, 11)] # 예측 날짜는 총 10개
)

# 주가 csv 파일을 불러옵니다.
df_list = load_files()

# 각 기업별로 predict 함수를 적용하여 10일치의 예측 결과를 얻고, 이를 result_df에 저장
for i in range(15):
    result_df['org_{}'.format(i+1)] = predict(df_list[i])

# result_df를 results_csv로 저장
result_df.to_csv("result.csv")

# display(result_df)