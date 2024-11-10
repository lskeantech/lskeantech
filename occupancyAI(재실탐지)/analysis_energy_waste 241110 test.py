
#24.11.10. 

import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM
import holidays
import warnings
warnings.filterwarnings('ignore')

file_list = [
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_4f_v0.2.csv",
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_5f_v0.2.csv",
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_6f_v0.2.csv",
    r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_7f_v0.2.csv",
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_8f_v0.2.csv",
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_9f_v0.2.csv",
    # r"C:\Users\come1\OneDrive\바탕 화면\승건\승건\EAN\데사팀\lskeantech\occupancyAI(재실탐지)\ean_energy\csv\merged_10f_v0.2.csv"
    # # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_1f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_2f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_3f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_4f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_5f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_6f.csv",
    # r"C:\Users\User\Desktop\승건\데사팀\occupancyAI(재실탐지)\namutech\csv\merged_7f.csv"
    ]

# 한국 공휴일 리스트 (2024년 예시)


start_date = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
end_date = datetime.datetime(year=2024, month=10, day=30, hour=23, minute=59)


daily_energy_waste = pd.DataFrame()


for f_name in file_list:
#  print(f'{f_name}')

 df = pd.read_csv(f_name, index_col=0, header=0)
##header: 이 앞에는 아무 행도 오지못하게 하겠다 라는 뜻

 df.dropna(inplace=True)
#  df.head()  
## 상위 5개 데이터만 보여주기
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index, y=df.plug, mode='lines', name='전열사용량'))
# fig.show()  #결과 보이고싶을떈 다시 보이게 하자!


 df.index=pd.to_datetime(df.index)
#  df['timestamp'] = df.index.map(datetime.datetime.timestamp)  # 타임스탬프 열 추가

# 한국 공휴일 설정
 kr_holidays = holidays.KR(years=[2024])

 # 주말 및 공휴일 구분 열 추가
 df['weekday'] = df.index.to_series().apply(lambda x: 0 if x.weekday() >= 5 or x in kr_holidays else 1)


#  target_data = df.loc[start_date: end_date, :]
 target_data = df.loc[start_date-datetime.timedelta(days=20): end_date, :]

 date_range = pd.date_range(start=start_date, end=end_date, freq='1d')



 for target_date in date_range:
     target_start = target_date
     target_end = target_start + datetime.timedelta(hours=23, minutes=59)
    
    # 이전 20일 전열 사용량으로 학습 데이터 설정
     date_start_20 = target_start - datetime.timedelta(days=20)
     data_20 = target_data.loc[date_start_20:target_end]
    
    # 결측치 제거
     data_20 = data_20.dropna(subset=['plug'])
     if data_20.empty:
        print(f"{target_start} - 학습 데이터가 부족하여 예측을 건너뜁니다.")
        continue

     hmm = GaussianHMM(n_components=2, covariance_type='full', n_iter=100, random_state=42) 
  ## n_components는 HMM의 은닉 상태(hidden states)의 수 => 이 뜻은 결과를 2개의 변수로 예측하겠다 이다. 그것이 켜진건지 안켜진건지는 결정하지 않은 상태이다. 단순히, 두가지 클러스터링으로 분류한 것이다.
  ## covariance_type은 모델의 가우시안 분포의 공분산 형태
  ## n_iter는 모델 학습을 위한 최대 반복 횟수

     hmm.fit(data_20[['plug']].values)
     plug_today = target_data.loc[target_start:target_end, ['plug']].dropna()
     if plug_today.empty:
        print(f"{target_start} - 예측 대상 데이터가 없어 건너뜁니다.")
        continue
    # 예측 수행
     mask = hmm.predict(plug_today[['plug']].values)
#   ].dropna()를 통해 결측치 제거
#   
  
# hmm부터는 머신러닝 코딩 부분임
# 2주동안 학습한 데이터들로, plug부분만의 데이터로 0인지 1인지 클러스팅을 진행


  
    # mask와 plug_today 길이 확인 후 평균 계산
     if len(mask) == len(plug_today):
        avg_0 = plug_today.loc[mask == 0, 'plug'].mean() if (mask == 0).any() else 0
        avg_1 = plug_today.loc[mask == 1, 'plug'].mean() if (mask == 1).any() else 0
     else:
        print(f"{target_start} - mask와 plug_today의 길이가 일치하지 않아 건너뜁니다.")
        continue

    # 재실 여부 설정
     if avg_0 > avg_1:
        target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 0 else 0 for _m in mask]
     else:
        target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 1 else 0 for _m in mask]

     data_1d = target_data.loc[target_start:target_end, :]

    # 1>0 이므로, occupancy1인경우, 재실로 나타나도록 값 설정

# 재실 시작 및 종료 시각 찾기
     if (data_1d.occupancy == 1).any():
        occ_start_time = data_1d.loc[data_1d.occupancy == 1].index[0]
        occ_end_time = data_1d.loc[data_1d.occupancy == 1].index[-1]
     else:
        occ_start_time = None
        occ_end_time = None



     # 기준 부하
     date_start_7d = target_start - datetime.timedelta(days=7)
     date_end_7d = target_end - datetime.timedelta(days=1)
     past_data = target_data.loc[date_start_7d:date_end_7d]

  # data_1d 부분에 원래 target_data있었음. occupancy가 nan으로 잡혀서 수정함
     plug_base = past_data.loc[past_data.occupancy == 0, 'plug'].mean()


  # 조명 낭비량
     data_1d.loc[:, 'light_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.light, axis=1)

  # 사람이 있을땐 낭비가 아니고, 사람이 없을땐 light값이 그대로 낭비다!

     target_data.loc[target_start:target_end, 'light_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.light, axis=1)



  # 환기 낭비량
  # data_1d.loc[:, 'vent_waste'] = data_1d.apply(lambda x: x.vent if not occ_start_time or (x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time) else 0, axis=1)

  # 냉난방 낭비량
     data_1d.loc[:, 'heatcool_waste'] = data_1d.apply(lambda x: x.heatcool if not occ_start_time or (x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time) else 0, axis=1)
     target_data.loc[target_start:target_end, 'heatcool_waste'] = data_1d.apply(lambda x: x.heatcool if not occ_start_time or (x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time) else 0, axis=1)
# occupancy 1시간 전~끝날때 까지의 데이터까지는 낭비가 아니다! 그외는 모두 낭비다



  # 전열 낭비량
     data_1d.loc[:, 'plug_waste'] = data_1d.apply(lambda x: x.plug - plug_base if (x.occupancy == 0 and x.plug > plug_base) else 0, axis=1)
     target_data.loc[target_start:target_end, 'plug_waste'] = data_1d.apply(lambda x: x.plug - plug_base if (x.occupancy == 0 and x.plug > plug_base) else 0, axis=1)
  

     daily_energy_waste.loc[target_start, 'occ_start_time'] = occ_start_time
     daily_energy_waste.loc[target_start, 'occ_end_time'] = occ_end_time
     daily_energy_waste.loc[target_start, 'plug_base'] = plug_base
     daily_energy_waste.loc[target_start, 'light'] = data_1d.light.sum()
     daily_energy_waste.loc[target_start, 'light_waste'] = data_1d.light_waste.sum()
     daily_energy_waste.loc[target_start, 'plug'] = data_1d.plug.sum()
     daily_energy_waste.loc[target_start, 'plug_waste'] = data_1d.plug_waste.sum()
     daily_energy_waste.loc[target_start, 'heatcool'] = data_1d.heatcool.sum()
     daily_energy_waste.loc[target_start, 'heatcool_waste'] = data_1d.heatcool_waste.sum()

    # 재실 추정 결과
 from plotly.subplots import make_subplots
 fig = make_subplots(
 rows = 4,
 specs = [[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
 shared_xaxes = True,
 vertical_spacing = 0.02
 )

# 1. 재실 추정 결과 확인
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.plug, mode='lines', name='전열사용량'), row=1, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.occupancy, mode='lines', name='재실 여부'), row=1, col=1, secondary_y=True)

# 2. 조명 낭비량
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.light, mode='lines', name='조명사용량'), row=2, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.light_waste, mode='lines', name='조명낭비량'), row=2, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.occupancy, mode='lines', name='재실 여부'), row=2, col=1, secondary_y=True)

# 3. 전열 낭비량
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.plug, mode='lines', name='전열사용량'), row=3, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.plug_waste, mode='lines', name='전열낭비량'), row=3, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.occupancy, mode='lines', name='재실 여부'), row=3, col=1, secondary_y=True)

# 4. 냉난방 낭비량
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.heatcool, mode='lines', name='냉난방사용량'), row=4, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.heatcool_waste, mode='lines', name='냉난방낭비량'), row=4, col=1)
 fig.add_trace(go.Scatter(x=target_data.index, y=target_data.occupancy, mode='lines', name='재실 여부'), row=4, col=1, secondary_y=True)

 fig.update_layout(height=700, width=1200, showlegend=True)
 fig.show()

 target_data.to_csv('target_data_result.csv', encoding='euc-kr')