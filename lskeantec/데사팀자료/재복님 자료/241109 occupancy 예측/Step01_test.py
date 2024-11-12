import pandas as pd
import datetime
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import holidays
import numpy as np
from plotly.subplots import make_subplots
from config import start_date, end_date, lookback_days, plug_threshold, hmm_params, eda, plot_title, var_list, plot_title
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.cluster import OPTICS
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from pykalman import KalmanFilter
from tensorflow.keras import layers, models


# TODO: 원본 데이터 불러오기(데이터 경로 확인)
f_name = "ean_energy/merged_7f.csv"
df = pd.read_csv(f_name, index_col=0, header=0)
df.index = pd.to_datetime(df.index)
df = df[(df.index>=start_date-datetime.timedelta(days=lookback_days))&(df.index<=end_date)] # 9월달로 설정
df_true_value = df.loc[start_date:end_date]

# TODO: 중복된 시간 제거
df = df[~df.index.duplicated()]

# TODO: 누락된 시간 데이터 Null로 생성 -> 선형보간
min_time = df.index.min()
max_time = df.index.max()
all_times = pd.date_range(start=min_time, end=max_time, freq='15min') 
df = df.reindex(all_times)
df = df.interpolate()
df.dropna(inplace=True)

# TODO: 9/1~9/30 데이터 추출 후 데이터 plot
# 전열 사용량 데이터 확인 (전처리 전)
if eda == 'on':
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df.index, y=df.plug, mode='lines', name='전열사용량'))
  fig.update_layout(title='전열사용량 (전처리 전)')
  fig.show()

# TODO: 한국 공휴일 컬럼 생성
kr_holidays = holidays.KR(years=[2024])
df['is_weekend_or_holiday'] = df.index.to_series().apply(
    lambda x: 1 if x.weekday() >= 5 or x in kr_holidays else 0
)

# TODO: 주말 및 공휴일이고 전열 threshold 초과일 경우 널값 후 선형보간
df.loc[(df['is_weekend_or_holiday']==1)&(df['plug']>plug_threshold), 'plug'] = np.nan
df['plug'] = df['plug'].interpolate()
df['plug'] = df['plug'].bfill()

# TODO: 수치형 데이터 스케일링
# scaler = MinMaxScaler()
# cols = df.select_dtypes(exclude='object').columns
# df[cols] = scaler.fit_transform(df[cols])

# TODO: 주말 및 공휴일이 아닐 경우 가중치 설정
# df.loc[:,'plug'] = df.apply(lambda x: x.plug*2 if x.is_weekend_or_holiday == 0 else x.plug, axis=1)

# TODO: 15분 이전 전열 사용량 파생 변수 생성
# df['plug_shift1'] = df['plug'].shift()  # 15분 이전
# df['plug_shift2'] = df['plug'].shift(2) # 30분 이전
# df['plug_shift4'] = df['plug'].shift(4) # 1시간 이전
df['plug_shift-1'] = df['plug'].shift(-1)  # 15분 이후
# df['plug_shift-2'] = df['plug'].shift(-2)  # 15분 이후
df.dropna(inplace=True)

# TODO: 전처리 이후 데이터 시각화
# 전열 사용량 데이터 확인
if eda == 'on':
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=df.index, y=df.plug, mode='lines', name='전열사용량'))
  fig.add_trace(go.Scatter(x=df.index, y=df.is_weekend_or_holiday, mode='lines', name='주말 및 공휴일'))
  fig.update_layout(title='전열사용량 (전처리 후)')
  fig.show()


# TODO: 전열(plug) 데이터로 재실 추정
target_data = df.loc[start_date-datetime.timedelta(days=lookback_days): end_date, :] # 예측하고자하는 날 - 14일()

daily_energy_waste = pd.DataFrame()

date_range = pd.date_range(start=start_date, end=end_date, freq='1d')
for i, target_date in enumerate(date_range):
  target_start = target_date
  target_end = target_start + datetime.timedelta(hours=23, minutes=59)
  
  # 이전 lookback_days 전열 사용량으로 재실 시간 분석
  date_start_2w = target_start - datetime.timedelta(days=lookback_days)
  data_2w = target_data.loc[date_start_2w:target_end]

  # TODO: 모델 생성  
  hmm = GaussianHMM(**hmm_params)

  hmm.fit(data_2w[var_list])
  plug_today = target_data.loc[target_start:target_end, var_list]
  mask = hmm.predict(plug_today[var_list])

  # 재실 처리(있으면 true, 없으면 false)
  avg_0 = plug_today.loc[mask==0, 'plug'].mean()
  avg_1 = plug_today.loc[mask==1, 'plug'].mean()

  # 조건 하나 더 추가해야할듯. 전부 1로 예측되는 경우에는 주말이나 공휴일일 경우가 있음. 그 경우에 모두 1로 돼서 재실처리가 됨. 해당 오류 해결해야함.
  # 조건 추가. 둘 중 하나 결측인 경우 0으로 설정
  if (avg_0 or avg_1) is np.nan:
    # if all(value == 0 for value in mask):
    target_data.loc[target_start:target_end, "occupancy"] = 0
  else:
    if avg_0 > avg_1:
      target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 0 else 0 for _m in mask]
    else:
      target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 1 else 0 for _m in mask]

  data_1d = target_data.loc[target_start:target_end, :] 

  # 재실 시작 및 종료 시각 찾기
  if (data_1d.occupancy == 1).any():
    occ_start_time = data_1d.loc[data_1d.occupancy == 1].index[0]
    occ_end_time = data_1d.loc[data_1d.occupancy == 1].index[-1]
  else:
    occ_start_time = None
    occ_end_time = None

  # 기준 부하 (대기 전력량 계산하는 거일거임)
  date_start_7d = target_start - datetime.timedelta(days=7)
  date_end_7d = target_end - datetime.timedelta(days=1)
  past_data = target_data.loc[date_start_7d:date_end_7d]
  plug_base = past_data.loc[past_data.occupancy == 0, 'plug'].mean()

  # 전열 낭비량
  data_1d.loc[:, 'plug_waste'] = data_1d.apply(lambda x: x.plug - plug_base if (x.occupancy == 0 and x.plug > plug_base) else 0, axis=1)
  target_data.loc[target_start:target_end, 'plug_waste'] = data_1d.apply(lambda x: x.plug - plug_base if (x.occupancy == 0 and x.plug > plug_base) else 0, axis=1)

  # 조명 낭비량
  data_1d.loc[:, 'light_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.light, axis=1)
  target_data.loc[target_start:target_end, 'light_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.light, axis=1)

  if occ_start_time != None:
    # 환기 낭비량
    data_1d.loc[:, 'vent_waste'] = data_1d.apply(lambda x: x.vent if x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time else 0, axis=1)
    target_data.loc[target_start:target_end, 'vent_waste'] = data_1d.apply(lambda x: x.vent if x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time else 0, axis=1)

    # 냉난방 낭비량
    data_1d.loc[:, 'heatcool_waste'] = data_1d.apply(lambda x: x.heatcool if x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time else 0, axis=1)
    target_data.loc[target_start:target_end, 'heatcool_waste'] = data_1d.apply(lambda x: x.heatcool if x.name < occ_start_time - datetime.timedelta(hours=1) or x.name > occ_end_time else 0, axis=1)

  if occ_start_time == None:
    # 환기 낭비량
    data_1d.loc[:, 'vent_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.vent, axis=1)
    target_data.loc[target_start:target_end, 'vent_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.vent, axis=1)

    # 냉난방 낭비량
    data_1d.loc[:, 'heatcool_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.heatcool, axis=1)
    target_data.loc[target_start:target_end, 'heatcool_waste'] = data_1d.apply(lambda x: 0 if x.occupancy == 1 else x.heatcool, axis=1)

  daily_energy_waste.loc[target_start, 'occ_start_time'] = occ_start_time
  daily_energy_waste.loc[target_start, 'occ_end_time'] = occ_end_time
  daily_energy_waste.loc[target_start, 'plug_base'] = plug_base
  daily_energy_waste.loc[target_start, 'light'] = data_1d.light.sum()
  daily_energy_waste.loc[target_start, 'light_waste'] = data_1d.light_waste.sum()
  daily_energy_waste.loc[target_start, 'plug'] = data_1d.plug.sum()
  daily_energy_waste.loc[target_start, 'plug_waste'] = data_1d.plug_waste.sum()
  daily_energy_waste.loc[target_start, 'heatcool'] = data_1d.heatcool.sum()
  daily_energy_waste.loc[target_start, 'heatcool_waste'] = data_1d.heatcool_waste.sum()
  # daily_energy_waste.loc[target_start, 'vent'] = data_1d.vent.sum()
  # daily_energy_waste.loc[target_start, 'vent_waste'] = data_1d.vent_waste.sum()

  # 낭비 summary
  # print(">> %d년 %d월 %d일" % (target_start.year, target_start.month, target_start.day))
  # if occ_start_time:
  #   print("1. 재실 시작시각: %dh %dm" % (occ_start_time.hour, occ_start_time.minute))
  # else:
  #   print("1. 재실 시작시각: 없음")
  # print("2. 지난 7일간 기저 전열 사용량: %.2fkWh" % plug_base)
  # print("3. 일간 조명 낭비량: %.2fkWh" % (data_1d.light_waste.sum()))
  # print("4. 일간 전열 낭비량: %.2fkWh" % (data_1d.plug_waste.sum()))
  # print("5. 일간 냉난방 낭비량: %.2fkWh" % (data_1d.heatcool_waste.sum()))
  # print("6. 일간 환기 낭비량: %.2fkWh" % (data_1d.vent_waste.sum()))


target_data = target_data.loc[start_date: end_date, :]

# TODO: 재실 추정 결과 시각화
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

fig.update_layout(height=700, width=1200, showlegend=True, title=plot_title)
fig.show()