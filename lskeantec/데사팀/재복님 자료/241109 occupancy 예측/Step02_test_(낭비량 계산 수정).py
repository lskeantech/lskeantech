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
df_true_value = df.copy()

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

# TODO: 이전 또는 이후 전열 사용량 파생 변수 생성
if 'plug_shift1' in var_list:
  df['plug_shift1'] = df['plug'].shift()  # 15분 이전
if 'plug_shift2' in var_list:
  df['plug_shift2'] = df['plug'].shift(2) # 30분 이전
if 'plug_shift4' in var_list:  
  df['plug_shift4'] = df['plug'].shift(4) # 1시간 이전
if 'plug_shift-1' in var_list:
  df['plug_shift-1'] = df['plug'].shift(-1)  # 15분 이후
if 'plug_shift-2' in var_list:
  df['plug_shift-2'] = df['plug'].shift(-2)  # 15분 이후

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
# 예측하고자하는 날 - 14일() - 7일(대기 전력 계산을 위해)
target_data = df.loc[start_date-datetime.timedelta(days=lookback_days)-datetime.timedelta(days=7): end_date, :] 

date_range = pd.date_range(start=start_date-datetime.timedelta(days=7), end=end_date, freq='1d') # -7은 대기 전력 계산을 위해.
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


# TODO: 실제 데이터 가져오기. (전처리로 인해 기존 데이터의 변형이 있을 수 있기에)
df_true_value['occupancy'] = target_data['occupancy']
df_true_value['occupancy'] = df_true_value['occupancy'].ffill() 
df_true_value.dropna(inplace=True)

# TODO: 7일간의 대기 전력 계산
df_true_value.loc[df_true_value['occupancy']==0, 'plug_7d_avg'] = df_true_value[df_true_value['occupancy']==0]['plug'].rolling(window='7D').mean()
df_true_value['plug-plug_7d_avg'] = df_true_value['plug'] - df_true_value['plug_7d_avg']
df_true_value = df_true_value.loc[start_date:end_date,:]

# TODO: 피쳐별 낭비량 계산
for col in ['heatcool', 'light', 'vent']:
    df_true_value[f'{col}_waste'] = df_true_value.apply(
        lambda x: x[col] if x['occupancy'] == 0 else 0, axis=1
    )
df_true_value['plug_waste'] = df_true_value.apply(
  lambda x: x['plug-plug_7d_avg'] if (x['occupancy']==0 and x['plug-plug_7d_avg']>0) else 0, axis=1
  )

# TODO: 재실 추정 결과 시각화
fig = make_subplots(
  rows = 4,
  specs = [[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
  shared_xaxes = True,
  vertical_spacing = 0.02
)

# 1. 재실 추정 결과 확인
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.plug, mode='lines', name='전열사용량'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.occupancy, mode='lines', name='재실 여부'), row=1, col=1, secondary_y=True)

# 2. 조명 낭비량
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.light, mode='lines', name='조명사용량'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.light_waste, mode='lines', name='조명낭비량'), row=2, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.occupancy, mode='lines', name='재실 여부'), row=2, col=1, secondary_y=True)

# 3. 전열 낭비량
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.plug, mode='lines', name='전열사용량'), row=3, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.plug_waste, mode='lines', name='전열낭비량'), row=3, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.occupancy, mode='lines', name='재실 여부'), row=3, col=1, secondary_y=True)

# 4. 냉난방 낭비량
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.heatcool, mode='lines', name='냉난방사용량'), row=4, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.heatcool_waste, mode='lines', name='냉난방낭비량'), row=4, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.occupancy, mode='lines', name='재실 여부'), row=4, col=1, secondary_y=True)

fig.update_layout(height=700, width=1200, showlegend=True, title=plot_title)
fig.show()


# TODO: 피쳐별 9월 총 낭비량 시각화
waste_columns = [col for col in df_true_value.columns if col.endswith('waste')]
waste_totals = df_true_value[waste_columns].sum()

fig = go.Figure()

for waste_column in waste_columns:
    fig.add_trace(go.Bar(
      x=[waste_column], 
      y=[waste_totals[waste_column]], 
      name=waste_column,
      text=waste_totals[waste_column].round(2),  # 소수점 둘째 자리로 반올림
      texttemplate='%{text}',  # 텍스트 표시 형식
      textposition='outside',  # 텍스트 위치 설정
      textfont=dict(size=15, color="black", family="Arial", weight="bold")  # 글자 크기, 색상, 두께 설정
      ))

fig.update_layout(
    title="9월 에너지 낭비량",
    xaxis_title="Features",
    yaxis_title="Total kWh",
    height=500,
    barmode='group',
    yaxis=dict(range=[0, waste_totals.max() * 1.2])  # y축 범위를 최대값의 120%로 설정 (값 표시 짤리는 거 방지)
    )
fig.show()


# TODO: 요일별 피쳐 낭비량 시각화
df_true_value['weekday'] = df_true_value.index.weekday
waste_columns = [col for col in df_true_value.columns if col.endswith('waste')]
weekday_totals = df_true_value.groupby('weekday')[waste_columns].sum()
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekday_totals.index = weekday_names

for waste_column in waste_columns:
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=weekday_totals.index, 
        y=weekday_totals[waste_column],
        name=waste_column,
        marker=dict(
            color=weekday_totals[waste_column],  # 값에 따라 색상 조절
            colorscale='Reds',  # 색상 스케일 선택
            showscale=True  # 색상 스케일 표시
        ),
        text=weekday_totals[waste_column].round(2),  # 소수점 둘째 자리로 반올림
        texttemplate='%{text}',  # 텍스트 표시 형식
        textposition='outside',  # 텍스트 위치 설정
        textfont=dict(size=15, color="black", family="Arial", weight="bold")  # 글자 크기, 색상, 두께 설정
    ))

    fig.update_layout(
        title=f"요일별 {waste_column}",
        xaxis_title="Day of the Week",
        yaxis_title="Total kWh",
        height=500,
        yaxis=dict(range=[0, weekday_totals[waste_column].max() * 1.2])  # y축 범위를 최대값의 120%로 설정 (값 표시 짤리는 거 방지)
    )
    
    fig.show()