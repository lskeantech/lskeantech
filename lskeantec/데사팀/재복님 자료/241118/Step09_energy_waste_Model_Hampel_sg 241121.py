import pandas as pd
import datetime
import plotly.graph_objects as go
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import holidays
import numpy as np
from plotly.subplots import make_subplots
from pathlib import Path

# TODO: 데이터 불러오기(데이터 경로 확인)
# 원본
current_dir = Path().resolve()
f_name = current_dir / 'lskeantec' / '삼성물산 재실 예측' / 'dt 5f 북동측 241011~241120.csv'
df_true_value = pd.read_csv(f_name, index_col=0, header=0, parse_dates=[0])

df_true_value.rename(columns={
    'plug_use': 'plug',
    'co2_15min_avg': 'Co2',
    'light_use': 'light',
    'ventilation_use': 'vent'
}, inplace=True)

df_true_value['heatcool'] = df_true_value['heating_use'] + df_true_value['cooling_use']


ll = ['plug', 'light', 'vent', 'heatcool']
for item in ll:
    df_true_value.loc[df_true_value[item]>4, item] = np.nan 


df_true_value = df_true_value.loc[df_true_value.index[0]:,:]
kr_holidays = holidays.KR(years=[2024])
# TODO: 예측 시작, 끝 날짜 설정
start_date = datetime.datetime(year=2024, month=3, day=1, hour=0, minute=0)
end_date = datetime.datetime(year=2024, month=10, day=30, hour=23, minute=59)


df_true_value.index=pd.to_datetime(df_true_value.index)

def is_working_day_and_time(x):
#     # 주말, 공휴일이거나 근무 시간 외일 때 0을 반환
    if x.weekday() >= 5 or x in kr_holidays:
        return 0
    elif 9 <= x.hour <= 18:  # 근무 시간 9:00~18:00
        return 1
    else:
        return 0

df_true_value['weekday'] = df_true_value.index.to_series().apply(is_working_day_and_time)
df_true_value['weekday'] = df_true_value['weekday'].astype(int)
        

# TODO: 사용 변수 및 파라미터
'''
lookback_days: 이전 며칠을 학습할건지
var_list: 사용 변수 목록
hmm_params: hmm 모델 하이퍼 파라미터
'''
lookback_days = 28
var_list = ['plug']
hmm_params = {'n_components': 2
            , 'covariance_type': 'full'
            , 'n_iter': 100
            # , 'random_state': 5609
            , 'random_state': 42
            # , 'tol': 1e-2
            , 'init_params': 'stmc'
            }
# TODO: 가중치 설정
# df_true_value['hour_neg_cos'] = df_true_value['hour_neg_cos'] * 1e-02
# df['plug'] = np.where(df['office_time'] == 1, df['plug'] * 1.5, df['plug']) # 너무 편향되지 않게..
# df['office_time'] = df['office_time'] * 1e-04
# df_true_value['plug*light'] = df_true_value['plug*light'] * 1e-02

target_data = df_true_value.loc[start_date-datetime.timedelta(days=lookback_days)-datetime.timedelta(days=7): end_date, :] 

date_range = pd.date_range(start=start_date-datetime.timedelta(days=7), end=end_date, freq='1d') # -7은 대기 전력 계산을 위해.
for i, target_date in enumerate(date_range):
  target_start = target_date
  target_end = target_start + datetime.timedelta(hours=23, minutes=59)
  
  # TODO: 모델 생성  
  hmm = GaussianHMM(**hmm_params)

  # 이전 lookback_days 전열 사용량으로 재실 시간 판별
  date_start_2w = target_start - datetime.timedelta(days=lookback_days)
  data_2w = target_data.loc[date_start_2w:target_end]
  
  hmm.fit(data_2w[['plug','weekday']])
  plug_today = target_data.loc[target_start:target_end, var_list]
  mask = hmm.predict(plug_today[var_list])

  # 재실 처리(있으면 true, 없으면 false)
  avg_0 = plug_today.loc[mask==0, 'plug'].mean()
  avg_1 = plug_today.loc[mask==1, 'plug'].mean()
  
  if (avg_0 or avg_1) is np.nan:
    target_data.loc[target_start:target_end, "occupancy"] = 0
  else:
    if avg_0 > avg_1:
      target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 0 else 0 for _m in mask]
    else:
      target_data.loc[target_start:target_end, "occupancy"] = [1 if _m == 1 else 0 for _m in mask]

      # TODO: 실제 데이터 가져오기. (전처리로 인해 기존 데이터의 변형이 있을 수 있기에)
df_true_value['occupancy'] = target_data['occupancy']
df_true_value['is_weekend_or_holiday'] = target_data['is_weekend_or_holiday']
df_true_value['office_time'] = target_data['office_time']
df_true_value['occupancy'] = df_true_value['occupancy'].ffill() 

true_1 = len(df_true_value[(df_true_value.index.hour>=8)&(df_true_value.index.hour<19)&(df_true_value['occupancy']==1)])
true_0 = len(df_true_value[(df_true_value.index.hour<8)|(df_true_value.index.hour>=19)&(df_true_value['occupancy']==0)])
accuracy = round((true_1 + true_0) / len(df_true_value),4)*100
# df_true_value.dropna(inplace=True)

# TODO: 7일간의 대기 전력 계산
df_true_value.loc[df_true_value['occupancy']==0, 'plug_7d_avg'] = df_true_value[df_true_value['occupancy']==0]['plug'].rolling(window='7D').mean()
df_true_value['plug-plug_7d_avg'] = df_true_value['plug'] - df_true_value['plug_7d_avg']
df_true_value = df_true_value.loc[start_date:end_date,:]

# TODO: 피쳐별 낭비량 계산
col_list = [col for col in df_true_value.columns if col in ['heatcool', 'light', 'vent']]
for col in col_list:
    df_true_value[f'{col}_waste'] = df_true_value.apply(
        lambda x: x[col] if x['occupancy'] == 0 else 0, axis=1
    )

# df_true_value['plug_waste'] = df_true_value.apply(
#   lambda x: x['plug-plug_7d_avg'] if (x['occupancy']==0 and x['plug-plug_7d_avg']>0) else 0, axis=1
#   )

# TODO: new 전열 낭비량 (승건님 아이디어)
df_true_value['plug_waste'] = df_true_value.apply(
  lambda x: x['plug-plug_7d_avg'] if (x['occupancy']==0) else 0, axis=1
  )

df_true_value['plug_neg_waste'] = abs(df_true_value[df_true_value['plug_waste']<0]['plug_waste'])
df_true_value['plug_pos_waste'] = df_true_value[df_true_value['plug_waste']>0]['plug_waste']

# TODO: 재실 추정 결과 시각화
fig = make_subplots(
  rows = 5,
  specs = [[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
  shared_xaxes = True,
  vertical_spacing = 0.02
)

# 1. 재실 추정 결과 확인
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.plug, mode='lines', name='전열사용량'), row=1, col=1)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.occupancy, mode='lines', name='재실 여부'), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=df_true_value.index, y=df_true_value.is_weekend_or_holiday, mode='lines', name='휴일 여부'), row=1, col=1, secondary_y=True)

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

# 5. 전처리된 데이터
fig.add_trace(go.Scatter(x=target_data.index, y=target_data.plug, mode='lines', name='전열사용량'), row=5, col=1)
fig.add_trace(go.Scatter(x=target_data.index, y=target_data.occupancy, mode='lines', name='재실 여부'), row=5, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=target_data.index, y=target_data.is_weekend_or_holiday, mode='lines', name='휴일 여부'), row=5, col=1, secondary_y=True)


fig.update_layout(height=700, width=1200, showlegend=True, title=f'{f_name} <br> {lookback_days}일 이전 데이터로 추정, 사용 변수: {var_list}, 오피스 타임 기준 정확도: {accuracy}% <br> hmm_params={hmm_params}')
fig.show()

# TODO: 피쳐별 총 낭비량 시각화
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
    title="에너지 낭비량",
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

# hampel을 적용해서 좋았던 부분 5월 1일, 좋지 않은 부분 9월 13일