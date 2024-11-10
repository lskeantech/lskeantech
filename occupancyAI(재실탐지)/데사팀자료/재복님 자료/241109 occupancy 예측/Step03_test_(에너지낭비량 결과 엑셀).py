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
from dateutil.relativedelta import relativedelta
import os
import warnings
warnings.filterwarnings('ignore')

result_df = pd.DataFrame()

folder_list = ['ean_energy', 'namutech']

for folder in folder_list:
  print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{folder} 시작>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
  csv_list = [item for item in os.listdir(f'{folder}') if item.endswith('.csv')]
  csv_list = sorted(csv_list, key=lambda x: int(x.split('_')[1][:-5]))
  for csv in csv_list:
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>{csv} 시작>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    if folder == 'ean_energy':
      start_date = datetime.datetime(year=2024, month=1, day=1, hour=0, minute=0)
      end_date = datetime.datetime(year=2024, month=2, day=1, hour=0, minute=0) - datetime.timedelta(minutes=1)
    if folder == 'namutech':
      start_date = datetime.datetime(year=2024, month=3, day=1, hour=0, minute=0)
      end_date = datetime.datetime(year=2024, month=4, day=1, hour=0, minute=0) - datetime.timedelta(minutes=1)

    while start_date < pd.to_datetime('2024-11'): # 2024년 1월부터 10월까지 낭비량 진단
      # TODO: 원본 데이터 불러오기(데이터 경로 확인)
      f_name = f"{folder}/{csv}"
      # f_name = f"{folder}/merged_9f.csv"
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

      # TODO: 한국 공휴일 컬럼 생성
      kr_holidays = holidays.KR(years=[2024])
      df['is_weekend_or_holiday'] = df.index.to_series().apply(
          lambda x: 1 if x.weekday() >= 5 or x in kr_holidays else 0
      )

      # TODO: 주말 및 공휴일이고 전열 threshold 초과일 경우 널값 후 선형보간
      df.loc[(df['is_weekend_or_holiday']==1)&(df['plug']>plug_threshold), 'plug'] = np.nan
      df['plug'] = df['plug'].interpolate()
      df['plug'] = df['plug'].bfill()

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

      col_list = [col for col in df_true_value.columns if col in ['heatcool', 'light', 'vent']]
      # TODO: 피쳐별 낭비량 계산
      for col in col_list:
          df_true_value[f'{col}_waste'] = df_true_value.apply(
              lambda x: x[col] if x['occupancy'] == 0 else 0, axis=1
          )
      df_true_value['plug_waste'] = df_true_value.apply(
        lambda x: x['plug-plug_7d_avg'] if (x['occupancy']==0 and x['plug-plug_7d_avg']>0) else 0, axis=1
        )

      # TODO: 데이터 정리 후 결과 프레임에 합치기
      waste_columns = [col for col in df_true_value.columns if col.endswith('waste')]
      waste_totals = df_true_value[waste_columns].sum()

      floor = csv.split('_')[-1].split('.')[0]
      df_year = int(df_true_value.index.year[0])
      df_month = int(df_true_value.index.month[0])
      df_xlsx = pd.DataFrame({
          'building_name': [f'{folder}'],
          'floor': floor,
          'year': df_year,
          'month': df_month,
      })
      df_xlsx = pd.concat([df_xlsx, waste_totals.to_frame().T], axis=1)

      result_df = pd.concat([result_df, df_xlsx])

      start_date += relativedelta(months=1)
      end_date   += relativedelta(months=1)

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
      # fig.show()
      fig.write_image(f'result/visualizations/{folder}_{floor}_{df_year}_{df_month}.png')

result_df.to_csv(f'result/result.csv', index=False, encoding='euc-kr')