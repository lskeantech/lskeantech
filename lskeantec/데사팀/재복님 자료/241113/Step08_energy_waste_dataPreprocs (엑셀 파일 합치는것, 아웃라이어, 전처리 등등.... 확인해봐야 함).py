import pandas as pd
import plotly.graph_objects as go
import holidays
import numpy as np
from plotly.subplots import make_subplots
import os

# building_nm = 'ean_energy'
# num_list = [4, 5, 6, 7, 8, 9, 10] # ean

building_nm = 'namutech'
num_list = [1, 2, 3, 4, 5, 6, 7] # namu

for num in num_list:
    # TODO: 원본 데이터 불러오기(데이터 경로 확인)
    f_name = f"/Users/ean/PycharmProjects/재실 탐지/data/{building_nm}/merged_{num}f.csv"
    df = pd.read_csv(f_name, index_col=0, header=0)
    df.index = pd.to_datetime(df.index)
    df = df.loc['2024-01-01':,:]

    # TODO: 중복된 시간 제거
    df = df[~df.index.duplicated()]

    # TODO: 누락된 시간 데이터 Null로 생성
    min_time = df.index.min()
    max_time = df.index.max()
    all_times = pd.date_range(start=min_time, end=max_time, freq='15min') 
    df = df.reindex(all_times)

    # TODO: 한국 공휴일 컬럼 생성
    kr_holidays = holidays.KR(years=[2024])
    df['is_weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: 1 if x.weekday() >= 5 or x in kr_holidays else 0
    )

    def iqr_outlier_to_nan(series):
        # 1사분위수(Q1)와 3사분위수(Q3) 계산
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # IQR을 기준으로 이상치 탐지 마스크 생성
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (series < lower_bound) | (series > upper_bound)

        # 이상치를 NaN으로 대체
        series = series.mask(outlier_mask, np.nan)
        return series

    # 공휴일 또는 주말일 때와 아닐 때 각각 IQR 기법 적용
    for column in df.select_dtypes(include='float').columns: 
        # 공휴일 또는 주말인 경우
        df.loc[df['is_weekend_or_holiday'] == 1, column] = iqr_outlier_to_nan(df.loc[df['is_weekend_or_holiday'] == 1, column])
        # 평일인 경우
        df.loc[df['is_weekend_or_holiday'] == 0, column] = iqr_outlier_to_nan(df.loc[df['is_weekend_or_holiday'] == 0, column])

    # TODO: 결측치 선형 보간
    df = df.interpolate()

    # TODO: 파생 변수 생성
    df['plug*light'] = df['plug'] * df['light']

    # TODO: 시간 컬럼 생성
    df['hours'] = df.index.hour
    df['minutes'] = df.index.minute 

    total_minutes = df['hours'] * 60 + df['minutes']
    df['hour_sin'] = np.sin(2 * np.pi * df['hours'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hours'] / 24)    
    df['hour_neg_sin'] = -np.sin(2 * np.pi * df['hours'] / 24) 
    df['hour_neg_cos'] = -np.cos(2 * np.pi * df['hours'] / 24) 

    df['minute_cos'] = np.cos(2 * np.pi * total_minutes / 1440)
    df['minute_sin'] = np.sin(2 * np.pi * total_minutes / 1440)    
    df['minute_neg_cos'] = -np.cos(2 * np.pi * total_minutes / 1440) 
    df['minute_neg_sin'] = -np.sin(2 * np.pi * total_minutes / 1440) 

    # TODO: 이전 시간대 컬럼 생성
    df['plug_shift1'] = df['plug'].shift()  # 15분 이전
    df['plug_shift2'] = df['plug'].shift(2) # 30분 이전
    df['plug_shift4'] = df['plug'].shift(4) # 1시간 이전
    df['plug_shift8'] = df['plug'].shift(8) # 2시간 이전
    df['plug_shift12'] = df['plug'].shift(12) # 3시간 이전
    df['plug_shift-1'] = df['plug'].shift(-1)  # 15분 이후
    df['plug_shift-2'] = df['plug'].shift(-2)  # 30분 이후

    # TODO: 8 ~ 19시 근무 컬럼 생성
    df['office_time'] = df.apply(
        lambda x: 1 if (x['is_weekend_or_holiday'] == 0 and 8 <= x['hours'] <= 19) else 0,
        axis=1
        )
    
    # TODO: 전열 사용량 변화량 생성
    df['plug_diff'] = df['plug'].diff()
    
    file_path = f'/Users/ean/PycharmProjects/재실 탐지/data/{building_nm}/추가 데이터/{num}층 실내환경.csv'
    if os.path.exists(file_path):
        df_ = pd.read_csv(file_path)
        df_['DATETIME'] = pd.to_datetime(df_['DATETIME'])
        df_ = df_.set_index('DATETIME')
        df = pd.merge(df, df_, left_index=True, right_index=True, how='inner')
        df = df.interpolate()
        df = df.ffill()
        df = df.bfill()
        df.drop(columns=['Radon', 'CH2O'], inplace=True)
    
    df.dropna(inplace=True)

    df.to_csv(f'/Users/ean/PycharmProjects/재실 탐지/data/{building_nm}/preprocs/merged_{num}f_preprocs.csv', encoding='euc-kr')