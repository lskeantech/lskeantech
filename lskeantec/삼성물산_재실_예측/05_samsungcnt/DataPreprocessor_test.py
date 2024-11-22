import pandas as pd
import numpy as np
import holidays
from scipy.stats import median_abs_deviation


def set_timestamp_index(df, index, start_date=None):
    if not pd.api.types.is_datetime64_any_dtype(df[index]):
        df[index] = pd.to_datetime(df[index])  # datetime 형식으로 변환 
    df.sort_values(index, inplace=True)
    df.set_index(index, inplace=True)
    if start_date:
        df = df.loc[start_date:]
    return df


def remove_duplicates_index(df):
    """
    데이터프레임에서 중복된 인덱스를 제거.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.

    Returns:
        pd.DataFrame: 중복 인덱스가 제거된 데이터프레임.
    """
    return df[~df.index.duplicated()]


def add_time_columns(df):
    """
    데이터프레임 인덱스에서 시간을 추출하여 'hours'와 'minutes' 컬럼을 추가.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.

    Returns:
        pd.DataFrame: 컬럼이 추가된 데이터프레임.
    """
    df['hours'] = df.index.hour
    df['minutes'] = df.index.minute
    return df


def add_holiday_weekend_flag(df):
    """
    주말 및 공휴일 여부를 나타내는 'is_weekend_or_holiday' 컬럼 추가.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.

    Returns:
        pd.DataFrame: 주말 및 공휴일 플래그가 추가된 데이터프레임.
    """
    kr_holidays = holidays.KR(years=[df.index.min().year, df.index.max().year])
    df['is_weekend_or_holiday'] = df.index.to_series().apply(
        lambda x: 1 if x.weekday() >= 5 or x in kr_holidays else 0
    )
    return df


def add_office_hours(df, start_hour=9, end_hour=18):
    """
    'office_time'과 'core_time' 컬럼 추가. 업무 시간과 핵심 시간 여부를 나타냄.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.
        start_hour (int, optional): 업무 시간 시작 시각. 기본값은 9.
        end_hour (int, optional): 업무 시간 종료 시각. 기본값은 18.

    Returns:
        pd.DataFrame: 컬럼이 추가된 데이터프레임.
    """
    df['office_time'] = df.apply(
        lambda x: 1 if (x['is_weekend_or_holiday'] == 0 and start_hour <= x['hours'] <= end_hour) else 0, axis=1
    )
    df['core_time'] = df.apply(
        lambda x: 1 if (x['is_weekend_or_holiday'] == 0 and 8 <= x['hours'] <= 17) else 0, axis=1
    )
    return df


def iqr_outlier_to_nan(series):
    """
    IQR(Interquartile Range)을 사용하여 이상치를 NaN으로 변환.

    Args:
        series (pd.Series): IQR 기법을 적용할 데이터 시리즈.

    Returns:
        pd.Series: 이상치가 NaN으로 대체된 데이터 시리즈.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.mask((series < lower_bound) | (series > upper_bound), np.nan)


def hampel_filter(series, window_size, n):
    """
    Hampel 필터를 사용하여 이상치를 중앙값으로 대체.

    Args:
        series (pd.Series): 입력 시리즈.
        window_size (int): 롤링 윈도우 크기.
        n (float): 허용하는 MAD의 배수.

    Returns:
        pd.Series: Hampel 필터가 적용된 시리즈.
    """
    rolling_median = series.rolling(window=window_size, center=False).median()
    mad = series.rolling(window=window_size, center=False).apply(lambda x: median_abs_deviation(x, scale='normal'), raw=True)
    
    threshold = n * mad
    difference = np.abs(series - rolling_median)
    outlier_idx = difference > threshold

    series[outlier_idx] = rolling_median[outlier_idx]
    return series


def apply_hampel_to_column(df, column_name, window_size, n, cond=None):
    """
    특정 컬럼에 Hampel 필터를 적용.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.
        column_name (str): Hampel 필터를 적용할 컬럼 이름.
        window_size (int): Hampel 필터의 윈도우 크기.
        n (float): 허용 가능한 MAD의 배수.
        cond (pd.Series, optional): 필터링 조건.

    Returns:
        pd.DataFrame: Hampel 필터가 적용된 데이터프레임.
    """
    if cond is None:
        df[column_name] = hampel_filter(df[column_name], window_size, n)
    else:
        df.loc[cond, column_name] = hampel_filter(df.loc[cond, column_name], window_size, n)
    return df


def add_time_cycle_features(df):
    """
    시간을 주기 함수(sinusoidal)로 변환하여 컬럼 추가.

    Args:
        df (pd.DataFrame): 입력 데이터프레임.

    Returns:
        pd.DataFrame: 주기 함수 컬럼이 추가된 데이터프레임.
    """
    df['hour_sin'] = np.sin(2 * np.pi * df['hours'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hours'] / 24)
    df['hour_neg_sin'] = -np.sin(2 * np.pi * df['hours'] / 24)
    df['hour_neg_cos'] = -np.cos(2 * np.pi * df['hours'] / 24)
    return df
