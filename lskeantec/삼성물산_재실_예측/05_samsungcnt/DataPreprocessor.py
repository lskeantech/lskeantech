import pandas as pd
import numpy as np
import holidays
from scipy.stats import median_abs_deviation
import os
from sqlalchemy.sql import text
from DBConnect import DBConnector



class DataPreprocessor:
    def __init__(self, floor=None, start_date=None, end_date=None, column_name=None):
        """
        데이터 전처리를 위한 클래스 초기화.

        Args:
            floor (int, optional): 데이터베이스에서 조회할 층 정보.
            start_date (str, optional): 데이터 필터링 시작 날짜.
            end_date (str, optional): 데이터 필터링 종료 날짜.
        """
        self.floor = floor
        self.start_date = start_date
        self.end_date = end_date
        self.column_name = column_name
        self.df = None
        self.db_connector = DBConnector()  # DBConnector 인스턴스 초기화

    # def load_data(self, query_template):
    #     """
    #     데이터베이스에서 데이터를 로드.

    #     Args:
    #         query_template (str): SQL 쿼리 템플릿. {floor}, {start_date}, {end_date}를 변수로 포함해야 함.
    #     """
    #     # SQL 쿼리 생성
    #     query = query_template.format(
    #         floor=self.floor,
    #         start_date=self.start_date,
    #         end_date=self.end_date
    #     )

    #     # SSH 터널 시작
    #     self.db_connector.start_ssh_tunnel()

    #     try:
    #         # 데이터베이스에서 데이터 가져오기
    #         engine = self.db_connector.get_engine()
    #         with engine.connect() as connection:
    #             self.df = pd.read_sql(query, connection)

    #         # 정렬 및 인덱스 설정
    #         if 'timerange' in self.df.columns:  # 인덱스로 사용할 열 이름 확인
    #             self.df['timerange'] = pd.to_datetime(self.df['timerange'])  # datetime 형식으로 변환
    #             self.df.sort_values('timerange', inplace=True)
    #             self.df.set_index('timerange', inplace=True)

    #         # 타임존 변환 및 로컬라이즈 제거
    #         self.df.index = self.df.index.tz_localize("UTC").tz_convert("Asia/Seoul").tz_localize(None)

    #         print("데이터 로드 및 정렬 성공!")
    #     except Exception as e:
    #         print(f"데이터 로드 실패: {e}")
    #     finally:
    #         # SSH 터널 종료
    #         self.db_connector.stop_ssh_tunnel()

    # def load_data(self, query_template):
    #     """
    #     데이터베이스에서 데이터를 로드.

    #     Args:
    #         query_template (str): SQL 쿼리 템플릿. {floor}, {start_date}, {end_date}를 변수로 포함해야 함.
    #     """
    #     # SQL 쿼리 생성
    #     query = query_template.format(
    #         floor=self.floor,
    #         start_date=self.start_date,
    #         end_date=self.end_date
    #     )

    #     # SSH 터널 시작
    #     self.db_connector.start_ssh_tunnel()

    #     try:
    #         # SQLAlchemy 엔진을 가져오기
    #         engine = self.db_connector.get_engine()

    #         # 데이터 로드
    #         with engine.connect() as connection:
    #             result = connection.execute(query)  # SQLAlchemy를 통한 쿼리 실행
    #             self.df = pd.DataFrame(result.fetchall(), columns=result.keys())  # 결과를 DataFrame으로 변환

    #         # 정렬 및 인덱스 설정
    #         if 'timerange' in self.df.columns:  # 인덱스로 사용할 열 이름 확인
    #             self.df['timerange'] = pd.to_datetime(self.df['timerange'])  # datetime 형식으로 변환
    #             self.df.sort_values('timerange', inplace=True)
    #             self.df.set_index('timerange', inplace=True)

    #         # 타임존 변환 및 로컬라이즈 제거
    #         self.df.index = self.df.index.tz_localize("UTC").tz_convert("Asia/Seoul").tz_localize(None)

    #         print("데이터 로드 및 정렬 성공!")
    #     except Exception as e:
    #         print(f"데이터 로드 실패: {e}")
    #     finally:
    #         # SSH 터널 종료
    #         self.db_connector.stop_ssh_tunnel()

    def load_data(self, query_template):
        """
        데이터베이스에서 데이터를 로드.

        Args:
            query_template (str): SQL 쿼리 템플릿. {floor}, {start_date}, {end_date}를 변수로 포함해야 함.
        """
        # SQL 쿼리 생성
        query = query_template.format(
            floor=self.floor,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # SSH 터널 시작
        self.db_connector.start_ssh_tunnel()

        try:
            # SQLAlchemy 엔진을 가져오기
            engine = self.db_connector.get_engine()

            # 데이터 로드
            with engine.connect() as connection:
                result = connection.execute(text(query))  # text()를 사용하여 쿼리를 실행 가능 객체로 변환
                self.df = pd.DataFrame(result.fetchall(), columns=result.keys())  # 결과를 DataFrame으로 변환

            # 정렬 및 인덱스 설정
            if 'timerange' in self.df.columns:  # 인덱스로 사용할 열 이름 확인
                self.df['timerange'] = pd.to_datetime(self.df['timerange'])  # datetime 형식으로 변환
                self.df.sort_values('timerange', inplace=True)
                self.df.set_index('timerange', inplace=True)

            # 타임존 변환 및 로컬라이즈 제거
            self.df.index = self.df.index.tz_localize("UTC").tz_convert("Asia/Seoul").tz_localize(None)

            print("데이터 로드 및 정렬 성공!")
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
        finally:
            # SSH 터널 종료
            self.db_connector.stop_ssh_tunnel()

    def remove_duplicates_index(self):
        """
        데이터프레임에서 중복된 인덱스를 제거.
        """
        self.df = self.df[~self.df.index.duplicated()]


    def add_time_columns(self):
        """
        데이터프레임 인덱스에서 시간을 추출하여 'hours'와 'minutes' 컬럼을 추가.
        """
        self.df['hours'] = self.df.index.hour
        self.df['minutes'] = self.df.index.minute


    def add_holiday_weekend_flag(self):
        """
        주말 및 공휴일 여부를 나타내는 'is_weekend_or_holiday' 컬럼 추가.
        """
        kr_holidays = holidays.KR(years=[self.df.index.min().year, self.df.index.max().year])
        self.df['is_weekend_or_holiday'] = self.df.index.to_series().apply(
            lambda x: 1 if x.weekday() >= 5 or x in kr_holidays else 0
        )


    def add_office_hours(self, start_hour=9, end_hour=18):
        """
        'office_time'과 'core_time' 컬럼 추가. 업무 시간과 핵심 시간 여부를 나타냄.

        Args:
            start_hour (int, optional): 업무 시간 시작 시각. 기본값은 9.
            end_hour (int, optional): 업무 시간 종료 시각. 기본값은 18.
        """
        self.df['office_time'] = self.df.apply(
            lambda x: 1 if (x['is_weekend_or_holiday'] == 0 and start_hour <= x['hours'] <= end_hour) else 0, axis=1
        )
        self.df['core_time'] = self.df.apply(
            lambda x: 1 if (x['is_weekend_or_holiday'] == 0 and 8 <= x['hours'] <= 17) else 0, axis=1
        )


    @staticmethod
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


    @staticmethod
    def hampel_filter(series, window_size, n):
        rolling_median =series.rolling(window=window_size, center=False).median()
        mad = series.rolling(window=window_size, center=False).apply(lambda x: median_abs_deviation(x, scale='normal'), raw=True)
        
        threshold = n * mad
        difference = np.abs(series - rolling_median)
        outlier_idx = difference > threshold

        series[outlier_idx] = rolling_median[outlier_idx]
        # series[outlier_idx] = np.nan
        return series


    def apply_iqr_to_column(self, column_name, cond=None):
        """
        특정 컬럼에 IQR 기법을 적용하여 이상치를 NaN으로 변환.

        Args:
            column_name (str): IQR 기법을 적용할 컬럼 이름.
            cond (pd.Series, optional): 조건부 필터링을 위한 불리언 시리즈. 기본값은 None.
        """
        if cond is None:
            self.df[column_name] = self.iqr_outlier_to_nan(self.df[column_name])
        else:
            self.df.loc[cond, column_name] = self.iqr_outlier_to_nan(self.df.loc[cond, column_name])


    def apply_hampel_to_column(self, window_size, n, column_name=None, cond=None): #NOTE:아름 수정
        """
        특정 컬럼에 hample filter를 적용하여 이상치를 중앙값으로 변환. (None으로 변환해도 좋을 거 같음)

        Args:
            column_name (str): IQR 기법을 적용할 컬럼 이름.
            cond (pd.Series, optional): 조건부 필터링을 위한 불리언 시리즈. 기본값은 None.
        """
        
        column_name = column_name or self.column_name  # column_name이 없으면 self.column_name 사용 #NOTE:아름 추가


        if cond is None:
            self.df[column_name] = self.hampel_filter(series=self.df[column_name], window_size=window_size, n=n)
        else:
            self.df.loc[cond, column_name] = self.hampel_filter(series=self.df.loc[cond, column_name], window_size=window_size, n=n)


    def add_time_cycle_features(self):
        """
        시간을 주기 함수(sinusoidal)로 변환하여 'hour_sin', 'hour_cos' 등의 컬럼 추가.
        """
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hours'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hours'] / 24)
        self.df['hour_neg_sin'] = -np.sin(2 * np.pi * self.df['hours'] / 24)
        self.df['hour_neg_cos'] = -np.cos(2 * np.pi * self.df['hours'] / 24)


    def add_shift_column(self, shift_values=[1, 2, 3, 4], column_name=None):
        """
        특정 컬럼에 대해 시간 지연(shift) 값을 생성하여 새로운 컬럼 추가.

        Args:
            column_name (str): Shift를 적용할 컬럼 이름.
            shift_values (list, optional): Shift 단계 리스트. 기본값은 [1, 2, 3, 4].
        """
        column_name = column_name or self.column_name  # column_name이 없으면 self.column_name 사용 #NOTE:아름 추가
        
        for shift in shift_values:
            self.df[f'{column_name}_shift{shift}'] = self.df[column_name].shift(shift)


    def fill_missing_time(self):
        """
        데이터프레임에서 누락된 시간을 채우기 위해 타임스탬프 간격을 생성.
        """
        min_time = self.df.index.min()
        max_time = self.df.index.max()
        all_times = pd.date_range(start=min_time, end=max_time, freq='15min')
        self.df = self.df.reindex(all_times)


    def interpolate_missing_values(self):
        """
        데이터프레임에서 결측치를 선형 보간으로 채움.
        """
        self.df = self.df.interpolate()


    def add_custom_features(self):
        """
        데이터프레임에 사용자 정의 파생 변수를 추가. 예: 'plug*light'.
        """
        if 'plug' in self.df.columns and 'light' in self.df.columns:
            self.df['plug*light'] = self.df['plug'] * self.df['light']


    def select_variable(self, columns):
        """
        데이터프레임에서 사용자가 선택한 열만 유지합니다.

        Args:
            columns (list): 선택할 열의 이름 리스트.

        Raises:
            ValueError: 유효하지 않은 열 이름이 포함된 경우.
        """
        available_columns = set(self.df.columns)
        requested_columns = set(columns) 
        missing_columns = requested_columns - available_columns
        if missing_columns:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_columns}")
        self.df = self.df[columns]


    def drop_na(self):
        """
        데이터프레임에서 결측치를 제거.
        """
        self.df.dropna(inplace=True)


    def save_data_to_csv(self, output_path):
        """
        데이터프레임을 CSV 파일로 저장.

        Args:
            output_path (str): 저장할 파일 경로.
        """
        self.df.to_csv(output_path, encoding='euc-kr')


    # def preprocess_data(self):
    #     """
    #     데이터 전처리를 순차적으로 수행하는 메서드.

    #     Returns:
    #         pd.DataFrame: 전처리가 완료된 데이터프레임.
    #     """
    #     # TODO: 하드 코딩 나중에 한 번 손 봐야할듯
    #     self.remove_duplicates_index()
    #     self.add_time_columns()
    #     self.add_holiday_weekend_flag()
    #     self.add_office_hours()
    #     self.add_time_cycle_features()
    #     # self.apply_iqr_to_column(column_name='전열', cond=self.df['core_time'] == 0)
    #     # self.apply_iqr_to_column(column_name='전열', cond=self.df['core_time'] == 1)
    #     self.apply_hampel_to_column(column_name='전열', cond=self.df['core_time'] == 0, window_size=8, n=3)
    #     self.apply_hampel_to_column(column_name='전열', cond=self.df['core_time'] == 1, window_size=96*3, n=3)
    #     self.fill_missing_time()
    #     self.interpolate_missing_values()
    #     self.add_shift_column(column_name='전열', shift_values=[1, 2, 3, 4])
    #     self.select_variable(columns=['전열', '전열_shift1', '전열_shift2', '전열_shift3', '전열_shift4'])
    #     self.drop_na()
    #     return self.df

    def preprocess_data(self):
        """
        데이터 전처리를 순차적으로 수행하는 메서드.

        Returns:
            pd.DataFrame: 전처리가 완료된 데이터프레임.
        """
        if not self.column_name:
            raise ValueError("전처리할 컬럼 이름(column_name)이 지정되지 않았습니다.")

        self.remove_duplicates_index()
        self.add_time_columns()
        self.add_holiday_weekend_flag()
        self.add_office_hours()
        self.add_time_cycle_features()

        # Hampel 필터 적용
        self.apply_hampel_to_column(column_name=self.column_name, cond=self.df['core_time'] == 0, window_size=8, n=3)
        self.apply_hampel_to_column(column_name=self.column_name, cond=self.df['core_time'] == 1, window_size=96 * 3, n=3)

        self.fill_missing_time()
        self.interpolate_missing_values()

        # 시간 지연(Shift) 컬럼 추가
        self.add_shift_column(column_name=self.column_name, shift_values=[1, 2, 3, 4])

        # 필요한 변수 선택
        shift_columns = [f"{self.column_name}_shift{i}" for i in range(1, 5)]
        self.select_variable(columns=[self.column_name] + shift_columns)

        # 결측치 제거
        self.drop_na()

        return self.df
    
    


