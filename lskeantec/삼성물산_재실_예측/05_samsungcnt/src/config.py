import os

# 프로젝트 루트 디렉토리
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 데이터 디렉토리
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 원본 데이터 디렉토리 및 파일 경로
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
raw_data_file = f"{floor}F_{zone_side}_raw_data.csv"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, raw_data_file)

# 전처리된 데이터 디렉토리 및 파일 경로
PREPROC_DATA_DIR = os.path.join(DATA_DIR, 'preproc')
PREPROC_DATA_PATH = os.path.join(PREPROC_DATA_DIR, 'data_preproc.csv')

# 디렉토리 확인 및 생성
for directory in [RAW_DATA_DIR, PREPROC_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# 경로 출력 (확인용)
if __name__ == "__main__":
    print(f"RAW_DATA_PATH: {RAW_DATA_PATH}")
    print(f"PREPROC_DATA_PATH: {PREPROC_DATA_PATH}")
