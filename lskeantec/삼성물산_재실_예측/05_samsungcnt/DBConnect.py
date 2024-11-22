import os
import pandas as pd
import psycopg2
from sshtunnel import SSHTunnelForwarder
from sqlalchemy import create_engine
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수로부터 SSH 및 DB 정보 불러오기
ssh_host = os.getenv("SSH_HOST")
ssh_user = os.getenv("SSH_USER")
ssh_key_path = os.getenv("SSH_KEY_PATH")
db_host = os.getenv("DB_HOST")
db_port = int(os.getenv("DB_PORT"))
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
local_port = int(os.getenv("LOCAL_PORT"))


class DBConnector:
    def __init__(self):
        """
        DBConnector 클래스 초기화.
        SSH 터널링 및 데이터베이스 연결을 관리합니다.
        """
        self.tunnel = None
        self.engine = None

    def start_ssh_tunnel(self):
        """
        SSH 터널링을 시작합니다.
        """
        try:
            self.tunnel = SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=ssh_user,
                ssh_pkey=ssh_key_path,
                remote_bind_address=(db_host, db_port),
                local_bind_address=('0.0.0.0', local_port)
            )
            self.tunnel.start()
            print(f"SSH 터널 활성화됨 - 로컬 포트: {self.tunnel.local_bind_port}")
        except Exception as e:
            print(f"SSH 터널 설정 실패: {e}")
            raise e

    def stop_ssh_tunnel(self):
        """
        SSH 터널링을 종료합니다.
        """
        if self.tunnel:
            self.tunnel.stop()
            print("SSH 터널이 종료되었습니다.")

    def get_engine(self):
        """
        SQLAlchemy 엔진을 반환합니다.
        """
        if not self.engine:
            self.engine = create_engine(
                f"postgresql+psycopg2://{db_user}:{db_password}@localhost:{local_port}/{db_name}"
            )
        return self.engine

    def fetch_data(self, query):
        """
        데이터베이스에서 쿼리를 실행하고 DataFrame으로 반환합니다.

        Args:
            query (str): 실행할 SQL 쿼리

        Returns:
            pd.DataFrame: 쿼리 결과를 담은 데이터프레임
        """
        conn = None
        try:
            # PostgreSQL 연결
            conn = psycopg2.connect(
                host="localhost",
                port=local_port,
                database=db_name,
                user=db_user,
                password=db_password,
                options="-c client_encoding=utf8"
            )
            # Pandas DataFrame으로 결과 가져오기
            df = pd.read_sql(query, conn)
            print("데이터 가져오기 성공")
            return df
        except Exception as e:
            print(f"쿼리 실행 실패: {e}")
            return None
        finally:
            if conn:
                conn.close()
