import pandas as pd
import glob
import os

# solar_seoul.csv 읽기
solar_df = pd.read_csv('solar_seoul.csv')

# weather 폴더의 모든 CSV 파일 읽기
weather_files = glob.glob('weather/weather_108_*.csv')
weather_dfs = []

for file in weather_files:
    # EUC-KR 인코딩으로 날씨 데이터 읽기
    df = pd.read_csv(file, encoding='euc-kr')
    weather_dfs.append(df)

# 모든 날씨 데이터 합치기
weather_df = pd.concat(weather_dfs, ignore_index=True)

# 날씨 데이터의 날짜/시간 컬럼 처리
weather_df['datetime'] = pd.to_datetime(weather_df['일시'])
weather_df['date'] = weather_df['datetime'].dt.date
weather_df['time'] = weather_df['datetime'].dt.hour

# solar_seoul.csv의 날짜 형식 변환
solar_df['date'] = pd.to_datetime(solar_df['date']).dt.date

# 두 데이터셋 병합
merged_df = pd.merge(solar_df, weather_df, on=['date', 'time'], how='inner')

# 결과 저장
merged_df.to_csv('merged_data.csv', index=False)
print("데이터 병합이 완료되었습니다.") 