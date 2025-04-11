import pandas as pd

# Excel 파일 읽기
excel_file = 'solar_seoul.xlsx'
df = pd.read_excel(excel_file)

# CSV 파일로 저장
csv_file = 'solar_seoul.csv'
df.to_csv(csv_file, index=False, encoding='utf-8')

print(f"변환 완료: {excel_file} -> {csv_file}") 