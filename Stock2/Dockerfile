# Python 공식 이미지를 사용
# FROM python:3.8-slim

# COPY . /app

# 작업 디렉토리 설정
# WORKDIR /app

# 의존성 파일 복사 (requirements.txt 필요, 이 부분을 직접 추가해야 합니다.)
# COPY requirements.txt ./

# 필요한 Python 라이브러리 설치
# RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
# COPY app.py model.py ProcessDataset.py supplydata.py weights2.h5 ./

# Flask가 사용할 포트 번호
# EXPOSE 5000

# Flask 애플리케이션 실행
# CMD ["python", "app.py"]

# RUN mkdir -p /app/static

# -------------------------------------------------------

# Python 공식 이미지를 사용
FROM python:3.8-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . /app

# Flask가 사용할 포트 번호
EXPOSE 5000

# Flask 애플리케이션 실행
CMD ["python", "app.py"]



