# 1. 베이스 이미지: 파이썬 3.10 슬림 버전
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 먼저 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 소스 코드 복사
COPY app_generator.py .

# 5. 컨테이너가 8000 포트를 사용함을 명시 (문서용)
EXPOSE 8000

# 6. 서버 실행
#    Uvicorn을 직접 실행하여 0.0.0.0 호스트의 8000 포트에서 대기
CMD ["uvicorn", "app_generator:app", "--host", "0.0.0.0", "--port", "8000"]