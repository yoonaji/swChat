FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. requirements.txt 먼저 복사 및 설치
#    (파일이 변경되지 않으면 캐시를 사용해 빌드 속도 향상)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 소스 코드 복사
COPY . .

# 5. 컨테이너가 8001 포트를 사용함을 명시 (문서용)
EXPOSE 8001

# 6. 서버 실행
#    Uvicorn을 직접 실행하여 0.0.0.0 호스트의 8001 포트에서 대기
CMD ["python", "app_retriever.py"] 