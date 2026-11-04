# 使用 Python 環境
FROM python:3.12


# Container 工作目錄
WORKDIR /app


 # 複製程式
COPY . .


# 安裝套件
RUN pip install -r requirements.txt


# 啟動 API
CMD ["python", "app.py"]