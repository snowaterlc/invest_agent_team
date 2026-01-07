FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY invest_team_agent_deepseek.py .
COPY .env .
RUN mkdir -p /app/cache
CMD ["python", "invest_team_agent_deepseek.py"]