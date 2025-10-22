FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN apt-get update
&& apt-get install -y build-essential git curl
&& pip install --upgrade pip
&& pip install -r requirements.txt
&& apt-get purge -y --auto-remove build-essential
COPY . /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
