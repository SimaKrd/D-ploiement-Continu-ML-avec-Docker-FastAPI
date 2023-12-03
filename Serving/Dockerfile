FROM python:3.10-slim 
WORKDIR /app 

COPY app.py .
COPY requirements.txt .


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD uvicorn app:app --host 0.0.0.0 --port 8080 