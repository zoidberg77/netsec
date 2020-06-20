FROM python:3

COPY . /app
COPY ./requirements.txt /app
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 5000:5000

CMD [ "python", "./app.py" ]