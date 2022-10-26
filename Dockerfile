FROM python:3.7

RUN pip install --upgrade pip

COPY ./requirements.txt /var/www/requirements.txt
RUN pip install -r /var/www/requirements.txt


COPY . /app/

copy client/ /app/client

WORKDIR /app/

ENTRYPOINT ["python"]

EXPOSE 5000

CMD ["app.py"]
