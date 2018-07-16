FROM ubuntu:16.04

RUN apt-get update -y && \
    apt-get install -y python3-pip

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY Test38k.csv /app/Test38k.csv
COPY Test38k.csv /app/Train38k.csv
COPY model1_demo.h5 /app/model_full.h5
COPY run_keras_server.py /app/run_keras_server.py

ENTRYPOINT [ "python3" ]

CMD [ "run_keras_server.py" ]
