FROM continuumio/miniconda3

RUN mkdir -p /app
RUN mkdir -p /storage && chmod 777 /storage

RUN conda update conda && conda install -n base conda-libmamba-solver && conda config --set solver libmamba

# pytorch with gpu support
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes

COPY requirements.txt requirements.txt 
RUN conda install -c conda-forge --yes --file requirements.txt

# dlib with gpu support
RUN conda install cmake 
RUN pip install --upgrade --no-deps --force-reinstall dlib

# переходим в директорию
WORKDIR /app 

RUN mkdir -p /.local && chmod 777 /.local
RUN mkdir -p /.cache && chmod 777 /.cache
RUN mkdir -p /.config && chmod 777 /.config

ENV PYTHONPATH "${PYTHONPATH}:/"

# запуск в dev
ENTRYPOINT /bin/bash