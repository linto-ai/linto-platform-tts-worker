FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

WORKDIR /srv/app

RUN apt-get update && \
    apt-get install -y libsndfile1 espeak nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY requirements.txt /srv/app
RUN pip install -r requirements.txt

COPY LinTTS /srv/app/LinSTT
COPY manifest.json /srv/app/LinSTT

CMD python /srv/app/LinSTT/tts_worker.py $CONFIG_PATH $MODEL_PATH --debug