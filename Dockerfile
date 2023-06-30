# pull official base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ADD . /usr/src/app

# set work directory
WORKDIR /usr/src/app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# update package
RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6

# install dependencies
COPY ./requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN python setup.py develop

# copy entrypoint.sh
COPY ./entrypoint.sh .

# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]