FROM nvcr.io/nvidia/tensorrt:24.02-py3

ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends ffmpeg
RUN apt-get install -y python3-pip
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'NCCL_SOCKET_IFNAME=lo' >> ~/.bashrc


WORKDIR /app
COPY requirements.txt requirements.txt
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "bash" ]