FROM ubuntu:latest

RUN apt-get update && \
  apt-get install -y git python3 python3-dev python3-pip curl build-essential

RUN pip3 install ovos-plugin-manager
RUN pip3 install ovos-stt-http-server==0.0.2a1
RUN pip3 install SpeechRecognition==3.8.1

COPY . /tmp/neon-stt-plugin-scribosermo
RUN pip3 install /tmp/neon-stt-plugin-scribosermo
RUN scribosermo-modeldl
ENTRYPOINT ovos-stt-server --engine neon-stt-plugin-scribosermo