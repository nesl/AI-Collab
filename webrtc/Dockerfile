FROM nvidia/opengl:1.1-glvnd-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update 
RUN apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get -qq install -y nodejs
#RUN apt-get -qq install nodejs=16.17.0
#RUN apt-get -qq install -y npm
RUN apt-get -qq install -y git make gcc g++

RUN git clone --branch v1.0.0 https://github.com/nesl/AI-Collab.git

WORKDIR "/AI-Collab/webrtc/"

RUN npm install

RUN openssl req -nodes -new -x509 -keyout server.key -out server.cert -subj "/C=US/ST=New Sweden/L=Stockholm /O=.../OU=.../CN=.../emailAddress=..."
