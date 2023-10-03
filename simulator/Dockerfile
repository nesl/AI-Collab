FROM nvidia/opengl:1.1-glvnd-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES},display

RUN apt-get -qq update && apt-get -qq install -y sudo curl

RUN apt-get -qq update && apt-get -qq install sudo gconf-service 


RUN apt-get -qq update && apt-get -qq install -y sudo libgcc1 lib32stdc++6 libasound2 libc6 libc6-i386 

RUN apt-get -qq update && apt-get -qq install -y  sudo libcairo2 libcap2 libcups2 libdbus-1-3 libexpat1 libfontconfig1 libfreetype6 libgcc1 libgconf-2-4 

RUN apt-get -qq update && apt-get -qq install -y  sudo libgdk-pixbuf2.0-0 libgl1-mesa-glx mesa-utils libglib2.0-0 libglu1-mesa libgtk2.0-0 libnspr4 libnss3 libpango1.0-0 libstdc++6 libx11-6 libxcomposite1 libxcursor1 libxdamage1 libxext6 

RUN apt-get -qq update && apt-get -qq install -y sudo libxfixes3 libxi6 libxrandr2 libxrender1 libxtst6 zlib1g debconf npm xdg-utils lsb-release libpq5 xvfb x11-apps python3-pip  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    unzip

RUN apt-get -qq update && apt-get -qq install -y git

#RUN addgroup --gid 44 groupcontainer
#RUN adduser -u 1024 -G video -h /home/containeruser -D containeruser
#USER containeruser
#WORKDIR "/home/containeruser"

RUN git clone --recurse-submodules https://github.com/nesl/AI-Collab.git

RUN python3 -m pip install -r AI-Collab/requirements.txt
RUN python3 -m pip install AI-Collab/magnebot/
RUN cp AI-Collab/controller.py "$(python3 -m pip show tdw | grep Location | cut -d ' ' -f 2)/tdw/controller.py"

WORKDIR "AI-Collab/simulator/"

RUN wget https://github.com/threedworld-mit/tdw/releases/download/v1.12.10/TDW_Linux.tar.gz
RUN tar -xzf TDW_Linux.tar.gz
#RUN echo '#!/bin/bash'"\n"'pattern="--sim-port +([0-9]+)"; tdw_command="./TDW/TDW.x86_64"; if  [[ "$@" =~ $pattern ]]; then tdw_command="$tdw_command -port ${BASH_REMATCH[1]}"; fi; nohup python3 simulation.py --no-launch-build $@ & $tdw_command' > start_sim.sh
RUN echo '#!/bin/bash'"\n"'python3 simulation.py --no-launch-build --sim-binary /AI-Collab/simulator/TDW/TDW.x86_64 $@' > start_sim.sh
RUN chmod +x start_sim.sh
#CMD ./start_sim.sh --address "https://172.17.40.64:5683"

#RUN python3 -m pip install tdw
