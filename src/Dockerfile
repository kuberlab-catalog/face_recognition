# This is a sample Dockerfile you can modify to deploy your own app based on face_recognition

FROM kuberlab/tensorflow:cpu-36-1.5.0-full

RUN apt-get update && apt-get install -y --fix-missing \
    build-essential \
    cmake \
    gfortran \
    git \
    wget \
    curl \
    graphicsmagick \
    libgraphicsmagick1-dev \
    libatlas-dev \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    liblapack-dev \
    libswscale-dev \
    libv4l-dev \
    libhdf5-serial-dev \
    libjasper-dev \
    libavresample-dev \
    libgphoto2-dev \
    libdc1394-22 \
    ffmpeg \
    libtbb2 \
    libtbb-dev \
    pkg-config \
    python3-dev \
    python3-numpy \
    software-properties-common \
    zip \
    && apt-get clean && rm -rf /tmp/* /var/tmp/*

RUN cd ~ && \
    mkdir -p dlib && \
    git clone -b 'v19.9' --single-branch https://github.com/davisking/dlib.git dlib/ && \
    cd  dlib/ && \
    python3 setup.py install --yes USE_AVX_INSTRUCTIONS

RUN pip install -U numpy Pillow 'scipy>=0.17.0' 'Click>=6.0' opencv-python
