FROM ubuntu:14.04

MAINTAINER Jing Ma <jma78@emory.edu>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y \
	wget \
	unzip \
	openjdk-7-jre \
	zip \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-numpy \
        python-pip \
        python-scipy \
        git \
        libhdf5-dev \
        graphviz \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        h5py \
        pydot-ng \
        graphviz \
        && \
    python -m ipykernel.kernelspec

# Install TensorFlow CPU version.
ENV TENSORFLOW_VERSION 1.0.0
RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-${TENSORFLOW_VERSION}-cp27-none-linux_x86_64.whl

RUN pip install git+git://github.com/Theano/Theano.git
RUN pip install keras

# Copy some examples
RUN git clone git://github.com/fchollet/keras.git


RUN git clone https://github.com/jma78/cdddvm-docker.git \
	&& (cd cdddvm-docker)

WORKDIR "/cdddvm-docker"

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

COPY keras.json /cdddvm-docker/.keras/keras.json
COPY . /root

CMD ["/bin/bash"]