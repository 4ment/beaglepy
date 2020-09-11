FROM debian:buster

MAINTAINER Mathieu Fourment <mathieu.fourment@uts.edu.au>

RUN  apt-get update -qq && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
	build-essential \
	git \
	libtool \
	python \
	python3 \
	python-dev \
	python3-dev \
	python-pip \
	python3-pip \
	python-setuptools \
	python3-setuptools \
	python-wheel \
	python3-wheel

RUN git clone https://github.com/beagle-dev/beagle-lib.git
WORKDIR beagle-lib
RUN ./autogen.sh && ./configure --without-opencl --with-jdk=no && make install

ENV CPLUS_INCLUDE_PATH /usr/local/include/libhmsbeagle-1
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

WORKDIR /beaglepy