FROM debian:buster

RUN  apt-get update -qq && apt-get install -y --no-install-recommends \
    autoconf \
    automake \
	build-essential \
	ca-certificates \
	git \
	libtool \
	openjdk-11-jdk \
	python

RUN git clone https://github.com/beagle-dev/beagle-lib.git
WORKDIR beagle-lib
RUN ./autogen.sh && ./configure && make install