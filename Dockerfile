# DHgeN: a Python module for generating District Heating Networks layouts.

# Copyright (c) 2022 Idiap Research Institute, http://www.idiap.ch/
# Written by Giuseppe Peronato <Giuseppe.Peronato@idiap.ch>

# This file is part of DHgeN.

# DHgeN is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.

# DHgeN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DHgeN. If not, see <http://www.gnu.org/licenses/>

FROM python:3.8-slim

RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
build-essential gcc mono-mcs libboost-all-dev \
libgdal-dev \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/goderik01/PACE2018.git
RUN cd PACE2018 && make
RUN cp /PACE2018/bin/star_contractions_test /bin/star_contractions_test

COPY . DHgeN
WORKDIR /DHgeN
RUN pip3 install .[test]
