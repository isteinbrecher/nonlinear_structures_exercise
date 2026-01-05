# The MIT License (MIT)
#
# Copyright (c) 2018-2025 BeamMe Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
ARG BASE_IMAGE_TAG=latest
FROM ghcr.io/4c-multiphysics/4c:main

USER root
RUN apt-get upgrade
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y build-essential
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y xvfb libgl1 libglx-mesa0

USER root
RUN chmod 777 -R /home/user

USER ubuntu
COPY --chown=ubuntu . /home/ubuntu/nonlinear_structures_exercise

WORKDIR /home/ubuntu

RUN python3 -m venv beamme-env
RUN . beamme-env/bin/activate && pip install --upgrade pip
WORKDIR /home/ubuntu/nonlinear_structures_exercise
RUN pwd
RUN ls -l
# ENV PATH="/home/ubuntu/beamme-env/bin:$PATH"
# RUN pip install -e .[fourc,dev]
# RUN pip install --no-cache-dir notebook jupyterlab jupyterhub tame
# RUN pip install pyvista[jupyter] trame ipywidgets
