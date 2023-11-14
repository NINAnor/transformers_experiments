FROM python:3.10

# Install the dependancies for cv2
ARG PACKAGES="ffmpeg build-essential"

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -qq $PACKAGES && \
    rm -rf /var/lib/apt/lists/*

# Install poetry (alternative to conda for package management)
RUN pip3 install poetry 

# Install all the poetry dependancies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry lock --no-update && \
    poetry install --no-root

RUN python -m spacy download en_core_web_sm

# Make a folder for my scripts
COPY ./ /app/