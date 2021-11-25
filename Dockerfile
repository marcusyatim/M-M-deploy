# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

# Set working directory
WORKDIR /app

# Copy everything in currect directory into the app directory.
ADD . /app

# Install all of the requirements
RUN pip3 install -r requirements.txt

RUN python -c "import nltk; nltk.download('wordnet')"
RUN python -c "import nltk; nltk.download('punkt')"
RUN cp -r /root/nltk_data /usr/local/share/nltk_data