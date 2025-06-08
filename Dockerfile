# syntax=docker/dockerfile:1
FROM debian:latest
ARG DEBIAN_FRONTEND=noninteractive

# install app dependencies
RUN apt-get -y update && apt-get -y upgrade && apt-get install --no-install-recommends -y \
	python3 \
	python3-pip \
	python3-venv
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# create user
RUN useradd --create-home myuser
USER myuser
RUN mkdir /home/myuser/democratizer
WORKDIR /home/myuser/democratizer
COPY . .

# create and activate virtual environment using final folder name to avoid
# path issues with packages
RUN python3 -m venv /home/myuser/venv
ENV PATH="/home/myuser/venv/bin:$PATH"

RUN pip install --no-cache-dir -r requirements.txt
CMD ["python3", "-OO", "src/bot.py"]
