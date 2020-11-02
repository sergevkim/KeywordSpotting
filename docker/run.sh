#!/bin/sh

NAME=$1
PORT=$2

docker container stop -t 0 $NAME

USER_NAME=$(basename $HOME)
echo "Run as user '$USER_NAME'"

HOST_PATH=$(readlink -f "$PWD/../../")
DOCKER_PATH="/root/$NAME"
DATASETS_PATH="/datasets"

cd $HOST_PATH

(docker container run \
    --rm \
    -d \
    --dns 217.10.39.4 --dns 8.8.8.8 \
    --privileged \
    -v $HOST_PATH:/$DOCKER_PATH \
    -v $DATASETS_PATH:$DATASETS_PATH \
    -v $HOME:/home/$USER_NAME \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --expose $PORT \
    -p $PORT:$PORT \
    -h $NAME \
    --name $NAME \
$NAME) || true

docker container exec -it -w $DOCKER_PATH $NAME bash -c \
    "cd docker && ./setup/install.sh && cd /home/ && \
    jupyter notebook --port=${PORT} --ip=0.0.0.0 --no-browser --allow-root && \
    bash"

