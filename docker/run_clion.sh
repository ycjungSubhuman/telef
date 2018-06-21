#!/bin/bash

if [ ! -d "/home/$USER/.CLionDocker" ]; then
    mkdir /home/$USER/.CLionDocker
fi

./telef_run.sh clion
