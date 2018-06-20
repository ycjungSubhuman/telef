#!/bin/bash

if [ ! -d "/home/$USER/.CLion2018.1" ]; then
    mkdir /home/$USER/.CLion2018.1
fi

./telef_run.sh clion
