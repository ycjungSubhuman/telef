#!/bin/bash

nvidia-docker build --no-cache --squash -t local/telef-build .
