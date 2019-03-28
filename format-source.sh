#!/bin/bash

find . -regextype sed \
    -regex "\.\/\(src\|app\|include\)\/.*\.\(cpp\|h\)" | \
    xargs clang-format -i -style=file
