#!/bin/bash
if [ $(uname) == Darwin ]; then
    open video/animation.mp4
else
    code video/animation.mp4
fi
