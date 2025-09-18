#!/bin/bash

DIR=video_images/4386880

find $DIR -maxdepth 1 \
    -type f \
    -name '0[3][1-9][0-9][0-9].jpg' -print 

find $DIR -maxdepth 1 \
    -type f \
    -name '0[3][1-9][0-9][0-9].jpg' -delete