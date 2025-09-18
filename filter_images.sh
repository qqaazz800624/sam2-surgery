#!/bin/bash

DIR=video_images/4430566

find $DIR -maxdepth 1 \
    -type f \
    -name '0[8][1][0-9][0-9].jpg' -print 

find $DIR -maxdepth 1 \
    -type f \
    -name '0[8][1][0-9][0-9].jpg' -delete