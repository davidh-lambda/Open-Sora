#!/bin/bash
parallel --progress -j+0 'ffmpeg -v error -i {} -f null - 2>{}.err' < $1
