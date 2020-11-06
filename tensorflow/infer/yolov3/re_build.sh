#!/bin/bash

 cd build
 rm -f CMakeCache.txt

 cmake ..
 make

 mv yolov3_exe ../
