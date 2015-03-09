#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l mem=64GB
#PBS -l walltime=2:00:00

cd $HOME
mkdir the_gurecki
cd the_gurecki

cp /home/asr443/DeepLearning/A2/*.lua .

cp /home/asr443/DeepLearning/A2/mean.dat .
cp /home/asr443/DeepLearning/A2/std.dat .
cp /home/asr443/DeepLearning/A2/pixelmeans.dat .
### COPY PDF HERE!!!###

/scratch/courses/DSGA1008/bin/th result.lua

