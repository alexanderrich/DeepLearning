#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:titan
#PBS -l mem=8GB
#PBS -l walltime=2:00:00

cd $HOME
mkdir the_gurecki
cd the_gurecki

cp /home/asr443/DeepLearning/A2/*.lua .

cp /home/asr443/DeepLearning/A2/mean.dat .
cp /home/asr443/DeepLearning/A2/std.dat .
cp /home/asr443/DeepLearning/A2/pixelmeans.dat .
cp /home/asr443/DeepLearning/A2/reportA2.pdf .

/scratch/courses/DSGA1008/bin/th result.lua

