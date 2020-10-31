#!/bin/sh

mkdir data
cd data
mkdir raw
cd raw
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar xvzf speech_commands_v0.01.tar.gz
rm speech_commands_v0.01.tar.gz
cd ../..

