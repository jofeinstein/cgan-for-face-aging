#!/usr/bin/env bash

mkdir -p data
cd data
wget https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar
echo "Untarring imdb_crop.tar"
tar -xf imdb_crop.tar
rm imdb_crop.tar
mkdir -p training_logs
mkdir -p weights