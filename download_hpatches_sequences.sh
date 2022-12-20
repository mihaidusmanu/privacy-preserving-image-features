#!/usr/bin/env bash

# Create data directiory.
mkdir data
cd data

# Download the dataset.
wget http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz

# Extract the dataset.
tar xvzf hpatches-sequences-release.tar.gz

# Delete the archive.
rm hpatches-sequences-release.tar.gz

