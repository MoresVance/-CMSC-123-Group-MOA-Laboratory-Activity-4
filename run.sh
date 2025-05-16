#!/bin/bash

while true; do
    echo "Running Octave script at $(date)"
    octave --quiet Main.m
    sleep 1  # optional: wait 1 second between runs
done

