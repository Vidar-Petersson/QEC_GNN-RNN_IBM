#!/usr/bin/env bash
# Build the Apptainer image used on the Alvis cluster
apptainer build bash_container.sif bash_container.def
