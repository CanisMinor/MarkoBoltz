# Simulation of Rotational Diffusion on a Surface

## Introduction
This command line tool is an interface to a program simulating the scattering properties of molecules are undergoing rotational on a surface.

There are two underlying models:
 - discrete rotation (wehre the atoms in the molecule hop discretely between minima in the surface potential) model based on a Monte Carlo model
 - continuous rotation model based on Langevin dynamics
 
Each model simulates a random walk, generating a rotational trajectory.  The elastic coherent structure factor (EISF) and the extinction efficiency factor (QEA) are calculated as functions of the momentum transfer.

## Usage
This tool is under construction and currently only contains two hard-coded models; one continuous model with all parameters hard-coded, and one discrete model with all parameters hard-coded.

The tool is used as follows; for the hard-coded discrete model, 

    python main.py --type 0
   
and for the hard-coded continuous model,
   
   pyhon main.py --type 1
