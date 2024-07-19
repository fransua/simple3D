# Your Library

Simple optimizer of spatial restraints using SciPy.

The optimizer uses 3 type of restraints:
 - A strong restraints to force adjacent particles (in genomic coordinates) to stay close in 3D too.
 - A weak restraint to prevent the model to collapse by applying a weak repulsive force between al pairs of particles.
 - A mid restraint that uses observed data input. If two particles are observed to be in contact (high Hi-C contact count, or observed FISH collocalization...), than the optimizer willapply a force to keep these particle together. Note that this force can be either proportional to the confidence/strength of the observation, or a fixed value.

 *We define particles as chromatin loci.*

 *See the Tuotorials folder for usage examples.*

## Installation

```bash
pip install simple3D


