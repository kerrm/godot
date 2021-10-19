# godot

*godot* is a collection of Python code for performing likelihood analysis with photon weights, primarily with *Fermi* Large Area Telescope data.

# Installation
*godot* requires an installation of fermitools from the Fermi Science Support Center.  I suggest the conda distribution, but this code should work with an older distribution, too.

Once you have a working fermitools/Science Tools installation, the following two steps should suffice.  Once accomplished, treat like any other Python code!

(1) You will need to have libskymaps.so in your PYTHONPATH.  This will be somewhere like e.g. miniconda2/envs/fermi/lib/fermitools.  This doesn't happen by default when you activate the fermi conda environment, so it's helpful to make a setup script that activates the environment then additionally sets the path.

(2) You will need CALDB to point to the right place.  This *should* be taken care of when you activate the fermi environment.

# Use

The best way to get started is to look at the [paper](https://ui.adsabs.harvard.edu/abs/2019ApJ...885...92K/abstract) and then have a look at paper_plots.py, which has code to generate every plot in the paper, and thus gives a nice range of examples.

Many of the data sets I used are given nicknames and saved.  To make one from scratch, see paper_plots.py/make_j1231_pulse_profile.
