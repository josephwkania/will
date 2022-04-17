# WILL - Weighted Injector of Luminous Lighthouses

`will` is a library to create, inject, and detect pulses from Fast Radio Bursts (FRBs) and pulsars.

<p align="center">
  <img src="https://github.com/josephwkania/will/blob/master/examples/Multi-Component_Pulse.png?raw=true" alt="Example pulse with multiple components">
</p>

<p align="center">
  <img src="https://github.com/josephwkania/will/blob/master/examples/Pulsar_with_Varying_Intensities.png" alt="Example pulsar">
</p>

# Overview
There are [many](#Other-Simulators) pulsar and FRB simulators. These lack ability to handle complex band shapes (from bandstop filters, rolloff, etc).
They also try to inject pulses at a given Signal-to-Noise ratio. This signal strength methodology can lead to circular logic, in worse radio frequency
environments, the injected signal is brighter and still detectible. 

`Will` attempts the following
- Signal energy fidelity
- Custom bandpass weighting
- Straightforward Pulse Detection
- Good Documentation

There are three submodules `will.create`, `will.inject`, and `will.detect`. 

## `create` 
- `GaussPulse` can make multiple independent component pulses.
- `SimpleGaussPulse` created pulses that are not correlated in frequency and time
- `filter_weights` Uses Gaussian smoothing to create bandpass weights model filter and rolloff
- `clone_spectra` makes dynamic spectra with Gaussian noise that copies statistics
- `log_normal_from_stats` creates a log-normal distro. with given median and Stand. Dev.
- `sort_subarrays` gives correlation to pulse powers
- `dynamic_from_statistics` Creates a noise dynamic spectra w/ given STD and median per channel
- `clone_spectra` Makes a noise clone of a give dynamic spectra

## `inject`
- `inject_constant_into_file` inject pulse(s) of the same intensity
- `inject_distribution_into_file` allows you to specify the pulse energies

## `detect`
- `find_first_pulse` Helps find the first pulse in a file
- `search_file` search a file for periodic pulses at given DM and pulse width

## Documentation
We have a [docs website](https://josephwkania.github.io/will/)
which contains the examples and and [API documentation](https://josephwkania.github.io/will/py-modindex.html)

# Installation
To install directly into your current Python environment
```bash
pip install git+https://github.com/josephwkania/will.git
```

If you want a local version
```bash
git clone https://github.com/josephwkania/will.git
pip install will

For tests `pip install will[tests]`, for docs `pip install will[docs]`
```

# Examples
There are [example notebooks](https://github.com/josephwkania/will/tree/master/examples) that show how to create, inject, and detect pulses.

# Questions + Contributing
See [CONTRIBUTING.md](https://github.com/josephwkania/will/tree/master/CONTRIBUTING.md)

# Other Simulators
## Single Pulses
- https://github.com/kiyo-masui/burst_search/blob/master/burst_search/simulate.py
- https://github.com/kmsmith137/simpulse
- https://github.com/liamconnor/injectfrb
- https://github.com/vivgastro/Furby
- https://github.com/astrogewgaw/pataka
- https://github.com/jayanthc/fakefrb
- https://gitlab.com/houben.ljm/frb-faker

## Pulsars
- https://github.com/SixByNine/sigproc/blob/master/src/fake.c
- https://github.com/PsrSigSim/PsrSigSim