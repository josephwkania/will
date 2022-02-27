# WILL - Weighted Injector of Luminous Lighthouses

`will` is a library to create, inject, and detect pulses from Fast Radio Bursts (FRBs) and pulsars.

<p align="center">
  <img src="https://github.com/josephwkania/will/blob/master/examples/Multi-Component_Pulse.png?raw=true" alt="Example pulse with multiple components">
</p>

<p align="center">
  <img src="https://github.com/josephwkania/will/blob/master/examples/Pulsar_with_Varying_Intensities.png" alt="Example pulsar">
</p>

# Overview
There are [many](#Other Simulators) pulsar and FRB simulators. These lack ability to handle complex band shapes (from bandstop filters, rolloff, etc).
They also try to inject pulses at a given Signal-to-Noise ratio. This signal strength methodology can lead to circular logic, in worse radio frequency
environments, the injected signal is brighter and still detectible. 

`Will` attempts the following
- Signal energy fidelity
- Custom bandpass weighting
- Straightforward Pulse Detection
- Good Documentation

# Installation
To install directly into your current Python environment
```bash
pip install git+https://github.com/josephwkania/will.git
```

If you want a local version
```bash
git clone https://github.com/josephwkania/will.git
cd will
pip install .
```

# Examples
There are [example notebooks](https://github.com/josephwkania/will/tree/master/examples) that show how to create, inject, and detect pulses.

# Question + Contributing
See [CONTRIBUTING.md](https://github.com/josephwkania/will/tree/master/CONTRIBUTING.md)

# Other Simulators
## Single Pulses
- https://github.com/kiyo-masui/burst_search/blob/master/burst_search/simulate.py
- https://github.com/kmsmith137/simpulse
- https://github.com/liamconnor/injectfrb
- https://github.com/vivgastro/Furby
- https://github.com/astrogewgaw/pataka
- https://gitlab.com/houben.ljm/frb-faker

## Pulsars
- https://github.com/SixByNine/sigproc/blob/master/src/fake.c
- https://github.com/PsrSigSim/PsrSigSim