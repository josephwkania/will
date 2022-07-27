---
title: 'WILL - Weighted Injector of Luminous Lighthouses'
tags:
  - Python
  - astronomy
  - fast transients
  - neutron stars
  - fast radio bursts

authors:
  - name: Joseph W Kania
    orcid: 0000-0002-3354-3859
    affiliation: "1,2"
  - name: Kevin Bandura
    ocrid: 0000-0003-3772-2798
    affiliation: "2, 3"

affiliations:
- name: West Virginia University, Department of Physics and Astronomy, P. O. Box 6315, Morgantown 26506, WV, USA
  index: 1
- name: Center for Gravitational Waves and Cosmology, West Virginia University, Chestnut Ridge Research Building, Morgantown 26506, WV, USA
  index: 2
- name: Lane Department of Computer Science and Electrical Engineering, 1220 Evansdale Drive, PO Box 6109, Morgantown, WV 26506, USA
  index: 3

date: 30 April 2022
bibliography: paper.bib
---
 
# Summary
Radio transients, including Fast Radio Bursts (FRBs) [@Lorimer-2007], Rotating Radio Transients (RRATs) [@McLaughlin-2006], and pulsars have gathered
significant interest. These sources produce bright pulses that last on order of milliseconds. The community interest in these sources has lead to
many dedicated transient backends. These backends have greatly increased the sky-hours searched, leading to 
an increased rate of discovery of transient sources. This trend will continue as new multibeam receivers and telescopes are built. As more bursts
are discovered, more science can be extracted from the sources, their environment, and matter the bursts pass through traveling to Earth. To accomplish
this science we need to understand a search pipeline's performance and limitations. These pipelines contain many steps. The signal must be received
by the telescope, then possibly filtered, and finally digitized. The digital signal is cleaned of Radio Frequency Interference (RFI). The signal has
pass through ionized media, this causes a quadratic delay in arrival time as a function of frequency. This must be corrected for a range of possible delays, or 
Dispersion Measures (DMs). The DM-time matrix is then searched over a range of pulse widths. The candidates are then clustered together to report
one candidate that may have show up at multiple dispersion measures or widths. See @Barsdell-2012 for a discussion of a search pipeline
on a Graphical Processing Unit (GPU). Candidates are then reviewed by humans or machine learned models such as @fetch.
 
To understand the pipeline selection function, the pipeline needs to be tested over a wide variety of pulse morphologies.
Pulses can vary in sky location, arrival time, time duration, center frequency, frequency width, number of scintillation patches, scintillation phase, spectral index, scatter
time,  dispersion measure, and brightness. Even millions of pulses will undersample this eleven dimensional space. (Although not all of these dimensions have the
same impact of the pulse search. Other works have focused width and flux density, see @target_search and @UTMOST)
Getting millions of pulses is not an easy task, new instruments will not have observed many objects and
data from other instruments may not have comparable properties. Also objects might not have been observed with certain properties, for example the
the highest DM FRB reported on the [Transient Name Server](https://www.wis-tns.org/) is 3038 DM. We many want to understand the sensitivity of searches to
higher DMs, we need to make synthetic pulses, which can be done with `WILL`.

Radio Frequency Interference (RFI) are anthropomorphic signals that are inadvertently received by radio telescopes. RFI can
degrade the observation by obscuring the astronomical signal, and can also produce false positive candidates. There are many RFI removal
algorithms, some built into a pipeline [@Barsdell-2012; @Ransom], others as stand alone packages [@iqrm; @rficlean; @jess] that
clean the data before the pipeline. Ideally these filters completely remove RFI while retaining all of the pulse energy. This does not
happen in practice, some RFI remains and bright parts of the pulse are removed. To better understand how these filters interact with
the pulse, we can create fake pulses and run them through a search pipeline. We can also use `WILL`'s pulse detection to see how
the signal to noise level changes for a given width and DM.

# Statement of Need
We looked for pulse simulation software, and we found [seven existing pulse simulators](https://github.com/josephwkania/will#single-pulses).
However none met all of our needs. Many had missing or incomplete documentation, making it impossible to know how to use the software or port it to our
data. Other burst simulators failed to run on our data. Finally some simulators only work on synthetic noise, this includes both
pulse simulators. We then proceeded to develop `WILL` to overcome these problems. `WILL` uses @Your to read and write files, allowing
`WILL` to read both Filerbank [@sigproc] and PSRFITS [@psrfits] files. We also have a [documentation website](https://josephwkania.github.io/will/) 
which has function
documentation as well as example notebooks showing how to create & inject pulses, pulse detection, pulsar analysis, and an example showing `WILL`
being used with SciPy to optimize filter inputs.
 
While writing `WILL` we came up with several improvements to the fidelity of the synthetic pulses. Many pulse simulators attempt
to inject a pulse at a given Signal to Noise Ratio (SNR), the notable exception being [FRB Faker](https://gitlab.com/houben.ljm/frb-faker).
The problem with this injection methodology is that RFI can drastically change the noise level. Injecting at constant SNR will mean the
pulse will also get brighter. This will lead to overestimating the sensitivity and uniformity of the search pipeline. `will.create` creates a pulse
profile and then samples that profile for a given number of samples. Thus the total power in the pulse is known, and the pulse can then be added
to a dynamic spectra. `will.detect` can  be used to then calculate the SNR, if desired.
A unique feature of `will.create` is the ability to take into account the changing sensitivity of radio telescopes. This change can be
due to band roll off or band stop filters. The sensitivity weights can be user provided and we have functions that can estimate weights.
Taking into account sensitivity changes is important when evaluating the effectiveness of RFI filters. The variance of channels that are
blocked by a bandstop filter are close to zero. A filter may then remove any pulse found here because the pulse does not follow the surrounding
statistics. This would lead to the incorrect conclusion that the filter is malfunctioning.
 
`will.detect` provides straightforward pulse detection at given width and DM. This allows analysis of pulsar pulses over an observation.
You can also see the folded pulse dynamic spectra, time series, and SNR. You can use this functionality to see how see how the pulses
respond to RFI filters using a full pulse search pipeline, reducing the degrees of freedom searched.

`WILL` uses NumPy [@NumPy], [rich](https://github.com/Textualize/rich), SciPy [@SciPy], jess [@jess], and your [@Your].
We found `injectfrb` [@injectfrb] to have useful pulse modeling functions, these were used in [@greenburst], and built
upon in `will.create`.  

# Figure

![Injected multicomponent pulse
](https://github.com/josephwkania/will/blob/master/examples/Multi-Component_Pulse.png?raw=true)
A dedispersed four component pulse created with `WILL`, the last component is rotated compared to the other components. 
The pulse has no power between channels 2500 to 3000, correctly accounting for the bandstop filter at these frequencies.


# Acknowledgment
`WILL` was developed with support from NSF Award 2006548. We thank Devansh Agarwal and Kshitij Aggarwal for useful discussions.

# References