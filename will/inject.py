#!/usr/bin/env python3
"""
Pulse injection routines
"""
import logging
from typing import Union

import numpy as np
from rich.progress import track
from your import Your
from your.formats.filwriter import Writer, sigproc_object_from_writer


def inject_into_file(
    yr_input: Your,
    pulse: np.ndarray,
    start: int = 0,
    period: float = None,
    gulp: int = 2 ** 16,
    out_fil: str = None,
    clip_powers: bool = True,
) -> Union[np.ndarray, None]:
    """
    Inject a pulse into a file.

    Args:
        yr_input - Your object for file to inject pulse

        pulse - 2D array that contains the pulse

        start - Start sample of the pulse

        period - Sample period of injected pulse

        out_fil - Out filterbank, if None, returns array

        clip_powers - Clip powers instead of wrap

    Returns
        None if out_fil, else dynamic spectra with injected pulse
    """
    nsamples, nchans = pulse.shape

    d_chan = yr_input.your_header.native_nchans
    if d_chan != nchans:
        raise RuntimeError(
            f"Mismatch in number of channels, pulse: {nchans}, data: {d_chan}"
        )

    if 2 * nsamples > gulp:
        raise RuntimeError(f"{gulp=} is not two times larger that {nsamples=}")

    if clip_powers:
        iinfo = np.iinfo(yr_input.your_header.dtype)
        dtype = np.uint64
    else:
        dtype = yr_input.your_header.dtype
        pulse = pulse.astype(dtype)

    if out_fil is None:
        array = np.zeros(
            (yr_input.your_header.nspectra, yr_input.your_header.nchans),
            dtype=dtype,
        )
    else:
        wrt = Writer(yr_input, outname=out_fil)
        sigproc_obj = sigproc_object_from_writer(wrt)
        sigproc_obj.write_header(out_fil)

    if start < 0:
        raise NotImplementedError(f"Negative start values not supported, {start=}")
        # start_block = False
    # else:
    start_block = True

    for j in track(range(0, yr_input.your_header.nspectra, gulp)):
        logging.debug("Adding pulse(s) to block starting at %i", j)
        chunk_end = j + gulp
        if chunk_end < yr_input.your_header.nspectra:
            data = yr_input.get_data(j, gulp)
        else:
            chunk_end = yr_input.your_header.nspectra  # - j
            data = yr_input.get_data(j, chunk_end)

        if clip_powers:
            data = data.astype(dtype)

        if not start_block:
            # Pulse comes from last section
            _start = start - period
            end = _start + nsamples
            while _start <= j <= end <= chunk_end:
                offset = np.around(_start).astype(int) - j + nsamples
                data[:offset] += pulse[-offset:]
                if period is not None:
                    _start -= period
                    end -= period

        while j <= start <= chunk_end:
            offset = np.around(start).astype(int) - j
            if chunk_end > start + nsamples:
                # pulse fits into chunk
                data[offset : offset + nsamples] += pulse

            else:
                # pulse tail is over end of black
                cut = chunk_end - np.around(start).astype(int)
                data[offset:] += pulse[:cut]
                start_block = False

            if period is not None:
                start += period
            else:
                break

        if clip_powers:
            # Clip to stop wrapping
            np.clip(data, iinfo.min, iinfo.max, out=data)

            data = data.astype(dtype)

        if out_fil is None:
            array[j:chunk_end] = data
        else:
            sigproc_obj.append_spectra(data, out_fil)

    if out_fil is None:
        return data
    return None