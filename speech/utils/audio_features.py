#!/usr/bin/env python3
"""
Comes from https://github.com/spokenlanguage/platalea
"""
import argparse
import json
import math
import os

import numpy
from cpc.feature_loader import buildFeature, FeatureModule, loadModel
from scipy.fftpack import dct

from speech.utils.filters import apply_filterbanks, filter_centers, create_filterbanks
from speech.utils.preproc import four, pad, preemph, hamming, notch


# this file contains the main bulk of the actuall feature creation functions


def delta(data, N):
    # calculate delta features, n is the number of frames to look forward and backward

    # create a delta array of the right shape
    dt = numpy.zeros(data.shape)
    # pad data with first and last frame for size of n
    for n in range(N):
        data = numpy.row_stack((data[0, :], data, data[-1, :]))
    # calc n*c[x+n] + c[x-n] for n in Nand sum them
    for n in range(1, N + 1):
        dt += numpy.array([n * (data[x+n, :] - data[x-n, :]) for x in range(N, len(data) - N)])
    # normalise the deltas for the size of N
    normalise = 2 * sum([numpy.power(x, 2) for x in range(1, N+1)])

    dt = dt/normalise

    return (dt)


def raw_frames(data, frame_shift, window_size):
    # this function cuts the data into frames and calculates each frames' accuracy

    # determine the number of frames to be extracted
    nframes = math.floor(data.size/frame_shift)
    # apply notch filter
    notched_data = notch(data)
    # pad the data
    data = pad(notched_data, window_size, frame_shift)
    # slice the frames from the wav file
    # keep a list with the frames and all the values of the samples and
    # list with the start and end sample# of each frame
    frames = []
    energy = []

    for f in range(0, nframes):
        frame = data[(f * frame_shift):(f * frame_shift + window_size)]
        energy.append(numpy.log(numpy.sum(numpy.square(frame), 0)))
        frames.append(frame)

    frames = numpy.array(frames)
    energy = numpy.array(energy)
    # if energy is 0 , the log can not be taken(results in -inf) so we set the
    # log energy to -50 (log of 2e-22 or approx 0 )
    energy[energy == numpy.log(0)] = -50

    return (frames, energy)


def get_freqspectrum(frames, alpha, fs, window_size):
    # this function prepares the raw frames for conversion to frequency spectrum
    # and applies fft

    # apply preemphasis
    frames = preemph(frames, alpha)
    # apply hamming windowing
    frames = hamming(frames)
    # apply fft
    freq_spectrum = four(frames, fs, window_size)

    return freq_spectrum


def get_fbanks(freq_spectrum, nfilters, fs):
    #  this function calculates the filters and creates filterbank features from
    #  the fft features

    # get the frequencies corresponding to the bins returned by the fft
    xf = numpy.linspace(0.0, fs/2, numpy.shape(freq_spectrum)[1])
    # get the filter frequencies
    fc = filter_centers(nfilters, fs, xf)
    # create filterbanks
    filterbanks = create_filterbanks(nfilters, xf, fc)
    # apply filterbanks
    fbanks = apply_filterbanks(freq_spectrum, filterbanks)

    return fbanks


def get_mfcc(fbanks):
    # this function creates mfccs from the fbank features

    # apply discrete cosine transform to get mfccs. According to convention,
    # we discard the first filterbank (which is roughly equal to the method
    # where we only space filters from 1000hz onwards)
    mfcc = dct(fbanks[:, 1:])
    # discard the first coefficient of the mffc as well and take the next 13
    # coefficients.
    mfcc = mfcc[:, 1:13]

    return mfcc

# CPC features loader

def read_args(path_args):
    print(f"Loading args from {path_args}")
    with open(path_args, 'r') as file:
        args = argparse.Namespace(**json.load(file))
    return args


def load_feature_maker_CPC(cp_path, gru_level=-1, on_gpu=True):
    assert cp_path[-3:] == ".pt"
    assert os.path.exists(cp_path), \
        f"CPC path at {cp_path} does not exist!!"

    pathConfig = os.path.join(os.path.dirname(cp_path), "checkpoint_args.json")
    CPC_args = read_args(pathConfig)

    # Load FeatureMaker
    if gru_level is not None and gru_level > 0:
        updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
    else:
        updateConfig = None

    #model = loadModel([cp_path], updateConfig=updateConfig)[0]
    model = loadModel([cp_path])[0]

    feature_maker = FeatureModule(model, CPC_args.onEncoder)
    feature_maker.eval()
    if on_gpu:
        feature_maker.cuda()
    return feature_maker


def cpc_feature_extraction(feature_maker, x, seq_norm=False, strict=True,
                           max_size_seq=10240):
    return buildFeature(feature_maker, x,
                        strict=strict,
                        maxSizeSeq=max_size_seq,
                        seqNorm=seq_norm)
