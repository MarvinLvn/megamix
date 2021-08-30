import numpy as np
import soundfile
import torch
from utils.audio_features import get_freqspectrum, get_mfcc, delta, raw_frames, get_fbanks
import argparse
import sys, os
import glob
from tqdm import tqdm


def get_acoustic_features(cap, config):
    try:
        data, fs = soundfile.read(cap)
    except ValueError:
        # try to repair the file
        raise ValueError("Error encountered while reading %s" % cap)

    # limit size
    if 'max_size_seq' in config:
        data = data[:config['max_size_seq']]
    # get window and frameshift size in samples
    window_size = int(fs * config['window_size'])
    frame_shift = int(fs * config['frame_shift'])

    [frames, energy] = raw_frames(data, frame_shift, window_size)
    freq_spectrum = get_freqspectrum(frames, config['alpha'], fs,
                                     window_size)
    fbanks = get_fbanks(freq_spectrum, config['n_filters'], fs)
    if config['type'] == 'fbank':
        features = fbanks
    else:
        features = get_mfcc(fbanks)
        #  add the frame energy
        features = np.concatenate([energy[:, None], features], 1)

    # optionally add the deltas and double deltas
    if config['delta']:
        single_delta = delta(features, 2)
        double_delta = delta(single_delta, 2)
        features = np.concatenate([features, single_delta, double_delta], 1)
    return features


def acoustic_audio_features(paths, config):
    # Adapted from https://github.com/gchrupala/speech2image/blob/master/preprocessing/audio_features.py#L45
    if config['type'] != 'mfcc' and config['type'] != 'fbank':
        raise NotImplementedError()
    output = []
    for cap in tqdm(paths):
        features = get_acoustic_features(cap, config)
        output.append(torch.from_numpy(features))
    return output


def cpc_audio_representations(paths, config, time_step=0.01, max_h=None):
    from speech.utils.audio_features import load_feature_maker_CPC, cpc_feature_extraction
    feature_maker_X = load_feature_maker_CPC(config['model_path'], gru_level=config['gru_level'], on_gpu=config['on_gpu'])
    output = []
    max_s = max_h*3600 if max_h is not None else None
    nb_frames = 0
    for cap in paths:
        print("Processing {}".format(cap))
        features = cpc_feature_extraction(feature_maker_X, cap,
                                          seq_norm=config['seq_norm'],
                                          strict=config['strict'],
                                          max_size_seq=config['max_size_seq'])[0]
        output.append(features)
        nb_frames += features.shape[0]
        if max_s is not None and nb_frames * time_step > max_s:
            break
    return output


def audio_features(paths, config, time_step=0.01, max_h=40):
    if config['type'] == 'mfcc' or config['type'] == 'fbank':
        return acoustic_audio_features(paths, config)
    elif config['type'] == 'cpc':
        return cpc_audio_representations(paths, config, time_step, max_h)
    else:
        raise NotImplementedError("Can't find audio feature extraction of type %s" % config['type'])


def save_features(features, paths, out_path):
    out_dict = dict(features=features, filenames=paths[0:len(features)])
    print("Size of all features : %.2f MB" % sys.getsizeof(features))
    torch.save(out_dict, out_path)


def find_audio_files(path, extension='.wav', debug=False):
    paths = glob.glob(os.path.join(path, '**/*%s' % extension), recursive=True)
    if len(paths) == 0:
        raise ValueError("The folder you provided doesn't contain any %s files" % extension)
    if debug:
        paths = paths[:50]
    print("Found %d audio files." % len(paths))
    return paths


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--db', type=str, required=True,
                        help='Path to folder containing audio files')
    parser.add_argument('--out', type=str, required=True,
                        help='Path where the output features will be stored (.pt)')
    parser.add_argument('--type', type=str, required=True, choices=['mfcc', 'cpc'],
                        help='Type of features that need to be extracted, either mfcc or cpc.')
    parser.add_argument('--cpc_path', type=str, default=None, required=False,
                        help='Path to cpc checkpoint, only used when --type == cpc.')
    parser.add_argument('--max_size_seq', type=int, default=20480, help='Size of the window to be considered '
                                                                         '(in number of frames).')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will consider only first 20 audio files.')
    parser.add_argument('--time_step', type=float, default=0.01, help='Size of the CPC time (default to 10 ms)')
    parser.add_argument('--max_h', type=int, default=None,
                        help='Maximum audio duration to extract in (h). Default to None : consider all audios found')
    args = parser.parse_args(argv)

    if args.out[-3:] != ".pt":
        raise ValueError("Parameter --out should end with .pt.")

    if args.cpc_path is not None and args.cpc_path[-3:] != '.pt':
        raise ValueError("Parameter --cpc_path should end with .pt.")

    if args.type == 'mfcc':
        config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40, window_size=0.025, frame_shift=0.010)
    elif args.type == 'cpc':
        assert args.cpc_path is not None
        config = dict(type='cpc', model_path=args.cpc_path,
                      strict=True, seq_norm=True, max_size_seq=args.max_size_seq,
                      gru_level=0, on_gpu=True)

    paths = find_audio_files(args.db, debug=args.debug)
    features = audio_features(paths, config, time_step=args.time_step, max_h=args.max_h)
    save_features(features, paths, args.out)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
