import torch
import argparse
import sys, os
import glob
from tqdm import tqdm
from speech.model import ClusteringModel
import numpy as np


def find_audio_files(path, extension='.wav', debug=False):
    paths = glob.glob(os.path.join(path, '**/*%s' % extension), recursive=True)
    if len(paths) == 0:
        raise ValueError("The folder you provided doesn't contain any %s files" % extension)
    if debug:
        paths = paths[:50]
    print("Found %d audio files." % len(paths))
    return paths


def extract_posteriors(paths, model, input, output, config, not_log=False):
    if config['type'] == 'mfcc':
        from speech.extract_features import get_acoustic_features
        extractor = lambda x: get_acoustic_features(x, config)
    elif config['type'] == 'cpc':
        from speech.utils.audio_features import load_feature_maker_CPC, cpc_feature_extraction
        feature_maker_X = load_feature_maker_CPC(config['model_path'], gru_level=config['gru_level'],
                                                 on_gpu=config['on_gpu'])
        extractor = lambda x: cpc_feature_extraction(feature_maker_X, x)[0]
    else:
        raise ValueError("Type %s unknown. Please choose amongst [cpc, mfcc].")

    for path in tqdm(paths):
        input_features = extractor(path)
        posteriors = model.predict(input_features)
        if not_log:
            posteriors = np.exp(posteriors)
        output_file = path.replace(input, output).replace('.wav', '.pt')
        save_posteriors(posteriors, output_file)


def save_posteriors(posteriors, output_file):
    dirname = os.path.dirname(output_file)
    os.makedirs(dirname, exist_ok=True)
    torch.save(torch.from_numpy(posteriors), output_file)


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to folder containing audio files')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output folder')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the pretrained GMM or DPGMM model.')
    parser.add_argument('--type', type=str, required=True, choices=['mfcc', 'cpc'],
                        help='Type of features that need to be extracted, either mfcc or cpc.')
    parser.add_argument('--model_type', type=str, required=True, choices=['gmm', 'dpgmm'],
                        help='Type of model to be considered. Either gmm or dpgmm.')
    parser.add_argument('--cpc_path', type=str, default=None, required=False,
                        help='Path to cpc checkpoint, only used when --type == cpc.')
    parser.add_argument('--not_log', action='store_true', help='If True, will return the responsabilities'
                                                               'instead of the log responsabilities.')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will consider only first 20 audio files.')
    args = parser.parse_args(argv)

    if args.cpc_path is not None and args.cpc_path[-3:] != '.pt':
        raise ValueError("Parameter --cpc_path should end with .pt.")

    if args.type == 'mfcc':
        config = dict(type='mfcc', delta=True, alpha=0.97, n_filters=40, window_size=0.025, frame_shift=0.010)
    elif args.type == 'cpc':
        assert args.cpc_path is not None
        config = dict(type='cpc', model_path=args.cpc_path,
                      strict=False, seq_norm=False, max_size_seq=10240,
                      gru_level=0, on_gpu=True)

    paths = find_audio_files(args.input, debug=args.debug)
    clustering_model = ClusteringModel(model_type=args.model_type,
                                       n_components=1,
                                       frac_train=0,
                                       out_dir=args.model,
                                       n_jobs=4)
    clustering_model.load_pretrained()
    extract_posteriors(paths, clustering_model, args.input, args.output, config, not_log=args.not_log)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)