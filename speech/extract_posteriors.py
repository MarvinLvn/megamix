import torch
import argparse
import sys, os
import glob
from tqdm import tqdm
from speech.model import ClusteringModel
import numpy as np
from cpc.feature_loader import toOneHot

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
        extractor = lambda x: cpc_feature_extraction(feature_maker_X, x,
                                                     seq_norm=config['seq_norm'],
                                                     strict=config['strict'],
                                                     max_size_seq=config['max_size_seq']
                                                     )[0]
    else:
        raise ValueError("Type %s unknown. Please choose amongst [cpc, mfcc].")

    for path in tqdm(paths):
        x = extractor(path)
        if model is not None:
            x = model.predict(x)
            if not_log and model.model_type not in ['skgmm', 'kmeans']:
                x = np.exp(x)
            if model.model_type == 'kmeans':
                x = one_hot_encoding(x, model.n_components)
        output_file = path.replace(input, output).replace('.wav', '.pt')
        save_posteriors(x, output_file)

def one_hot_encoding(x, n_components):
    x = torch.from_numpy(x).to(torch.int64)
    x = x[None,:]
    x = toOneHot(x, nItems=n_components)
    x = x.squeeze(0)
    return x


def save_posteriors(posteriors, output_file):
    dirname = os.path.dirname(output_file)
    os.makedirs(dirname, exist_ok=True)
    if not torch.is_tensor(posteriors):
        posteriors = torch.from_numpy(posteriors)
    torch.save(posteriors, output_file)


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to folder containing audio files')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output folder')
    parser.add_argument('--model', type=str, required=False, default=None,
                        help='Path to the pretrained GMM or DPGMM model. If not provided, will skip this step.')
    parser.add_argument('--type', type=str, required=True, choices=['mfcc', 'cpc'],
                        help='Type of features that need to be extracted, either mfcc or cpc.')
    parser.add_argument('--model_type', type=str, required=False, choices=['gmm', 'dpgmm', 'online_gmm', 'kmeans', 'skgmm'], default=None,
                        help='Type of model to be considered.')
    parser.add_argument('--cpc_path', type=str, default=None, required=False,
                        help='Path to cpc checkpoint, only used when --type == cpc.')
    parser.add_argument('--not_log', action='store_true', help='If True, will return the responsabilities'
                                                               'instead of the log responsabilities.')
    parser.add_argument('--window', type=int, default=1000, help='Number of data point to consider for online gmm')
    parser.add_argument('--kappa', type=float, default=0.5,
                        help='Kappa, or weight associated to new points (online gmm)')
    parser.add_argument('--max_size_seq', type=int, default=20480, help='Size of the window to be considered '
                                                                         '(in number of frames).')
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
                      strict=True, seq_norm=True, max_size_seq=args.max_size_seq, gru_level=0, on_gpu=True)

    if args.model_type == 'kmeans':
        args.not_log = False # turn off this parameter
    paths = find_audio_files(args.input, debug=args.debug)
    if args.model_type is not None:
        clustering_model = ClusteringModel(model_type=args.model_type,
                                           n_components=1,
                                           out_dir=args.model,
                                           n_jobs=4,
                                           window=args.window,
                                           kappa=args.kappa
                                           )
        clustering_model.load_pretrained()
    else:
        clustering_model = None
    extract_posteriors(paths, clustering_model, args.input, args.output, config, not_log=args.not_log)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)