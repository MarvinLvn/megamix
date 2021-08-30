import torch
import argparse
import sys, os
import glob
from tqdm import tqdm
from speech.model import ClusteringModel
import numpy as np
from cpc.feature_loader import toOneHot
import h5py
from pathlib import Path


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
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model folder.')

    args = parser.parse_args(argv)

    model = ClusteringModel(model_type="dpgmm",
                   n_components=1,
                   out_dir=args.model_path,
                   n_jobs=1)
    model.load_pretrained()
    print(model.model._limiting_model(model.train_data))
    exit()


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)