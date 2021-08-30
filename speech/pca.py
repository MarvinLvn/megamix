import numpy as np
import torch
import argparse
import sys, os
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA as SKPCA
import pickle
import json
from pathlib import Path


class PCA:
    def __init__(self, feature_path, out, dim):
        self.feature_path = feature_path
        self.dim = dim
        self.out = Path(out)
        self.model = SKPCA(n_components=dim)

    def fit(self):
        print("Start fitting the PCA...")
        features = np.float32(torch.cat(torch.load(self.feature_path)['features']).cpu().numpy())
        print("input dim : ", features.shape)
        self.model.fit(features)
        print("Done.")

    def save(self):
        self.save_args()
        self.save_model()

    def save_model(self):
        filename = self.out / 'checkpoint.pkl'
        pickle.dump(self.model, open(filename, 'wb'))

    def save_args(self):
        # Save user parameters
        args = {'feature_path': self.feature_path,
                'dimension': self.dim,
                'out': str(self.out)}

        out_args = self.out / 'checkpoint_args.json'
        self.out.mkdir(parents=True, exist_ok=True)
        with open(out_args, 'w') as outfile:
            json.dump(args, outfile, indent=4)

    def load_pretrained(self, path):
        self.model = pickle.load(open(str(Path(path) / 'checkpoint.pkl'), 'rb'))
        self.dim = self.model.n_components
        self.out = path

    def project(self, X):
        return self.model.transform(X)


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--feat', type=str, required=True,
                        help='Path to the feature file (.pt)')
    parser.add_argument('--out', type=str, required=True,
                        help='Path where the model will be stored (must be a folder)')
    parser.add_argument('--dim', type=int, required=True, help='Dimension of the PCA.')
    args = parser.parse_args(argv)

    if os.path.isdir(args.out):
        raise ValueError("%s already exists.")
    else:
        os.makedirs(args.out, exist_ok=True)

    pca = PCA(args.feat, args.out, args.dim)
    pca.fit()
    pca.save()

    # Example code to project features :
    #   pca.load_pretrained('/checkpoint/marvinlvn/megamix/PCA/cpc_debug')
    #   features = np.float32(torch.cat(torch.load(args.feat)['features']).cpu().numpy())
    #   pca.project(features)



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
