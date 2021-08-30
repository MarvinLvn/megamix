from sklearn.decomposition import PCA
import argparse
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

def main(argv):
    parser = argparse.ArgumentParser(
        description='Train PCA on saved features (.pt file)')
    parser.add_argument('--feat', type=str, required=True,
                        help='Path to the .pt file containing input features.')
    parser.add_argument('--out', type=str, required=True,
                        help='Path to the output .png file.')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of components to consider in the PCA.')
    parser.add_argument('--debug', action='store_true',
                        help='If True, will consider only a 100 frames to train the PCA.')

    args = parser.parse_args(argv)

    if args.out[-4:] != '.png':
        raise ValueError("Parameter --output should end with .png")

    training_set = np.float32(torch.cat(torch.load(args.feat)['features']).cpu().numpy())
    if args.debug:
        training_set = training_set[:100, :]

    pca = PCA(n_components=args.n_components).fit(training_set)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig(args.out)

if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)
