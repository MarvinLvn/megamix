import argparse
from megamix.batch import DPVariationalGaussianMixture
from megamix.batch import GaussianMixture
from megamix.batch import GaussianMixture as OnlineGaussianMixture
import numpy as np
import torch
import sys
import json
import h5py
from pathlib import Path
import os
np.random.seed(2021)

class ClusteringModel:
    def __init__(self, model_type, n_components, frac_train, out_dir, n_jobs, simplify_dpgmm=True):
        assert model_type in ['gmm', 'dpgmm', 'online_gmm']
        assert 0 <= frac_train <= 1
        self.model_type = model_type
        self.n_components = n_components
        self.frac_train = frac_train
        self.simplify_dpgmm = simplify_dpgmm
        self.out_dir = Path(out_dir)
        self.n_jobs = n_jobs
        self.init_model(model_type, n_components, n_jobs)

    def init_model(self, model_type, n_components, n_jobs):
        if model_type == 'gmm':
            self.model = GaussianMixture(n_components, n_jobs=n_jobs)
        elif model_type == 'dpgmm':
            self.model = DPVariationalGaussianMixture(n_components, n_jobs=n_jobs)
        elif model_type == 'online_gmm':
            raise ValueError("Not tested yet.")
            #self.model = OnlineGaussianMixture(n_components, n_jobs=n_jobs, store_cov=self.store_cov)


    def load_data(self, data_path):
        if hasattr(self, 'data_path'):
            raise ValueError("You can't retrain the model on another dataset.")
        else:
            self.data_path = data_path
            self.train_data = np.float32(torch.cat(torch.load(data_path)['features']).cpu().numpy())
            train_size = int(np.floor(len(self.train_data) * self.frac_train))
            self.test_data = self.train_data[train_size:,:]
            self.train_data = self.train_data[:train_size, :]

    def fit(self, patience):
        self.patience = patience
        self.save_args()
        self.model.fit(self.train_data, self.test_data,
                       saving='linear', saving_iter=2,
                       file_name=str(self.out_dir / 'intermediate'),
                       patience=patience)
        if self.simplify_dpgmm:
            self.model = self.model.simplified_model(self.train_data)
        self.save_model()

    def save_args(self):
        # Save user parameters
        args = {'model_type': self.model_type,
                'n_components': self.n_components,
                'frac_train': self.frac_train,
                'simplify_model': self.simplify_dpgmm,
                'data_path': self.data_path,
                'out_dir': str(self.out_dir),
                'n_jobs': self.n_jobs,
                'patience': self.patience}
        out_args = self.out_dir / 'checkpoint_args.json'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_args, 'w') as outfile:
            json.dump(args, outfile, indent=4)

    def save_model(self):
        # Save model
        model_file = h5py.File(self.out_dir / 'checkpoint.h5', 'w')
        grp = model_file.create_group('model_fitted')
        self.model.write(grp)
        model_file.close()

    def load_pretrained(self):
        print("Attempt to load pretrained model")
        with open(self.out_dir / 'checkpoint_args.json') as fin:
            params = json.load(fin)

        print("Loading training set to run the initialization procedure ...")
        self.load_data(params['data_path'])
        if 'n_jobs' not in params:
            params['n_jobs'] = 20
        self.init_model(params['model_type'], params['n_components'], params['n_jobs'])

        if os.path.isfile(str(self.out_dir / 'checkpoint.h5')):
            # Load best model
            print("Found converged model.")
            model_path = str(self.out_dir / 'checkpoint.h5')
            file = h5py.File(model_path, 'r')
            grp = file['model_fitted']
        elif os.path.isfile(str(self.out_dir / 'intermediate.h5')):
            print("Found still converging model.")
            # Load last intermediate model
            model_path = str(self.out_dir / 'intermediate.h5')
            file = h5py.File(model_path, 'r')
            last_iter = sorted([key for key in file.keys() if key.startswith('iter')], key=lambda x: int(x.replace('iter', '')))[-1]
            print("Found model's %dth iteration" % int(last_iter.replace('iter', '')))
            grp = file[last_iter]
        else:
            raise ValueError("Couldn't find any pretrained model in %s" % self.out_dir)

        print("Initializing the model.")
        # not clear why we need points to init the model
        # need to have a look at : https://github.com/geomphon/CogSci-2019-Unsupervised-speech-and-human-perception/blob/master/script_extract_posteriors.py
        self.model.read_and_init(grp, self.train_data)
        print("Succesfully loaded pretrained model.")
        file.close()

    def predict(self, input_features):
        return self.model.predict_log_resp(input_features)


def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--model_type', type=str, required=True, choices=['gmm', 'dpgmm', 'online_gmm'],
                        help='Type of model to be trained. Either gmm for Gausiann Mixture Model or DPGMM for ' \
                             'Dirichlet Process Variational Gaussian Mixture.')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of clusters to be considered in GMM (typically set to 50), or higher boundary on number of clusters '
                             'for DPGMM (typically set to 1000).')
    parser.add_argument('--frac_train', type=float, default=0.8,
                        help='Number of points to keep in the training set. Default to 80% of the points.')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Folder where to save the model.')
    parser.add_argument('--feat', type=str, required=True, help='Path to the feature file (.pt).')
    parser.add_argument('--reload', action='store_true', help='If activated, will try to find a pretrained model in the'
                                                              'out_dir.')
    parser.add_argument('--store_cov', action='store_true', help='If activated, will store the covariance matrices on disk.'
                                                                 'It''ll greatly reduce the memory footprint.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to consider when training the model')
    args = parser.parse_args(argv)

    clustering_model = ClusteringModel(model_type=args.model_type,
                                       n_components=args.n_components,
                                       frac_train=args.frac_train,
                                       out_dir=args.out_dir,
                                       n_jobs=args.n_jobs)
    if not args.reload:
        clustering_model.load_data(args.feat)
    else:
        clustering_model.load_pretrained()
    print("Start fitting the model...")
    clustering_model.fit(patience=5)
    print("Done")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)