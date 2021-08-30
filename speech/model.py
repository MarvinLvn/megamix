import argparse
from megamix.batch import DPVariationalGaussianMixture
from megamix.batch import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn import mixture
from megamix.online import GaussianMixture as OnlineGaussianMixture
import numpy as np
import torch
import sys
import json
import h5py
from pathlib import Path
import os
import pickle
from speech.pca import PCA
np.random.seed(2021)

class ClusteringModel:
    def __init__(self, model_type, n_components, out_dir, n_jobs, window=1000, update_online=False, kappa=0.5,
                 simplify_model=True, cov_type='full', pca_path=None):
        assert model_type in ['gmm', 'dpgmm', 'online_gmm', 'kmeans', 'skgmm']
        self.model_type = model_type
        self.n_components = n_components
        self.simplify_model = simplify_model
        self.out_dir = Path(out_dir)
        self.n_jobs = n_jobs
        self.window = window
        self.update_online = update_online
        self.kappa = kappa
        self.cov_type = cov_type
        self.pca_path = pca_path
        self.init_model(model_type, n_components, n_jobs)

    def init_model(self, model_type, n_components, n_jobs):
        self.code_type = 'megamix'
        if model_type == 'gmm':
            self.model = GaussianMixture(n_components, n_jobs=n_jobs)
        elif model_type == 'dpgmm':
            self.model = DPVariationalGaussianMixture(n_components, n_jobs=n_jobs)
        elif model_type == 'online_gmm':
            self.model = OnlineGaussianMixture(n_components, n_jobs=n_jobs, window=self.window, update=self.update_online,
                                               kappa=self.kappa)
        elif model_type == 'kmeans':
            self.model = MiniBatchKMeans(n_clusters=n_components, random_state=0, batch_size=500, max_iter=500)
            self.code_type = 'scikit_learn'
        elif model_type == 'skgmm':
            # Previously, it was 500, but it's way too much when applying full cov.
            self.model = mixture.GaussianMixture(n_components=n_components, covariance_type=self.cov_type, max_iter=300)
            self.code_type = 'scikit_learn'
        self.load_pca()

    def load_pca(self):
        if self.pca_path is not None:
            print("Loading PCA")
            # Fake initialization
            self.pca = PCA("", "", 1)
            # Load pretrained .pkl file
            self.pca.load_pretrained(self.pca_path)

    def load_data(self, data_path):
        if hasattr(self, 'data_path'):
            raise ValueError("You can't retrain the model on another dataset.")
        else:
            self.data_path = data_path
            self.train_data = np.float32(torch.cat(torch.load(data_path)['features']).cpu().numpy())
            if self.pca_path is not None:
                print("Projecting features with PCA")
                self.train_data = self.pca.project(self.train_data)

    def fit(self):
        self.save_args()
        if self.model_type == 'online_gmm' and not self.model._is_initialized:
            self.model.initialize(self.train_data, init_choice='plus')

        if self.code_type == 'megamix':
            self.model.fit(self.train_data, saving='linear', saving_iter=2,
                       file_name=str(self.out_dir / 'intermediate'))
            if self.simplify_model and self.model_type != 'online_gmm':
                self.model = self.model.simplified_model(self.train_data)
        elif self.code_type == 'scikit_learn':
            self.model.fit(self.train_data)
        self.save_model()

    def save_args(self):
        # Save user parameters
        args = {'model_type': self.model_type,
                'n_components': self.n_components,
                'simplify_model': self.simplify_model,
                'data_path': self.data_path,
                'out_dir': str(self.out_dir),
                'n_jobs': self.n_jobs,
                'window': self.window,
                'update_online': self.update_online,
                'kappa': self.kappa,
                'code_type': self.code_type,
                'cov_type': self.cov_type,
                'pca_path': self.pca_path}
        out_args = self.out_dir / 'checkpoint_args.json'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_args, 'w') as outfile:
            json.dump(args, outfile, indent=4)

    def save_model(self):
        if self.code_type == 'megamix':
            # Save model
            model_file = h5py.File(self.out_dir / 'checkpoint.h5', 'w')
            grp = model_file.create_group('model_fitted')
            self.model.write(grp)
            model_file.close()
        elif self.code_type == 'scikit_learn':
            filename = self.out_dir / 'checkpoint.pkl'
            pickle.dump(self.model, open(filename, 'wb'))

    def load_pretrained(self):
        print("Attempt to load pretrained model")
        # Load parameters
        with open(self.out_dir / 'checkpoint_args.json') as fin:
            params = json.load(fin)

        # dirty hack to ensure compatibility
        if 'code_type' not in params:
            params['code_type'] = 'megamix'

        # Load megamix model
        if params['code_type'] == 'megamix':
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
        # Load scikit_learn model
        elif params['code_type'] == 'scikit_learn':
            self.model = pickle.load(open(str(self.out_dir / 'checkpoint.pkl'), 'rb'))
        self.n_components = params['n_components']

        # Check if pretrained PCA needs to be loaded
        self.pca_path = params['pca_path']
        self.load_pca()

    def predict(self, input_features):
        if self.pca_path is not None:
            input_features = self.pca.project(input_features)
        if self.code_type == 'megamix':
            return self.model.predict_log_resp(input_features)
        elif self.code_type == 'scikit_learn' and self.model.__class__.__name__ == 'MiniBatchKMeans':
            return self.model.predict(input_features)
        elif self.code_type == 'scikit_learn' and self.model.__class__.__name__ == 'GaussianMixture':
            return self.model.predict_proba(input_features)
        else:
            raise ValueError("Can't identify the type of the model with %s or %s." % (self.code_type, self.model.__class__.__name__))



def main(argv):
    parser = argparse.ArgumentParser(
        description='Given a folder containing audio files, will extract MFCCs features stored under a .pt file')
    parser.add_argument('--model_type', type=str, required=True, choices=['gmm', 'dpgmm', 'online_gmm', 'kmeans', 'skgmm'],
                        help='Type of model to be trained. Either gmm for Gausiann Mixture Model or DPGMM for ' \
                             'Dirichlet Process Variational Gaussian Mixture.')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of clusters to be considered in GMM (typically set to 50), or higher boundary on number of clusters '
                             'for DPGMM (typically set to 1000).')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Folder where to save the model.')
    parser.add_argument('--feat', type=str, required=True, help='Path to the feature file (.pt).')
    parser.add_argument('--reload', action='store_true', help='If activated, will try to find a pretrained model in the'
                                                              'out_dir.')
    parser.add_argument('--store_cov', action='store_true', help='If activated, will store the covariance matrices on disk.'
                                                                 'It''ll greatly reduce the memory footprint.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to consider when training the model')
    parser.add_argument('--window', type=int, default=1000, help='Number of data point to consider for online gmm')
    parser.add_argument('--kappa', type=float, default=0.5, help='Kappa, or weight associated to new points (online gmm)')
    parser.add_argument('--update_online', action='store_true', help='If True, the matrices of Cholesky of covariance matrices are updated, '
                                                                     'else they are computed at each iteration. Set it to True if window < dimension'
                                                                     'of the problem.')
    parser.add_argument('--cov_type', type=str, default='full', choices=['full', 'diag', 'tied', 'spherical'],
                        help='Covariance type to be used in scikit learn GMM')
    parser.add_argument('--pca', type=str, required=False, help='Path to a PCA model folder  to be used before training (optional).')
    args = parser.parse_args(argv)

    if args.reload and args.model_type == 'online_gmm':
        raise ValueError("Can't reload a model of type %s." % args.model_type)

    clustering_model = ClusteringModel(model_type=args.model_type,
                                       n_components=args.n_components,
                                       out_dir=args.out_dir,
                                       n_jobs=args.n_jobs,
                                       window=args.window,
                                       update_online=args.update_online,
                                       kappa=args.kappa,
                                       cov_type=args.cov_type,
                                       pca_path=args.pca)
    if not args.reload:
        clustering_model.load_data(args.feat)
    else:
        clustering_model.load_pretrained()
    print("Start fitting the model...")
    clustering_model.fit()
    print("Done")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)