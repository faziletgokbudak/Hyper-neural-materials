import glob
import os
import torch
import numpy as np
import pandas as pd
import os.path as op

from common import get_mgrid

from torch.utils.data import Dataset, TensorDataset, DataLoader

device = 'cpu'  # 'cuda' or 'cpu'
Xvars = ['hx', 'hy', 'hz', 'dx', 'dy', 'dz']
Yvars = ['brdf_r', 'brdf_g', 'brdf_b']

import coords
import common
import fastmerl


def brdf_to_rgb(rvectors, brdf):
    hx = np.reshape(rvectors[:, 0], (-1, 1))
    hy = np.reshape(rvectors[:, 1], (-1, 1))
    hz = np.reshape(rvectors[:, 2], (-1, 1))
    dx = np.reshape(rvectors[:, 3], (-1, 1))
    dy = np.reshape(rvectors[:, 4], (-1, 1))
    dz = np.reshape(rvectors[:, 5], (-1, 1))

    theta_h = np.arctan2(np.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = np.arctan2(np.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = np.arctan2(dy, dx)
    wiz = np.cos(theta_d) * np.cos(theta_h) - \
          np.sin(theta_d) * np.cos(phi_d) * np.sin(theta_h)
    rgb = brdf * np.clip(wiz, 0, 1)
    return rgb


class EPFL(Dataset):
    def __init__(self, merlPath):
        super(EPFL, self).__init__()
        self.fnames = glob.glob(op.join(merlPath, "*.csv"))
        self.train_coords, self.train_brdfvals = [], []

        self.brdfs = []

        for fname in self.fnames:
            df = pd.read_csv(fname, sep=" ")
            tensor = torch.tensor(df.values, dtype=torch.float32)
            amps = tensor[:, 9:]
            amps = amps.detach().numpy()

            any_fullbin = 'brdf.fullbin'
            merl = fastmerl.Merl(any_fullbin)
            merl.from_array(amps)
            merl.brdf_np = np.array(merl.brdf)
            merl.sampling_phi_d = 180
            reflectance_train = generate_nn_datasets(merl, nsamples=180 * 90 * 90, dataset='EPFL', pct=1)

            self.train_coords.append(reflectance_train[Xvars].values)
            self.train_brdfvals.append(reflectance_train[Yvars].values)
            print(fname)

        self.train_coords = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in self.train_coords]
        self.train_brdfvals = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in self.train_brdfvals]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.train_coords[idx], 'amps': self.train_brdfvals[idx]}
        gt_dict = {'amps': self.train_brdfvals[idx]}
        return in_dict, gt_dict


class MerlDataset(Dataset):
    def __init__(self, merlPath):
        super(MerlDataset, self).__init__()
        self.fnames = glob.glob(op.join(merlPath, "*.binary"))  # [:10]
        self.train_coords, self.train_brdfvals = [], []

        self.brdfs = []

        for fname in self.fnames:
            BRDF = fastmerl.Merl(fname)
            reflectance_train = generate_nn_datasets(BRDF, nsamples=180*90*90, dataset='MERL', pct=1)
            self.train_coords.append(reflectance_train[Xvars].values)
            self.train_brdfvals.append(reflectance_train[Yvars].values)
            print(fname)
        self.train_coords = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in self.train_coords]
        self.train_brdfvals = [torch.tensor(arr, dtype=torch.float32, device=device) for arr in self.train_brdfvals]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        in_dict = {'idx': idx, 'coords': self.train_coords[idx], 'amps': self.train_brdfvals[idx]}
        gt_dict = {'amps': self.train_brdfvals[idx]}
        return in_dict, gt_dict


def generate_nn_datasets(brdf, nsamples, dataset, pct=1):

    rangles = np.random.uniform([0, 0, 0], [np.pi / 2., np.pi / 2., 2 * np.pi], [int(nsamples * pct), 3]).T
    rangles[2] = common.normalize_phid(rangles[2])
    rvectors = coords.rangles_to_rvectors(*rangles)

    if dataset == 'MERL':
        median = fastmerl.Merl('data/merl_median.binary')
        median_vals = brdf_values(rvectors, brdf=median)

    elif dataset == 'EPFL':
        median_np = np.load('epfl_median.npy')
        any_fullbin = 'brdf.fullbin'
        median = fastmerl.Merl(any_fullbin)
        median.from_array(median_np)
        median.brdf_np = np.array(median.brdf)
        median.sampling_phi_d = 180
        median_vals = brdf_values(rvectors, brdf=median)

    brdf_vals = brdf_values(rvectors, brdf=brdf)
    normalized_brdf_vals = np.log(1 + ((brdf_vals + 0.002) / (median_vals + 0.002)))

    df = pd.DataFrame(np.concatenate([rvectors.T, normalized_brdf_vals], axis=1), columns=[*Xvars, *Yvars])
    df = df[(df.T != 0).any()]
    df = df.drop(df[df['brdf_r'] < 0].index)
    return df


def brdf_values(rvectors, brdf=None, model=None):
    if brdf is not None:
        rangles = coords.rvectors_to_rangles(*rvectors)
        brdf_arr = brdf.eval_interp(*rangles).T
    elif model is not None:
        # brdf_arr = model.predict(rvectors.T)        # nnModule has no .predict
        raise RuntimeError("Should not have entered that branch at all from the original code")
    else:
        raise NotImplementedError("Something went really wrong.")
    brdf_arr *= common.mask_from_array(rvectors.T).reshape(-1, 1)
    return brdf_arr

