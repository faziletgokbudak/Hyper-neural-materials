import numpy as np

import fastmerl

epfl = np.load('epfl_median.npy')
merl = fastmerl.Merl('merl_median.binary')

any_fullbin = 'brdf.fullbin'
any_merl = fastmerl.Merl(any_fullbin)

print('he')


