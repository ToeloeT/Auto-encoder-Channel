import numpy as np
import h5py


class DatasetGenerator:
    
    # filename: path to the dataset file
    # num_paths: number of paths desired.
    #  If the number of paths of a dataset example is higher, the last path will be ignored
    #  If the number of paths of a dataset example is lower, the example will be zero padded
    # get_metadata: if True, returns metatda on the channel sample
    def __init__(self, filename, num_paths, get_metadata=False):
        self.filename = filename
        self.num_paths = num_paths
        self.get_metadata = get_metadata
    
    # Return an complex-valued array with shape [2, number of spatial samples, self.num_paths].
    # The first array along the first dimension correspond to the channel coefficients
    # The second array along the first dimension correspond to the delays
    def __call__(self):
        with h5py.File(self.filename, 'r') as hf:
            for key in hf:
                item = hf[key]
                # Metadata
                pos_x, pos_y = item.attrs['pos']
                channel_model = item.attrs['ch_mod']
                
                # Extract data
                # expected with shape [num spatial samples, num paths]
                paths_coeff = item['paths_coeff'][:]
                paths_delay = item['paths_delay'][:]
                
                s = paths_coeff.shape[0]
                p = paths_coeff.shape[1]
                
                paths_coeff = paths_coeff[:,:self.num_paths]
                paths_delay = paths_delay[:,:self.num_paths]
                if p < self.num_paths:
                    paths_coeff = np.concatenate([paths_coeff, np.zeros([s, self.num_paths-p], np.complex128)], axis=1)
                    paths_delay = np.concatenate([paths_delay, np.zeros([s, self.num_paths-p], np.float64)], axis=1)
                
                data = np.stack([paths_coeff, paths_delay], axis=0)
                
                if self.get_metadata:
                    yield data, channel_model, (pos_x, pos_y)
                else:
                    yield data