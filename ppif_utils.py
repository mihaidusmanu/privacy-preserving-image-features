from enum import Enum

import numpy as np

import pyppif


class LiftingAlgorithms(Enum):
    RAND = 0    # Random lifting.
    ADV = 1     # Adversarial lifting.
    HYB = 2     # Hybrid lifting (half random, half adversarial).
    SUBADV = 3  # Adversarial lifting with sub-databases.
    SUBHYB = 4  # Hybrid lifting with sub-databases.


def lifting_config_to_str(lifting_config):
    if lifting_config is None:
        return f'NONE'
    lifting_alg = lifting_config['alg']
    lifting_dim = lifting_config['dim']
    if lifting_alg == LiftingAlgorithms.RAND:
        return f'RAND. - dim. {lifting_dim}'
    elif lifting_alg == LiftingAlgorithms.ADV:
        return f'ADV. - dim. {lifting_dim}'
    elif lifting_alg == LiftingAlgorithms.HYB:
        return f'HYB. - dim. {lifting_dim}'
    elif lifting_alg == LiftingAlgorithms.SUBADV:
        num_sub_databases = lifting_config['num_sub_databases']
        return f'SUB-ADV. - dim. {lifting_dim} ({num_sub_databases})'
    elif lifting_alg == LiftingAlgorithms.SUBHYB:
        num_sub_databases = lifting_config['num_sub_databases']
        return f'SUB-HYB. - dim. {lifting_dim} ({num_sub_databases})'
    else:
        raise NotImplementedError(lifting_alg)


def load_lifting_database(descriptor):
    # The databases are stored as float64, but the bindings take float32. Without
    # this conversion, pybind11 makes a temporary float32 copy of the whole database
    # (~130 MB) on every lifting call, i.e. once per image.
    return np.ascontiguousarray(np.load(f'databases/{descriptor}.npy'), dtype=np.float32)


def select_lifting_function(lifting_config, descriptor):
    lifting_alg = lifting_config['alg']
    lifting_dim = lifting_config['dim']
    if lifting_alg == LiftingAlgorithms.RAND:
        def lifting_function(descriptor, seed=0):
            return pyppif.random_lifting(descriptor, lifting_dim, seed=seed)
    elif lifting_alg == LiftingAlgorithms.ADV:
        database = load_lifting_database(descriptor)
        def lifting_function(descriptor, seed=0):
            return pyppif.adversarial_lifting(descriptor, lifting_dim, database, num_sub_databases=1, seed=seed)
    elif lifting_alg == LiftingAlgorithms.HYB:
        database = load_lifting_database(descriptor)
        def lifting_function(descriptor, seed=0):
            return pyppif.hybrid_lifting(descriptor, lifting_dim, database, num_sub_databases=1, seed=seed)
    elif lifting_alg == LiftingAlgorithms.SUBADV:
        database = load_lifting_database(descriptor)
        num_sub_databases = lifting_config['num_sub_databases']
        def lifting_function(descriptor, seed=0):
            return pyppif.adversarial_lifting(descriptor, lifting_dim, database, num_sub_databases=num_sub_databases, seed=seed)
    elif lifting_alg == LiftingAlgorithms.SUBHYB:
        database = load_lifting_database(descriptor)
        num_sub_databases = lifting_config['num_sub_databases']
        def lifting_function(descriptor, seed=0):
            return pyppif.hybrid_lifting(descriptor, lifting_dim, database, num_sub_databases=num_sub_databases, seed=seed)
    else:
        raise NotImplementedError(lifting_alg)
    return lifting_function


def subspace_to_subspace_exhaustive_matcher(descriptors1, descriptors2, subspace_dim):
    if subspace_dim == 2 or subspace_dim == 4:
        try:
            import pyppifcuda
            return pyppifcuda.subspace_to_subspace_exhaustive_matcher(descriptors1, descriptors2)
        except ModuleNotFoundError:
            return pyppif.subspace_to_subspace_exhaustive_matcher(descriptors1, descriptors2)
    return pyppif.subspace_to_subspace_exhaustive_matcher(descriptors1, descriptors2)
