import argparse
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from ppif_utils import (
    LiftingAlgorithms,
    select_lifting_function,
    lifting_config_to_str,
    subspace_to_subspace_exhaustive_matcher
)


def retrieve_features_from_database(image_name, db_cursor):
    # Retrieve image id.
    image_id = db_cursor.execute('SELECT image_id FROM images WHERE name=?', (image_name,)).fetchone()[0]
    # Retrieve keypoints.
    blob, rows, cols = db_cursor.execute("SELECT data, rows, cols FROM keypoints WHERE image_id=?", (image_id,)).fetchone()
    keypoints = np.frombuffer(blob, dtype=np.float32).reshape((rows, cols))
    # Retrieve descriptors.
    blob, rows, cols = db_cursor.execute("SELECT data, rows, cols FROM descriptors WHERE image_id=?", (image_id,)).fetchone()
    try:
        descriptors = np.frombuffer(blob, dtype=np.uint8).reshape((rows, cols)).astype(np.float32)
        descriptors /= (np.linalg.norm(descriptors, axis=1)[:, np.newaxis] + 1e-8)
    except ValueError:
        descriptors = np.frombuffer(blob, dtype=np.float32).reshape((rows, cols))
    return image_id, keypoints, descriptors


def mnn_matcher(sqdistance_matrix):
    nn12 = np.argmin(sqdistance_matrix, axis=1)
    nn21 = np.argmin(sqdistance_matrix, axis=0)
    ids1 = np.arange(0, sqdistance_matrix.shape[0])
    mask = (ids1 == nn21[nn12])
    matches = np.concatenate([ids1[mask, np.newaxis], nn12[mask, np.newaxis]], axis=1)
    return matches


def benchmark_features(dataset_path, descriptor, lifting_config, rng=np.arange(1, 6)):
    private = lifting_config is not None
    if private:
        lifting_function = select_lifting_function(lifting_config, descriptor)

    db_path = dataset_path / f'{descriptor}-features.db'
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    db = sqlite3.connect(db_path)
    db_cursor = db.cursor()

    n_matches = []
    seq_type = []
    i_err = {thr: [] for thr in rng}
    v_err = {thr: [] for thr in rng}

    seq_paths = sorted([seq_path for seq_path in dataset_path.glob('*/') if seq_path.name[: 2] in ['i_', 'v_']])
    for seq_path in tqdm.tqdm(seq_paths):
        seq_name = seq_path.name

        # Reference image.
        image_name1 = f'{seq_name}/1.ppm'
        image_id1, keypoints1, descriptors1 = retrieve_features_from_database(image_name1, db_cursor)
        # raw_descriptors1 = descriptors1
        if private:
            descriptors1 = lifting_function(descriptors1, seed=image_id1)

        # Query image.
        for image_idx2 in range(2, 7):
            image_name2 = f'{seq_name}/{image_idx2}.ppm'
            image_id2, keypoints2, descriptors2 = retrieve_features_from_database(image_name2, db_cursor)
            # raw_descriptors2 = descriptors2
            if private:
                descriptors2 = lifting_function(descriptors2, seed=image_id2)

            # Feature matching.
            if private:
                sqdistance_matrix = subspace_to_subspace_exhaustive_matcher(descriptors1, descriptors2, lifting_config['dim'])
                assert ~np.any(np.isnan(sqdistance_matrix))
            else:
                sqdistance_matrix = 2 - 2 * np.clip(descriptors1 @ descriptors2.T, -1, 1)
            # if private:
            #     raw_sqdistance_matrix = 2 - 2 * np.clip(raw_descriptors1 @ raw_descriptors2.T, -1, 1)
            #     if np.max(sqdistance_matrix - raw_sqdistance_matrix) > 1e-5:
            #         print(np.max(sqdistance_matrix - raw_sqdistance_matrix))
            #         print(np.sum(sqdistance_matrix - raw_sqdistance_matrix > 0))
            matches = mnn_matcher(sqdistance_matrix)

            # Load homography from disk.    
            homography = np.loadtxt(os.path.join(dataset_path, seq_name, "H_1_" + str(image_idx2)))
            
            # Inverse mapping such that pixel errors are compatible during aggregation.
            # See also Slide 52 @ https://drive.google.com/file/d/1uJjRtQWbGUyXz9LWEnVn9bF0VZaioJ1u
            # Warp points from image 2 to image 1.
            # pos2 = keypoints2[matches[:, 1], : 2]
            # pos2_h = np.concatenate([pos2, np.ones([matches.shape[0], 1])], axis=1)
            # pos1_proj_h = np.transpose(np.dot(np.linalg.inv(homography), np.transpose(pos2_h)))
            # pos1_proj = pos1_proj_h[:, : 2] / pos1_proj_h[:, 2 :]
            # pos1 = keypoints1[matches[:, 0], : 2]
            # dist = np.sqrt(np.sum((pos1 - pos1_proj) ** 2, axis=1))
            
            # Legacy code from D2-Net repository.
            # Warp points from image 1 to image 2.
            pos1 = keypoints1[matches[:, 0], : 2] 
            pos1_h = np.concatenate([pos1, np.ones([matches.shape[0], 1])], axis=1)
            pos2_proj_h = np.transpose(np.dot(homography, np.transpose(pos1_h)))
            pos2_proj = pos2_proj_h[:, : 2] / pos2_proj_h[:, 2 :]
            pos2 = keypoints2[matches[:, 1], : 2]
            dist = np.sqrt(np.sum((pos2 - pos2_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])
            
            if dist.shape[0] == 0:
                dist = np.array([float("inf")])
            
            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr].append(np.mean(dist <= thr))
                else:
                    v_err[thr].append(np.mean(dist <= thr))
    
    db_cursor.close()
    db.close()

    seq_type = np.array(seq_type)
    n_matches = np.array(n_matches)
    
    return i_err, v_err, [seq_type, n_matches]


def summary(stats):
    seq_type, n_matches = stats
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / seq_type.shape[0], 
        np.sum(n_matches[seq_type == 'i']) / np.sum(seq_type == 'i'), 
        np.sum(n_matches[seq_type == 'v']) / np.sum(seq_type == 'v')
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HPatches Sequences evaluation script')
    parser.add_argument(
        '--dataset_path', type=Path, required=True,
        help='path to the dataset root'
    )
    parser.add_argument(
        '--descriptor', type=str, choices=['sift', 'hardnet'], required=True,
        help='list of descriptors to evaluate'
    )
    parser.add_argument(
        '--lifting_dim', type=int, default=4,
        help='lifting dimension'
    )
    parser.add_argument(
        '--num_sub_databases', type=int, default=16,
        help='number of sub-databases to use'
    )
    args = parser.parse_args()
    assert args.descriptor in ['sift', 'hardnet']

    # List of methods and plotting styles.
    plt_title = 'PPIF - %s, %d, %d' % (args.descriptor, args.lifting_dim, args.num_sub_databases)
    linestyles = {
        'sift': '-',
        'hardnet': '--'
    }
    lifting_configs = [
        None,
        {
            'alg': LiftingAlgorithms.RAND,
            'dim': args.lifting_dim
        },
        {
            'alg': LiftingAlgorithms.ADV,
            'dim': args.lifting_dim
        },
        {
            'alg': LiftingAlgorithms.SUBADV,
            'dim': args.lifting_dim,
            'num_sub_databases': args.num_sub_databases
        },
        {
            'alg': LiftingAlgorithms.SUBHYB,
            'dim': args.lifting_dim,
            'num_sub_databases': args.num_sub_databases
        }
    ]
    colors = {
        lifting_config_to_str(lifting_configs[0]): 'black',
        lifting_config_to_str(lifting_configs[1]): 'purple',
        lifting_config_to_str(lifting_configs[2]): 'red',
        lifting_config_to_str(lifting_configs[3]): 'blue',
        lifting_config_to_str(lifting_configs[4]): 'green'
    }

    # Benchmarking.
    errors = {}
    assert args.descriptor in linestyles
    for lifting_config in lifting_configs:
        assert lifting_config_to_str(lifting_config) in colors
        method = (args.descriptor, lifting_config)
        key = (args.descriptor, lifting_config_to_str(lifting_config))
        print(method)
        errors[key] = benchmark_features(args.dataset_path, args.descriptor, lifting_config)
        summary(errors[key][-1])

    # Plotting settings.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times'] + plt.rcParams['font.serif']
    plt.rcParams['font.style'] = 'normal'

    # Plot.
    plt_lim = [1, 5]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

    plt.rc('axes', titlesize=12.5)
    plt.rc('axes', labelsize=12.5)

    fig = plt.figure(figsize=(2, 3))
    axes = [fig.subplots(nrows=1, ncols=1)]

    for method in errors:
        descriptor, lifting_config_str = method
        ls = linestyles[descriptor]
        color = colors[lifting_config_str]
        i_err, v_err, _ = errors[method]
        name = descriptor + ' ' + lifting_config_str
        axes[0].plot(
            plt_rng, [np.mean(i_err[thr] + v_err[thr]) for thr in plt_rng],
            color=color, ls=ls, linewidth=3, label=name)
    axes[0].set_title(plt_title)
    axes[0].set_xlim(plt_lim)
    axes[0].set_xticks(plt_rng)
    axes[0].set_ylabel('Matching Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].grid()
    axes[0].tick_params(axis='both', which='major', labelsize=12.5)

    lines, labels = axes[0].get_legend_handles_labels()

    fig.legend(lines, labels, fontsize='x-small')

    plt.savefig(
        f'hseq_{args.descriptor}_{args.lifting_dim}_{args.num_sub_databases}.png',
        pad_inches=0, bbox_inches='tight', dpi=300)
