import argparse
import os
import subprocess


if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='SIFT feature extraction script')
    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to the dataset root'
    )
    parser.add_argument(
        '--image_path', type=str, default=None,
        help='path to the images (defaults to dataset path)'
    )
    parser.add_argument(
        '--colmap_path', type=str, required=True,
        help='path to the COLMAP executable'
    )
    args = parser.parse_args()
    if args.image_path is None:
        args.image_path = args.dataset_path

    database_path = os.path.join(args.dataset_path, 'sift-features.db')
    subprocess.call([
        args.colmap_path, 'feature_extractor',
        '--database_path', database_path,
        '--image_path', args.image_path,
        '--descriptor_normalization', 'l2',
        '--SiftExtraction.first_octave', '0',
        '--SiftExtraction.num_threads', '1'
    ])
