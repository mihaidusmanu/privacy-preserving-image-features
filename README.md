# Privacy-Preserving Image Features
# (via Adversarial Affine Subspace Embedding)

🚧 This repository is still work in progress. It contains a clean-up reimplementation of the following paper:

```text
"Privacy-Preserving Image Features via Adversarial Affine Subspace Embeddings".
M. Dusmanu, J.L. Schönberger, S.N. Sinha, and M. Pollefeys. CVPR 2021.
```
[[Paper on arXiv]](https://arxiv.org/abs/2006.06634)

## Requirements

OpenMP and Eigen must be installed prior to the setup of this repository. CUDA is also a recommended dependency for best performance.

1. Start by creating and activating the base python environment:
```
conda env create --file=env.yml; conda activate ppif
```
2. Install the `pyppif` bindings:
```
cd py_ppif; pip install .; cd ..
```
3. [Optional] Install the `pyppifcuda` bindings:
```
cd py_ppif_cuda; pip install .; cd ..
```

## HPatches Sequences evaluation

Start by installing COLMAP (used for SIFT feature extraction) and set the env variable COLMAP_PATH to the COLMAP executable, e.g.:
```
export COLMAP_PATH=~/sources/colmap/build/src/exe/colmap
```

1. Download the dataset:
```
bash download_hpatches_sequences.sh
```
2. Extract SIFT / DoG + HardNet features:
```
python feature-utils/extract_sift.py --dataset_path data/hpatches-sequences-release/ --colmap_path $COLMAP_PATH
python feature-utils/extract_hardnet.py --dataset_path data/hpatches-sequences-release/
```
3. Run the evaluation:
```
python evaluate_hpatches_sequences.py --dataset_path data/hpatches-sequences-release/ --descriptor sift
python evaluate_hpatches_sequences.py --dataset_path data/hpatches-sequences-release/ --descriptor hardnet
```


## BibTeX

If you use this code in your project, please cite the following paper:
```
@InProceedings{Dusmanu2021Privacy,
    author = "Dusmanu, Mihai and Sch\"onberger, Johannes L. and Sinha, Sudipta N. and Pollefeys, Marc",
    title = "{P}rivacy-{P}reserving {I}mage {F}eatures via {A}dversarial {A}ffine {S}ubspace {E}mbeddings",
    booktitle = "Proceedings of the 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
    year = "2021"
}
```