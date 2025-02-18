# MegaSaM

[Project Page]() | [Paper]()

This code accompanies the paper

**MegaSam: Accurate, Fast and Robust Casual Structure and Motion from Casual
Dynamic Videos** \
Zhengqi Li, Richard Tucker, Forrester Cole, Qianqian Wang, Linyi Jin, Vickie Ye,
Angjoo Kanazawa, Aleksander Holynski, Noah Snavely

*This is not an officially supported Google product.*

## Clone

Make sure to clone the repository with the submodules by using:
`git clone --recursive git@github.com:mega-sam/mega-sam.git`

## Instructions for installing dependencies

### Python Environment

The following codebase was successfully run with Python 3.10, CUDA11.8, and
Pytorch2.0.1. We suggest installing the library in a virtual environment such as
Anaconda.

1.  To install main libraries, run: \
    `conda env create -f environment.yml`

2.  To install xformers for UniDepth model, follow the instructions from
    https://github.com/facebookresearch/xformers. If you encounter any
    installation issue, we suggest installing it from a prebuilt file. For
    example, for Python 3.10+Cuda11.8+Pytorch2.0.1, run: \
    `wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2`

    `conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2`

3.  Compile the extensions for the camera tracking module: \
    `python setup.py install`

### Downloading pretrained checkpoints

1.  Include a pretrained checkpoint in
    mega_sam_release/checkpoints/megasam_final.pth

2.  Include DepthAnything checkpoint in
    mega_sam_release/Depth-Anything/checkpoints/depth_anything_vitl14.pth

3.  clone torchhub folder from https://github.com/LiheYoung/Depth-Anything/tree/main/torchhub
    to mega_sam_release/Depth-Anything/torchhub

4. clone unidepth folder from https://github.com/lpiccinelli-eth/UniDepth/tree/main/unidepth
    to mega_sam_release/UniDepth/unidepth

5.  Include UniDepth checkpoint in
    mega_sam_release/UniDepth/unidepth_v2.pth

6.  Include RAFT checkpoint at mega_sam_release/cvd_opt/raft-things.pth

### Running MegaSaM on Sintel

1.  Download Sintel data

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_sintel.sh`

3.  Run camera tracking (Please modify DATA_PATH in the script. Adding
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_sintel.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script): `./cvd_opt/cvd_opt_sintel.sh`

5.  Evaluate camera poses and depths:
    `python ./evaluations_poses/evaluate_sintel.py`

    `python ./evaluations_depth/evaluate_depth_ours_sintel.py`

### Running MegaSaM on DyCheck

1.  Download Dycheck data

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_dycheck.sh`

3.  Running camera tracking (Please modify DATA_PATH in the script. Add
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_dycheck.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script):
    `./cvd_opt/cvd_opt_dycheck.sh`

5.  Evaluate camera poses and depths:
    `python ./evaluations_poses/evaluate_dycheck.py`

    `python ./evaluations_depth/evaluate_depth_ours_dycheck.py`

### Running MegaSaM on in-the-wild video, for example from DAVIS videos

1.  Download DAVIS data

2.  Precompute mono-depth (Please modify img-path in the script):
    `./mono_depth_scripts/run_mono-depth_demo.sh`

3.  Running camera tracking (Please modify DATA_PATH in the script. Add
    argument --opt_focal to enable focal length optimization):
    `./tools/evaluate_demo.sh`

4.  Running consistent video depth optimization given estimated cameras (Please
    modify datapath in the script):
    `./cvd_opt/cvd_opt_demo.sh`

### Contact

For any questions related to our paper, please send email to zl548@cornell.com.


### Bibtex

```
@inproceedings{li2024_megasam,
  title     = {MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos},
  author    = {Li, Zhengqi and Tucker, Richard and Cole, Forrester and Wang, Qianqian and Jin, Linyi and Ye, Vickie and Kanazawa, Angjoo and Holynski, Aleksander and Snavely, Noah},
  booktitle = {arxiv},
  year      = {2024}
}
```

### Copyright

Copyright 2025 Google LLC  

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.

