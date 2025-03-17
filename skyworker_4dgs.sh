#!/bin/bash
# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# List of folders
# evalset=(
#   movi_a_0001
# )

for i in {1..9}
do
  evalset+=("movi_a_000$i")
done

# Main directory
DATA_DIR=/home/skyworker/data/sets/movie_a/train/
CKPT_PATH=checkpoints/megasam_final.pth

for seq in ${evalset[@]}; do
  # Organize the data
  mkdir -p $DATA_DIR/$seq/images/seq1
  mkdir -p $DATA_DIR/$seq/ano
  mv $DATA_DIR/$seq/*.jpg $DATA_DIR/$seq/images/seq1
  mv $DATA_DIR/$seq/*.png $DATA_DIR/$seq/ano
  rm -f $DATA_DIR/$seq/*.npz

  # Extract masks
  python ano_mask.py --dir $DATA_DIR/$seq
done

export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

for seq in ${evalset[@]}; do

  # Skip if the output already exists
  if [ -f "outputs/${seq}_droid.npz" ]; then
    cp outputs/${seq}_droid.npz $DATA_DIR/$seq/${seq}.npz
    cp outputs_cvd/${seq}_sgd_cvd_hr.npz $DATA_DIR/$seq/${seq}_cvd.npz
    echo "Skipping $seq"
    continue
  fi

  # Run DepthAnything
  CUDA_VISIBLE_DEVICES=0 python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path $DATA_DIR/$seq/images/seq1 \
  --outdir Depth-Anything/video_visualization/$seq

  # Run UniDepth
  CUDA_VISIBLE_DEVICES=0 python UniDepth/scripts/demo_mega-sam.py \
  --scene-name $seq \
  --img-path $DATA_DIR/$seq/images/seq1 \
  --outdir UniDepth/outputs

  # Run MegaSaM
  CUDA_VISIBLE_DEVICE=0 python camera_tracking_scripts/test_demo.py \
  --datapath=$DATA_DIR/$seq/images/seq1 \
  --weights=$CKPT_PATH \
  --scene_name $seq \
  --mono_depth_path $(pwd)/Depth-Anything/video_visualization \
  --metric_depth_path $(pwd)/UniDepth/outputs \
  --disable_vis $@

  cp outputs/${seq}_droid.npz $DATA_DIR/$seq/${seq}.npz

  # Run Raft Optical Flows
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/preprocess_flow.py \
  --datapath=$DATA_DIR/$seq/images/seq1 \
  --model=cvd_opt/raft-things.pth \
  --scene_name $seq --mixed_precision

  # Run CVD optmization
  CUDA_VISIBLE_DEVICES=0 python cvd_opt/cvd_opt.py \
  --scene_name $seq \
  --w_grad 2.0 --w_normal 5.0

  cp outputs_cvd/${seq}_sgd_cvd_hr.npz $DATA_DIR/$seq/${seq}_cvd.npz

done






