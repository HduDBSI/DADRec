#!/usr/bin/env bash
set -euo pipefail

mkdir -p results logs/eval logs/tb

PYTHON_BIN="${PYTHON_BIN:-/home/lcg/miniconda3/envs/DiffGRM/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=python
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" PYTHONPATH="$PWD" "$PYTHON_BIN" tools/eval_checkpoint_once.py \
  --model DADRec \
  --dataset AmazonReviews2014 \
  --checkpoint ckpt/beauty.pth \
  --result_json results/beauty_test.json \
  --result_tsv results/beauty_test.tsv \
  --run_name beauty_preopq_softquant_test \
  --cache_dir=cache \
  --log_dir=logs/eval \
  --ckpt_dir=ckpt \
  --tensorboard_log_dir=logs/tb \
  --category=Beauty \
  --eval_batch_size=32 \
  --n_head=4 \
  --n_embd=256 \
  --n_inner=1024 \
  --n_digit=4 \
  --masking_strategy=guided \
  --guided_refresh_each_step=false \
  --guided_select=least \
  --guided_conf_metric=msp \
  --encoder_n_layer=1 \
  --decoder_n_layer=4 \
  --sent_emb_model=sentence-transformers/sentence-t5-base \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --normalize_after_pca=true \
  --force_regenerate_opq=false \
  --share_decoder_output_embedding=true \
  --cadd.enabled=true \
  --cadd.aux_target=opq_subvector \
  --cadd.hint_injection=soft_quantized_sid \
  --cadd.hint_scale=0.30 \
  --cadd.code_prob_temperature=0.50 \
  --drift_moe.enabled=true \
  --drift_moe.n_experts=4 \
  --drift_moe.bottleneck=32 \
  --drift_moe.recency_gamma=0.90 \
  --drift_moe.hint_code_temperature=1.0 \
  --drift_moe.bucket_temperature=0.03 \
  --drift_moe.bucket_strategy=train_quantile_full \
  --drift_moe.novel_topk=8 \
  --drift_moe.evidence_temperature=1.0
