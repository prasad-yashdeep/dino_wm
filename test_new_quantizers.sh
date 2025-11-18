#!/bin/bash

# Test script for new quantizers with relu activation
# Tests EMA Vector Quantizer and Gumbel Quantizer

set -e  # Exit on error

# eval "$(conda shell.bash hook)"
# conda activate /scratch/yp2693/world_models/penv

python train.py \
    --config-name train.yaml \
    env=wall \
    env.dataset.n_rollout=200 \
    outputs_dir=small_outputs \
    wandb_project=dino_wm_small \
    img_size=224 \
    frameskip=5 \
    concat_dim=1 \
    normalize_action=True \
    action_emb_dim=10 \
    num_action_repeat=1 \
    num_hist=1 \
    num_pred=1 \
    has_predictor=True \
    has_decoder=True \
    training.seed=0 \
    training.epochs=100 \
    training.batch_size=32 \
    training.save_every_x_epoch=3 \
    training.reconstruct_every_x_batch=500 \
    training.num_reconstruct_samples=6 \
    training.encoder_lr=1e-4 \
    training.decoder_lr=3e-4 \
    training.predictor_lr=5e-4 \
    training.action_encoder_lr=5e-4 \
    training.action_quantizer_lr=1e-4 \
    training.state_quantizer_lr=1e-4 \
    training.max_grad_norm=1.0 \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=5 \
    training.scheduler.warmup_start_lr_factor=0.01 \
    training.scheduler.min_lr_factor=0.0 \
    training.scheduler.step_size=10 \
    training.scheduler.gamma=0.1 \
    training.scheduler.stage1_end=30 \
    training.scheduler.stage2_end=50 \
    training.scheduler.stage3_encoder_lr=null \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=1.0 \
    model.quantization_loss_weight=1.0 \
    quantize=True \
    state_vocabulary_size=128 \
    action_vocabulary_size=128 \
    encoder=conv2d \
    encoder.input_channels=3 \
    encoder.emb_dim=64 \
    encoder.image_size=224 \
    encoder.depth=4 \
    encoder.hidden_dim=256 \
    action_encoder=proprio \
    decoder=vqvae \
    decoder.channel=384 \
    decoder.n_res_block=4 \
    decoder.n_res_channel=128 \
    predictor=conv3d \
    predictor.hidden_dim=128 \
    predictor.depth=3 \
    predictor.dropout=0.1 \
    state_quantizer=ema_vector_quantizer \
    state_quantizer.n_embed=128 \
    state_quantizer.commitment_cost=0.35 \
    state_quantizer.kld_scale=10.0 \
    state_quantizer.decay=0.99 \
    state_quantizer.epsilon=1e-5 \
    state_quantizer.reset_unused_codes=true \
    state_quantizer.reset_threshold=1.0 \
    state_quantizer.reset_interval=100 \
    action_quantizer=ema_vector_quantizer \
    action_quantizer.n_embed=128 \
    action_quantizer.commitment_cost=0.25 \
    action_quantizer.kld_scale=10.0 \
    action_quantizer.decay=0.99 \
    action_quantizer.epsilon=1e-5 \
    action_quantizer.reset_unused_codes=true \
    action_quantizer.reset_threshold=1.0 \
    action_quantizer.reset_interval=100 \
    debug=False \
    disable_wandb=False \
    force_restart=False \
    resume_from=null \

