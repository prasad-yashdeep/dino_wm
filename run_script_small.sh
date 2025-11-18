#!/bin/bash
python train.py \
    --config-name train.yaml \
    env=wall \
    frameskip=5 \
    num_hist=1 \
    training.epochs=60 \
    training.encoder_lr=1e-4 \
    training.predictor_lr=3e-4 \
    training.decoder_lr=3e-4 \
    training.action_encoder_lr=1e-4 \
    training.action_quantizer_lr=2e-4 \
    training.state_quantizer_lr=2e-4 \
    training.scheduler.type=cosine_with_warmup \
    training.scheduler.warmup_epochs=1 \
    training.scheduler.warmup_start_lr_factor=0.1 \
    training.max_grad_norm=3.0 \
    quantize=True \
    state_vocabulary_size=128 \
    action_vocabulary_size=16 \
    model.train_encoder=True \
    model.train_predictor=True \
    model.train_decoder=True \
    model.vcreg_loss_weight=10.0 \
    model.quantization_loss_weight=0.5 \
    encoder=conv2d \
    predictor=conv3d \
    decoder=vqvae \
    state_quantizer=ema_vector_quantizer \
    action_quantizer=ema_vector_quantizer \
    disable_wandb=false \
    force_restart=true


# Test 2: Gumbel Quantizer (optional - commented out by default)
# Uncomment to test Gumbel quantizer
# echo "=========================================="
# echo "Test 2: Gumbel Quantizer"
# echo "=========================================="
#
# python train.py \
#     --config-name train.yaml \
#     env=wall \
#     env.dataset.n_rollout=200 \
#     frameskip=5 \
#     num_hist=1 \
#     training.epochs=10 \
#     training.encoder_lr=1e-4 \
#     training.predictor_lr=3e-4 \
#     training.decoder_lr=3e-4 \
#     training.action_encoder_lr=1e-4 \
#     training.action_quantizer_lr=1e-4 \
#     training.state_quantizer_lr=1e-4 \
#     training.scheduler.type=cosine_with_warmup \
#     training.scheduler.warmup_epochs=3 \
#     training.scheduler.warmup_start_lr_factor=0.1 \
#     training.max_grad_norm=3.0 \
#     quantize=True \
#     state_vocabulary_size=128 \
#     action_vocabulary_size=16 \
#     model.train_encoder=True \
#     model.train_predictor=True \
#     model.train_decoder=True \
#     model.vcreg_loss_weight=10.0 \
#     model.quantization_loss_weight=2.0 \
#     encoder=conv2d \
#     predictor=conv3d \
#     decoder=vqvae \
#     state_quantizer=gumbel_quantizer \
#     action_quantizer=gumbel_quantizer \
#     disable_wandb=true \
#     force_restart=true
#
# echo ""
# echo "✅ Gumbel Quantizer test completed!"
# echo ""

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
