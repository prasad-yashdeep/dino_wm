"""
Test script to verify the quantized DINO world model architecture.
"""
import torch
import sys
from models.visual_world_model import VWorldModel
from models.encoder.conv2d import Conv2DEncoder
from models.action_encoder.proprio import ProprioEncoder
from models.decoder.conv2d import Conv2DDecoder
from models.predictor.conv3d import Conv3DPredictorCausal
from models.quantizer.vector_quantizer import VectorQuantizer

def test_architecture():
    print("Testing Quantized DINO World Model Architecture...")

    # Hyperparameters
    batch_size = 2
    num_hist = 3
    num_pred = 1
    img_size = 224
    action_dim = 10
    num_action_repeat = 1
    action_emb_dim = 10

    # Create components
    print("\n1. Initializing components...")
    encoder = Conv2DEncoder(emb_dim=64, depth=4, hidden_dim=16)
    action_encoder = ProprioEncoder(in_chans=action_dim, emb_dim=action_emb_dim)
    decoder = Conv2DDecoder(emb_dim=64, output_channels=3, depth=4, hidden_dim=16)

    # For conv2d encoder, the output is (B, T, H, W, C)
    # With img_size=224 and depth=4, output size is 224 / 2^4 = 14x14
    predictor = Conv3DPredictorCausal(
        emb_dim=64 + action_emb_dim * num_action_repeat,  # visual + action
        hidden_dim=128,
        depth=3,
        num_frames=num_hist,
        dropout=0.1
    )

    # Create quantizers
    state_quantizer = VectorQuantizer(n_embed=128, embedding_dim=64, commitment_cost=0.25)
    action_quantizer = VectorQuantizer(n_embed=128, embedding_dim=action_emb_dim, commitment_cost=0.25)

    # Create world model
    print("2. Creating VWorldModel...")
    model = VWorldModel(
        image_size=img_size,
        num_hist=num_hist,
        num_pred=num_pred,
        encoder=encoder,
        action_encoder=action_encoder,
        decoder=decoder,
        predictor=predictor,
        state_quantizer=state_quantizer,
        action_quantizer=action_quantizer,
        action_dim=action_emb_dim,
        concat_dim=1,  # Concatenate along channel dimension
        num_action_repeat=num_action_repeat,
        train_encoder=False,
        train_predictor=True,
        train_decoder=True,
        vcreg_loss_weight=1.0,
        quantization_loss_weight=1.0,
    )

    # Create dummy data
    print("3. Creating dummy data...")
    obs = {
        'visual': torch.randn(batch_size, num_hist + num_pred, 3, img_size, img_size)
    }
    act = torch.randn(batch_size, num_hist + num_pred, action_dim)

    # Test forward pass
    print("4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        z_pred, visual_pred, visual_reconstructed, loss, loss_components = model(obs, act)

    print("\n5. Checking outputs...")
    print(f"   z_pred shape: {z_pred.shape if z_pred is not None else 'None'}")
    print(f"   visual_pred shape: {visual_pred.shape if visual_pred is not None else 'None'}")
    print(f"   visual_reconstructed shape: {visual_reconstructed.shape if visual_reconstructed is not None else 'None'}")
    print(f"   loss: {loss.item():.4f}")
    print(f"   loss_components: {list(loss_components.keys())}")

    # Check specific loss components
    print("\n6. Loss components:")
    for key, value in loss_components.items():
        if isinstance(value, torch.Tensor):
            print(f"   {key}: {value.item():.4f}")
        else:
            print(f"   {key}: {value:.4f}")

    # Test rollout
    print("\n7. Testing rollout...")
    obs_0 = {'visual': obs['visual'][:, :num_hist]}
    act_rollout = torch.randn(batch_size, num_hist + 5, action_dim)
    z_obses, z = model.rollout(obs_0, act_rollout)
    print(f"   Rollout z_obses shape: {z_obses['visual'].shape}")
    print(f"   Rollout z shape: {z.shape}")

    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        test_architecture()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
