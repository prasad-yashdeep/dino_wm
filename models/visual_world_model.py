import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from objectives.vcreg import VCRegObjective, VCRegObjectiveConfig

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        action_encoder,
        decoder,
        predictor,
        action_quantizer=None,
        state_quantizer=None,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        vcreg_loss_weight=1.0,
        quantization_loss_weight=1.0,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.action_dim = action_dim * num_action_repeat
        self.action_quantizer = action_quantizer
        self.state_quantizer = state_quantizer
        self.quantize = False if (action_quantizer is None or state_quantizer is None) else True
        self.quantization_loss_weight = quantization_loss_weight
        self.emb_dim = self.encoder.emb_dim + self.action_dim * concat_dim

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"action encoder: {action_encoder}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")
        print(f"State quantizer: {state_quantizer}")
        print(f"Action quantizer: {action_quantizer}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        self.vcreg_loss_weight = vcreg_loss_weight
        self.vcreg_objective = VCRegObjective(VCRegObjectiveConfig())
        print(f"Using VCReg Loss: {self.vcreg_loss_weight > 0}")

    def train(self, mode=True):
        super().train(mode)
        # Explicitly set train/eval mode for each component based on flags
        if self.train_encoder:
            self.encoder.train(mode)
        else:
            self.encoder.eval()  # Force eval mode when frozen to freeze BatchNorm stats

        if self.predictor is not None:
            if self.train_predictor:
                self.predictor.train(mode)
            else:
                self.predictor.eval()

        # Action encoder always trains in this implementation
        self.action_encoder.train(mode)

        if self.decoder is not None:
            if self.train_decoder:
                self.decoder.train(mode)
            else:
                self.decoder.eval()

        # Quantizers always train when present
        if self.state_quantizer is not None:
            self.state_quantizer.train(mode)
        if self.action_quantizer is not None:
            self.action_quantizer.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.state_quantizer is not None:
            self.state_quantizer.eval()
        if self.action_quantizer is not None:
            self.action_quantizer.eval()

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual" (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, H, W, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)

        # Get spatial dimensions from visual embeddings
        b, t, H, W, visual_dim = z_dct['visual'].shape

        if self.concat_dim == 0:
            # Action is added as extra spatial token
            act_expanded = act_emb.unsqueeze(2).unsqueeze(3)  # (b, t, 1, 1, action_dim)
            act_repeated = repeat(act_expanded, "b t 1 1 d -> b t h 1 d", h=H)
            z = torch.cat([z_dct['visual'], act_repeated], dim=3)  # (b, t, H, W+1, emb_dim)
        elif self.concat_dim == 1:
            # Action is added as extra channels
            act_tiled = repeat(act_emb, "b t d -> b t h w d", h=H, w=W)
            act_repeated = act_tiled.repeat(1, 1, 1, 1, self.num_action_repeat)
            z = torch.cat([z_dct['visual'], act_repeated], dim=4)  # (b, t, H, W, visual_dim + action_dim)

        return z
    
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual" (b, t, H, W, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) ... -> b t ...", b=b)

        # Convert to consistent 4D format: (b, t, H, W, emb_dim)
        if len(visual_embs.shape) == 4:  # Patch-based encoders: (b, t, num_patches, emb_dim)
            # Reshape patches to 2D grid
            num_patches = visual_embs.shape[2]
            emb_dim = visual_embs.shape[3]
            # Assuming square grid of patches
            grid_size = int(num_patches ** 0.5)
            visual_embs = rearrange(visual_embs, "b t (h w) d -> b t h w d", h=grid_size, w=grid_size)

        return {"visual": visual_embs}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, H, W, emb_dim)
        output: z: (b, num_hist, H, W, emb_dim)
        """
        z = self.predictor(z)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, H, W, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs = self.decode_obs(z_obs)
        return obs

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, H, W, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, H, W, emb_dim = z_obs["visual"].shape
        visual = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
        }
        return obs
    
    def separate_src_tgt(self, obs, act):
        """
        input: obs (dict): "visual" (b, num_frames, 3, img_size, img_size)
               act: (b, num_frames, action_dim)
        output: obs_src (dict), obs_tgt (dict), act_src (tensor), act_tgt (tensor)
        """
        obs_src, obs_tgt = {}, {}
        obs_src['visual'] = obs['visual'][:, :self.num_hist]
        obs_tgt['visual'] = obs['visual'][:, self.num_pred:]
        act_src = act[:, :self.num_hist]
        act_tgt = act[:, self.num_pred:]
        return obs_src, obs_tgt, act_src, act_tgt

    def separate_obs_act_emb(self, z):
        """
        input: z (tensor): (b, num_frames, H, W, emb_dim)
        output: z_obs (tensor), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_obs, z_act = z[:, :, :, :-1, :], z[:, :, :, -1, :]
        elif self.concat_dim == 1:
            z_obs, z_act = z[..., :-self.action_dim], z[..., -self.action_dim:]
        return z_obs, z_act

    def separate_emb(self, z):
        """
        input: z (tensor): (b, num_frames, H, W, emb_dim)
        output: z_obs (dict), z_act (tensor)
        """
        z_obs, z_act = self.separate_obs_act_emb(z)
        z_obs = {"visual": z_obs}
        return z_obs, z_act

    def quantize_embeddings(self, z, quantize_action=True):
        """
        input : z: (b, num_frames, H, W, emb_dim)
        output: z_quantized: (b, num_frames, H, W, emb_dim)
        """
        quantization_loss, quantization_indices = 0., {}

        if self.concat_dim == 0:
            # Action embeddings are contained as extra spatial dimension
            z_state_q = z[:, :, :, :-1, :]   # Visual features
            z_action_q = z[:, :, :, -1, :]   # Action features
        elif self.concat_dim == 1:
            # Action embeddings are contained as extra channels
            z_state_q = z[..., :-self.action_dim]
            z_action_q = z[..., -self.action_dim:]

        if self.state_quantizer is not None:
            # We quantize the obs part of the embeddings
            state_quantization_loss, state_quantization_indices = 0, None
            z_state_q, state_quantization_loss, state_quantization_indices = self.state_quantizer(z_state_q)
            quantization_loss += state_quantization_loss
            quantization_indices['state'] = state_quantization_indices

        if quantize_action and self.action_quantizer is not None:
            # We quantize the action part of the embeddings
            act_quantization_loss, act_quantization_indices = 0, None
            z_action_q, act_quantization_loss, act_quantization_indices = self.action_quantizer(z_action_q)
            quantization_loss += act_quantization_loss
            quantization_indices['action'] = act_quantization_indices

        # z_q is obtained by concatenating the quantized state and action embeddings
        if self.concat_dim == 0:
            z_q = torch.cat([z_state_q, z_action_q.unsqueeze(3)], dim=3)
        elif self.concat_dim == 1:
            z_q = torch.cat([z_state_q, z_action_q], dim=4)

        return z_q, quantization_loss, quantization_indices



    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, H, W, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0.
        loss_components = {}

        # First, actions and observations are separated into source and target parts
        obs_src, obs_tgt, act_src, act_tgt = self.separate_src_tgt(obs, act)

        # Observations and actions are encoded by the student network
        z_src = self.encode(obs_src, act_src)

        # Optionally quantize the source embeddings
        if self.quantize:
            z_src, enc_quantization_loss, enc_quantization_indices = self.quantize_embeddings(z_src)

            # Compute codebook utilization for monitoring
            if 'state' in enc_quantization_indices:
                state_util, state_n_unique = self.state_quantizer.compute_codebook_utilization(
                    enc_quantization_indices['state']
                )
                loss_components["state_codebook_utilization"] = state_util
                loss_components["state_codebook_n_unique"] = state_n_unique

            if 'action' in enc_quantization_indices:
                action_util, action_n_unique = self.action_quantizer.compute_codebook_utilization(
                    enc_quantization_indices['action']
                )
                loss_components["action_codebook_utilization"] = action_util
                loss_components["action_codebook_n_unique"] = action_n_unique

        # Observations of the target are encoded by the teacher network (no grad)
        with torch.no_grad():
            z_tgt = self.encode(obs_tgt, act_tgt)
            # If quantization is enabled, we quantize the target embeddings (only state, not action)
            if self.quantize:
                z_tgt, _, _ = self.quantize_embeddings(z_tgt, quantize_action=False)

        # z_src and z_tgt are of shape (b, num_hist, H, W, emb_dim)
        # visual_src and visual_tgt are of shape (b, num_hist, 3, img_size, img_size)
        visual_src = obs_src['visual']
        visual_tgt = obs_tgt['visual']

        # If the encoder is trained, the regularization losses are computed
        if self.train_encoder:
            if self.vcreg_loss_weight > 0:
                # Compute vcreg loss on state embeddings only (exclude action)
                z_src_obs, _ = self.separate_obs_act_emb(z_src)
                vcreg_loss, vcreg_loss_components = self.vcreg_objective(z_src_obs)
                loss = loss + self.vcreg_loss_weight * vcreg_loss
                loss_components["vcreg_std_loss"] = vcreg_loss_components["std_loss"]
                loss_components["vcreg_cov_loss"] = vcreg_loss_components["cov_loss"]
                loss_components["vcreg_loss"] = vcreg_loss

        # Quantization loss should be computed even when encoder is frozen
        # This allows the quantizers to learn to represent the (frozen) encoder outputs
        if self.quantize:
            loss = loss + self.quantization_loss_weight * enc_quantization_loss
            loss_components["quantization_loss"] = enc_quantization_loss

        # If the world model has a predictor, we compute the prediction
        if self.predictor is not None:
            z_pred = self.predict(z_src)

            # Prediction is quantized if quantization is enabled (only state, not action)
            if self.quantize:
                z_pred, pred_quantization_loss, _ = self.quantize_embeddings(z_pred, quantize_action=False)

            # If a decoder is present, we decode the predicted embeddings
            # This is not used for training but for monitoring the performance of the model
            if self.decoder is not None:
                with torch.no_grad():
                    obs_pred = self.decode(z_pred.detach())
                    visual_pred = obs_pred['visual']
                    # We compare the decoded prediction with the ground truth visual
                    decoder_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                    loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # To train the predictor, we compute the loss between the predicted embeddings and the target embeddings
            # The loss is only computed for visual encoded dims (i.e. exclude action dims)
            z_pred_obs, _ = self.separate_obs_act_emb(z_pred)
            z_tgt_obs, _ = self.separate_obs_act_emb(z_tgt)
            z_src_obs, _ = self.separate_obs_act_emb(z_src)

            with torch.no_grad():
                z_visual_loss = self.emb_criterion(z_pred_obs, z_tgt_obs)
                z_collapse_loss = self.emb_criterion(z_src_obs, z_tgt_obs)  # checking if representation collapses

                # DEBUG: Log detailed statistics about embeddings
                if torch.rand(1).item() < 0.01:  # Log 1% of batches to avoid spam
                    print("\n" + "="*80)
                    print("🔍 COLLAPSE LOSS DEBUG (First batch sample)")
                    print("="*80)
                    print(f"z_src_obs shape: {z_src_obs.shape}")
                    print(f"z_tgt_obs shape: {z_tgt_obs.shape}")
                    print(f"\nz_src_obs stats:")
                    print(f"  Mean: {z_src_obs.mean().item():.6f}")
                    print(f"  Std:  {z_src_obs.std().item():.6f}")
                    print(f"  Min:  {z_src_obs.min().item():.6f}")
                    print(f"  Max:  {z_src_obs.max().item():.6f}")
                    print(f"\nz_tgt_obs stats:")
                    print(f"  Mean: {z_tgt_obs.mean().item():.6f}")
                    print(f"  Std:  {z_tgt_obs.std().item():.6f}")
                    print(f"  Min:  {z_tgt_obs.min().item():.6f}")
                    print(f"  Max:  {z_tgt_obs.max().item():.6f}")
                    print(f"\nDifference (z_src - z_tgt):")
                    diff = (z_src_obs - z_tgt_obs).abs()
                    print(f"  Mean abs diff: {diff.mean().item():.6f}")
                    print(f"  Max abs diff:  {diff.max().item():.6f}")
                    print(f"\nLosses:")
                    print(f"  z_collapse_loss: {z_collapse_loss.item():.10f}")
                    print(f"  z_visual_loss:   {z_visual_loss.item():.10f}")

                    # Check if embeddings are all zeros or very close to zero
                    if z_src_obs.abs().max() < 1e-3:
                        print("\n⚠️  WARNING: z_src_obs is nearly zero!")
                    if z_tgt_obs.abs().max() < 1e-3:
                        print("\n⚠️  WARNING: z_tgt_obs is nearly zero!")
                    if diff.mean() < 1e-3:
                        print("\n⚠️  WARNING: Embeddings are nearly identical!")
                    print("="*80 + "\n")

            z_loss = self.emb_criterion(z_pred_obs, z_tgt_obs.detach())

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_collapse_loss"] = z_collapse_loss
        else:
            visual_pred = None
            z_pred = None

        # We train the decoder if it is present
        if self.decoder is not None:
            with torch.no_grad():
                z_shape = [z_src.shape[0], act.shape[1], z_src.shape[2], z_src.shape[3], z_src.shape[4]]
                z = torch.zeros(z_shape, device=z_src.device, dtype=z_src.dtype)
                z[:, :self.num_hist, :, :, :] = z_src
                z[:, self.num_pred:, :, :, :] = z_tgt

            obs_reconstructed = self.decode(z.detach())  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            decoder_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            loss_components["decoder_loss_reconstructed"] = decoder_loss_reconstructed
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None

        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.quantize:
            # Quantize action separately
            if self.concat_dim == 0:
                act_emb_to_quantize = act_emb.unsqueeze(2)  # Add spatial dimension for quantization
            else:
                act_emb_to_quantize = act_emb
            act_emb_q, _, _ = self.action_quantizer(act_emb_to_quantize)
            if self.concat_dim == 0:
                act_emb = act_emb_q.squeeze(2)
            else:
                act_emb = act_emb_q

        if self.concat_dim == 0:
            # Replace the last spatial dimension with action embedding
            z[:, :, :, -1, :] = act_emb.unsqueeze(2)  # Add spatial dimension
        elif self.concat_dim == 1:
            # Get spatial dimensions
            H, W = z.shape[2], z.shape[3]
            act_tiled = repeat(act_emb, "b t d -> b t h w d", h=H, w=W)
            act_repeated = act_tiled.repeat(1, 1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z

    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, H, W, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        if self.quantize:
            z, _, _ = self.quantize_embeddings(z)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)
        return z_obses, z