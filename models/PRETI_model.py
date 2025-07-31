from .components import MAEViT, DecoderBlock
import torch.nn as nn
import torch
from functools import partial
import torch.nn.functional as F
import torchvision.transforms.v2 as T2
from .VGGPerceptualLoss import PerceptualLoss
from torch.cuda.amp import autocast
import timm
import math

'''
    1. vit-base
    2. recon loss / feature loss / age loss / gender loss / perceptual loss
'''
class SiameseMAE(MAEViT):
    
    @property
    def n_params(self, unit=1e6, ndigits=4):
        """ Number of parameters in model, divided by `unit`, rounded to `ndigits` digits """
        count_params = lambda params: round(sum(params) / unit, ndigits)
        return {
            "total_n_params" : count_params(p.numel() for p in self.parameters()),
            "total_n_trainable_params" : count_params(p.numel() for p in self.parameters() if p.requires_grad),
            "total_n_trainable_params_encoder" : count_params(p.numel() for p in self.blocks.parameters()),
            "total_n_trainable_params_decoder" : count_params(p.numel() for p in self.decoder_blocks.parameters()),
        }

    def __init__(self, pretrained = True, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.perceptual_loss_fn = PerceptualLoss(layers=[2, 7, 14], device='cuda')

        # Overwrite decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(kwargs["decoder_embed_dim"], kwargs["decoder_num_heads"])
            for _ in range(kwargs["decoder_depth"])
        ])

        print(f"\nSiamRMAE with: Decoder Embed Dim: {kwargs['decoder_embed_dim']}, Decoder Depth: {kwargs['decoder_depth']}, Decoder Num Heads: {kwargs['decoder_num_heads']}")

        # ImageNet Pretrained Weight
        if pretrained:
            print("Loading ImageNet pretrained weights for encoder...")
            # state_dict = timm.create_model("vit_small_patch16_224", pretrained=True).state_dict()
            state_dict = timm.create_model("vit_base_patch16_224", pretrained=True).state_dict()
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            print(f"Missing keys: {missing_keys}")
            print(f"Unexpected keys: {unexpected_keys}")
        
        # Learnable metadata embeddings
        self.age_embedding = nn.Parameter(torch.randn(1, 1, kwargs["embed_dim"]))
        self.gender_embedding = nn.Parameter(torch.randn(1, 1, kwargs["embed_dim"]))

        # Add Gender & Age Prediction Heads
        self.gender_classifier = nn.Linear(kwargs["embed_dim"], 2)  # 2-class classification
        self.age_regressor = nn.Linear(kwargs["embed_dim"], 1)  # Regression

        # After applying pretrained weights, reinitialize only the decoder and the additional components
        if pretrained:
            print("Pretrained weights loaded. Initializing decoder and additional parameters only...")
            self.initialize_decoder_weights()
        else:
            print("Training from scratch. Initializing all weights...")
            self.initialize_weights()

    # -------------------------------------------------------------------------------
    # [ENCODER] ---------------------------------------------------------------------
    def forward_encoder(self, base_x, mask_ratio, roi_mask):
        # Embed patches
        x = self.patch_embed(base_x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.roi_random_masking(x, roi_mask.squeeze(-1),  mask_ratio)
        
        # Metadata embedding
        age_emb = self.age_embedding.expand(x.shape[0], -1, -1)
        gender_emb = self.gender_embedding.expand(x.shape[0], -1, -1) 

        # Append CLS token and metadata embeddings
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    
        x = torch.cat([cls_tokens, age_emb, gender_emb, x], dim=1)
  
        # Apply Transformer blocks
        age_emb_out, gender_emb_out = None, None
 
        for blk in self.blocks:
            x = blk(x)

        # Extract age_emb and gender_emb from x
        age_emb_out = x[:, 1:2, :]  # Position of the first added age_emb
        gender_emb_out = x[:, 2:3, :]  # Position of the second added gender_emb

        x = self.norm(x)
        
        return torch.cat([x[:, :1, :], x[:, 3:, :]], dim=1), mask, ids_restore, age_emb_out, gender_emb_out

    
    def forward_encoder_no_masking(self, x):
        # Embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Metadata embedding
        age_emb = self.age_embedding.expand(x.shape[0], -1, -1)
        gender_emb = self.gender_embedding.expand(x.shape[0], -1, -1)

        # Append CLS token and metadata embeddings
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)

        x = torch.cat([cls_tokens, age_emb, gender_emb, x], dim=1)
        
        age_emb_out, gender_emb_out = None, None
        for blk in self.blocks:
            x = blk(x)
        
        # Extract age_emb and gender_emb from x
        age_emb_out = x[:, 1:2, :]  # Position of the first added age_emb
        gender_emb_out = x[:, 2:3, :]  # Position of the second added gender_emb
        
        x = self.norm(x)
        
        return torch.cat([x[:, :1, :], x[:, 3:, :]], dim=1), age_emb_out, gender_emb_out
    # -------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------
    # [DECODER] ---------------------------------------------------------------------
    # Utility functions for the decoder
    def _append_mask_token(self, x_latent, x_ids_restore):
        x_mask_tokens = self.mask_token.repeat(x_latent.shape[0], x_ids_restore.shape[1] + 1 - x_latent.shape[1], 1)
        x_ = torch.cat([x_latent[:, 1:, :], x_mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=x_ids_restore.unsqueeze(-1).repeat(1, 1, x_latent.shape[2]))  # unshuffle
        x = torch.cat([x_latent[:, :1, :], x_], dim=1)  # append cls token      
        return x  
    
    def _apply_decoder_blocks(self, q, kv):
        for blk in self.decoder_blocks:
            q = blk(q, kv)
        return q

    def forward_decoder(self, x_unmasked, x_masked_latent, x_masked_ids_restore):
        # embed tokens
        x_unmasked = self.decoder_embed(x_unmasked)
        x_masked_latent = self.decoder_embed(x_masked_latent)

        # append mask token
        x_masked_latent = self._append_mask_token(x_masked_latent, x_masked_ids_restore)

        # Add pos embed
        x_masked_latent = x_masked_latent + self.decoder_pos_embed

        # Apply decoder blocks (q: masked input, kv: unmasked original patches)
        x_masked_latent = self._apply_decoder_blocks(x_masked_latent, x_unmasked)

        # Normalize
        x_masked_latent = self.decoder_norm(x_masked_latent)

        # Predictor projection
        x_masked_latent = self.decoder_pred(x_masked_latent)

        # Remove cls token
        x_masked_latent = x_masked_latent[:, 1:, :] 

        return x_masked_latent
    # -------------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------------
    # [MASKING FUNCTIONS] -----------------------------------------------------------
    def apply_mask_on_unmasked_img(self, unmasked_img, masked_mask):
        batch_size, num_patches, embed_dim = unmasked_img.shape  # Original input shape

        # 1. Keep only the parts where the mask is `0` (remove if 1)
        mask_expanded = masked_mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # [batch_size, num_patches, embed_dim]
        unmasked_img_no_mask = unmasked_img[:, 1:, :][mask_expanded[:, :, :] == 0]  # Select only where mask is 0

        # 2. Use `.reshape()` instead of `.view()` for automatic dimension adjustment
        num_remaining_patches = unmasked_img_no_mask.numel() // (batch_size * embed_dim)
        unmasked_img_no_mask = unmasked_img_no_mask.reshape(batch_size, num_remaining_patches, embed_dim)

        return unmasked_img_no_mask

    def patchify_mask(self, mask, patch_size=16):
        """
        Convert the mask in the same way as image patchification.
        - mask: 5D mask of shape (B, 1, 1, H, W)
        - returns: (B, N, 1) (converted to match number of patches)
        """
        # Remove dimension D if it's 1 using squeeze
        mask = mask.squeeze(2)  # Change to shape (B, 1, H, W)
        
        B, C, H, W = mask.shape  # (e.g., 128, 1, 224, 224)

        # Convert mask to range 0–1
        mask = mask.float() / 255.0  # Convert to 0 or 1

        # Apply unfold to create patches
        mask = mask.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) # (128, 1, 14, 14, 16, 16)
        mask = mask.permute(0, 1, 2, 3, 4, 5).contiguous()  # Preserve dimension order after unfolding
        mask = mask.reshape(B, C, (H // patch_size) * (W // patch_size), -1)  # (128, 1, 196, 256) (B, C, N, P*P)
        
        # If any value inside a patch is 1, mark the patch as 1
        # Use average as the criterion
        mask = (mask.mean(dim=-1) > 0.5).float()  # (B, C, N) → mark as 1 if average ≥ 0.5

        # Remove channel dimension C → (B, N, 1)
        mask = mask.squeeze(1).unsqueeze(-1)  # (B, N, 1)

        return mask

    def get_mask_ratio(self, epoch, max_epochs, initial_ratio, final_ratio):
        """Cosine-based mask_ratio decay"""
        progress = 0.5 * (1 - math.cos(math.pi * epoch / max_epochs))  # Cosine decay
        return initial_ratio + (final_ratio - initial_ratio) * progress
    # -------------------------------------------------------------------------------

    # -------------------------------------------------------------------------------
    # [LOSS FUNCTIONS] --------------------------------------------------------------
    def compute_reconstruction_loss(self, masked_img, masked_pred, masked_mask):
        """Compute Reconstruction Loss"""
        return self.forward_loss(masked_img, masked_pred, masked_mask)

    def compute_feature_similarity_loss(self, masked_latent, unmasked_img_no_mask):
        """Compute Feature Consistency Loss (Cosine Similarity Loss)"""
        cos_sim_fn = nn.CosineEmbeddingLoss()
        
        # Get the embed_dim dynamically: [512, 2, 384]
        embed_dim = masked_latent.size(-1)
        masked_latent_flat = masked_latent[:, 1:, :].reshape(-1, embed_dim) # [512*2, embed_dim]
        unmasked_img_no_mask_flat = unmasked_img_no_mask.reshape(-1, embed_dim) # [512*2, embed_dim]
        
        # Normalize vectors
        masked_latent_flat = F.normalize(masked_latent_flat, p=2, dim=-1)
        unmasked_img_no_mask_flat = F.normalize(unmasked_img_no_mask_flat, p=2, dim=-1)
        
        # CosineEmbeddingLoss target (1: same direction)
        cosine_target = torch.ones(masked_latent_flat.shape[0], device=masked_latent_flat.device)
        
        return cos_sim_fn(masked_latent_flat, unmasked_img_no_mask_flat, cosine_target).mean()

    def compute_gender_loss(self, masked_gender_emb, unmasked_gender_emb, meta_data):
        """Compute Gender Prediction Loss (CrossEntropy)"""
        ce_loss_fn = nn.CrossEntropyLoss()
        
        masked_predicted_gender = self.gender_classifier(masked_gender_emb.squeeze(1))
        unmasked_predicted_gender = self.gender_classifier(unmasked_gender_emb.squeeze(1))
        
        gt_gender = meta_data["gender"].detach().long()
        
        return (ce_loss_fn(masked_predicted_gender, gt_gender) +
                ce_loss_fn(unmasked_predicted_gender, gt_gender)) / 2
    
    def compute_age_loss(self, masked_age_emb, unmasked_age_emb, meta_data):
        """Compute Age Prediction Loss (MSE)"""
        mse_loss_fn = nn.MSELoss()
        
        masked_predicted_age = self.age_regressor(masked_age_emb.squeeze(1))
        unmasked_predicted_age = self.age_regressor(unmasked_age_emb.squeeze(1))
        
        gt_age = meta_data["age"].float()
        
        return (mse_loss_fn(masked_predicted_age.squeeze(-1), gt_age) +
                mse_loss_fn(unmasked_predicted_age.squeeze(-1), gt_age)) / 2
    
    def unpatchify(self, x, patch_size=16, img_size=(224, 224)):
        """
        Convert patchified data back to original image shape (B, C, H, W)
        - x: [Batch, num_patches, patch_dim] (e.g., [B, 196, 768])
        - patch_size: size of each patch (e.g., 16)
        - img_size: original image size (e.g., 224x224)
        """
        B, num_patches, patch_dim = x.shape
        h, w = img_size
        h_patches = h // patch_size
        w_patches = w // patch_size
        C = patch_dim // (patch_size * patch_size)  # Get number of channels

        x = x.reshape(B, h_patches, w_patches, patch_size, patch_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.reshape(B, C, h, w)  # Convert to [B, C, H, W]
        return x

    def compute_perceptual_loss(self, masked_img, masked_pred):
        # Convert predicted patches to full image
        masked_pred = self.unpatchify(masked_pred)

        return self.perceptual_loss_fn(masked_img, masked_pred)
    # -------------------------------------------------------------------------------

    def forward(self, imgs, roi_masks, meta_data, inital_mask_ratio, final_mask_ratio, max_epochs, epoch):
        assert imgs.shape[1] == 2, "Number of frames must be equal to 2"

        # Decrease mask_ratio as epoch progresses
        mask_ratio = self.get_mask_ratio(epoch, max_epochs, inital_mask_ratio, final_mask_ratio)

        imgs = imgs.chunk(2, dim=1)
        imgs = [img.squeeze(1) for img in imgs]
        unmasked_img, masked_img = imgs[0], imgs[1] # masked_img.shape = torch.Size([128,3,224,224])

        roi_masks = roi_masks.chunk(2, dim=1)
        roi_masks = [roi_mask.squeeze(1) for roi_mask in roi_masks]
        unmasked_roi_mask, masked_roi_mask = roi_masks[0], roi_masks[1]
        
        masked_roi_mask = self.patchify_mask(masked_roi_mask, 16)  # Convert to patch format (patch size = 16)
        
        # Encode
        unmasked_img, unmasked_age_emb, unmasked_gender_emb = self.forward_encoder_no_masking(unmasked_img)
        masked_latent, masked_mask, masked_ids_restore, masked_age_emb, masked_gender_emb = self.forward_encoder(masked_img, mask_ratio, masked_roi_mask)
        
        # Decode
        masked_pred = self.forward_decoder(unmasked_img, masked_latent, masked_ids_restore)

        # (1) Reconstruction Loss
        reconstruct_loss = self.compute_reconstruction_loss(masked_img, masked_pred, masked_mask)
        
        # (2) Consistency Loss - using Cosine Similarity Loss
        unmasked_img_no_mask = self.apply_mask_on_unmasked_img(unmasked_img, masked_mask) # unmasked_img에 마스크 적용
        feature_sim_loss = self.compute_feature_similarity_loss(masked_latent, unmasked_img_no_mask)
        
        # (3) CrossEntropy Loss - gender prediction loss
        gender_ce_loss = self.compute_gender_loss(masked_gender_emb, unmasked_gender_emb, meta_data)

        # (4) MSE Loss - age prediction loss
        age_mse_loss = self.compute_age_loss(masked_age_emb, unmasked_age_emb, meta_data)

        # (5) VGGPerceptual Loss
        perceptual_loss = self.compute_perceptual_loss(masked_img, masked_pred)

        # Adjust weight for each loss (initial attempt: balance all equally)
        reconstruct_loss = reconstruct_loss  # reconstruction loss
        gender_ce_loss = (2 * gender_ce_loss)  # gender CE loss
        age_mse_loss = (0.0007 * age_mse_loss)  # age MSE loss
        feature_sim_loss = (2 * feature_sim_loss)  # feature similarity loss
        perceptual_loss = (0.3 * perceptual_loss)
        
        # Alternative weight adjustment
        reconstruct_loss = reconstruct_loss  # reconstruction loss
        feature_sim_loss = 0.4 * feature_sim_loss  # feature similarity
        gender_ce_loss = 0.2 * gender_ce_loss  # gender CE loss
        age_mse_loss = 0.2 * age_mse_loss  # age MSE loss
        perceptual_loss = 0.4 * perceptual_loss

        # Total Loss
        total_loss = (
            reconstruct_loss +
            feature_sim_loss +
            gender_ce_loss +
            age_mse_loss +
            perceptual_loss
        )

        return total_loss, reconstruct_loss, feature_sim_loss, gender_ce_loss, age_mse_loss, perceptual_loss, [masked_pred], [masked_mask]
    

def siam_mae_vit_small(patch_size=16, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def siam_mae_vit_small_big_decoder(patch_size=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def siam_mae_vit_base(patch_size=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# def siam_mae_vit_base_small_decoder(patch_size=16, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8, **kwargs):
#     model = SiameseMAE(
#         patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model

def siam_mae_vit_large(patch_size=16, decoder_embed_dim=1024, decoder_depth=12, decoder_num_heads=16, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
SIAM_MODELS = {
    "vits" : siam_mae_vit_small,
    "vits_big_decoder": siam_mae_vit_small_big_decoder,
    "vitb" : siam_mae_vit_base,
    # "vitb_small_decoder" : siam_mae_vit_base_small_decoder,
    "vitl" : siam_mae_vit_large
}