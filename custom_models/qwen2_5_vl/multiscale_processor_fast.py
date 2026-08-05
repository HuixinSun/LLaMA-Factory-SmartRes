# coding=utf-8

import copy
import glob
import json
import logging
import math
import random
import re
import traceback
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import patches

import os
import os as _os
from PIL import Image, ImageDraw
from scipy.ndimage import zoom

logger = logging.getLogger(__name__)

# Module-level global variable to store image processor reference
_IMAGE_PROCESSOR = None

def set_global_image_processor(processor):
    """Set the global image processor for visualization"""
    global _IMAGE_PROCESSOR
    _IMAGE_PROCESSOR = processor

def visualize_features_with_mask(features, mask, grid_thw, save_path='debug_features.png', 
                                  title='Feature Map with ROI Mask', batch_idx=0,
                                  image_data=None, bboxes=None, labels=None):
    """
    Visualize features as heatmap with positive regions marked
    (Only called in debug mode, safe to use CPU conversions)
    
    Args:
        features: [N, C] tensor - flattened feature tokens
        mask: [N] tensor - binary mask (1 for ROI, 0 for background)
        grid_thw: [num_images, 3] - temporal, height, width grid info
        save_path: path to save visualization
        title: plot title
        batch_idx: which image to visualize (for multi-image batches)
        image_data: dict with 'original', 'base', 'hr' PIL images
        bboxes: list of bboxes in [x1, y1, x2, y2] normalized format
        labels: list of object labels
    """
    try:
        # Move to CPU and convert to numpy
        features_np = features.detach().cpu().float().numpy()
        mask_np = mask.detach().cpu().float().numpy()
        
        # Get grid dimensions for the batch_idx-th image
        if grid_thw.dim() == 2:  # Multiple images
            t, h, w = grid_thw[batch_idx].int().tolist()
            start_idx = sum([int(grid_thw[i, 0] * grid_thw[i, 1] * grid_thw[i, 2]) 
                           for i in range(batch_idx)])
            num_tokens = t * h * w
            features_np = features_np[start_idx:start_idx + num_tokens]
            mask_np = mask_np[start_idx:start_idx + num_tokens]
        else:  # Single image
            t, h, w = grid_thw.int().tolist()
        
        # Average across channel dimension for visualization
        feature_map = features_np.mean(axis=-1)  # [N] -> scalar per token
        
        # Reshape to spatial grid (ignoring temporal for now)
        spatial_tokens = h * w
        feature_map = feature_map[:spatial_tokens].reshape(h, w)
        mask_map = mask_np[:spatial_tokens].reshape(h, w)
        
        # Create visualization (4 subplots if image_data provided, else 3)
        num_plots = 4 if image_data is not None else 3
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
        if num_plots == 3:
            axes = list(axes)
        
        # 1. Feature heatmap
        im1 = axes[0].imshow(feature_map, cmap='RdYlBu_r', aspect='auto')
        axes[0].set_title('Feature Activation (avg across channels)')
        axes[0].set_xlabel('Width')
        axes[0].set_ylabel('Height')
        plt.colorbar(im1, ax=axes[0])
        
        # 2. Mask overlay
        im2 = axes[1].imshow(mask_map, cmap='Reds', alpha=0.6, aspect='auto')
        axes[1].set_title('ROI Mask (1=positive, 0=background)')
        axes[1].set_xlabel('Width')
        axes[1].set_ylabel('Height')
        plt.colorbar(im2, ax=axes[1])
        
        # 3. Combined view
        axes[2].imshow(feature_map, cmap='RdYlBu_r', aspect='auto')
        # Overlay mask with contour
        mask_binary = (mask_map > 0.5).astype(float)
        axes[2].contour(mask_binary, colors='red', linewidths=2, levels=[0.5])
        axes[2].set_title('Features + ROI Contour (red)')
        axes[2].set_xlabel('Width')
        axes[2].set_ylabel('Height')
        
        # 4. Original image with bbox annotations (if provided)
        print(f"[Vis Check in visualize_features_with_mask]")
        print(f"  image_data: {image_data is not None}, type={type(image_data)}")
        print(f"  bboxes: {bboxes is not None}, type={type(bboxes) if bboxes is not None else None}")
        print(f"  num_bboxes: {len(bboxes) if bboxes else 0}")
        if bboxes:
            print(f"  first bbox: {bboxes[0] if len(bboxes) > 0 else 'N/A'}")
        print(f"  Condition check: image_data={image_data is not None}, bboxes={bboxes is not None}, len={len(bboxes) if bboxes else 0}")
        
        if image_data is not None and bboxes is not None and len(bboxes) > 0:
            # Use base resolution image for consistency with feature map
            img = image_data.get('base', image_data.get('original')).copy()
            draw = ImageDraw.Draw(img)
            
            # Get image dimensions
            orig_w, orig_h = image_data.get('orig_size', img.size)
            base_w, base_h = image_data.get('base_size', img.size)
            
            # Draw bboxes
            for idx, bbox in enumerate(bboxes):
                # bbox: [x1, y1, x2, y2] normalized [0, 1]
                x1, y1, x2, y2 = bbox
                
                # Convert to base image coordinates
                # bbox是相对原始图像的归一化坐标，需要转换到base图像像素坐标
                x1_px = int(x1 * orig_w * (base_w / orig_w))
                y1_px = int(y1 * orig_h * (base_h / orig_h))
                x2_px = int(x2 * orig_w * (base_w / orig_w))
                y2_px = int(y2 * orig_h * (base_h / orig_h))
                
                # Draw rectangle
                draw.rectangle([x1_px, y1_px, x2_px, y2_px], outline='red', width=3)
                
                # Draw label if available
                if labels and idx < len(labels):
                    label_text = labels[idx]
                    draw.text((x1_px + 2, y1_px + 2), label_text, fill='red')
            
            axes[3].imshow(img)
            axes[3].set_title(f'Input Image with GT Boxes\n({len(bboxes)} objects)')
            axes[3].axis('off')
        
        # Add statistics
        roi_features = features_np[mask_np > 0.5]
        bg_features = features_np[mask_np < 0.5]
        
        if len(roi_features) > 0 and len(bg_features) > 0:
            roi_mean = roi_features.mean()
            bg_mean = bg_features.mean()
            discrimination = abs(roi_mean - bg_mean)
            
            title_text = f'{title}\nROI mean: {roi_mean:.4f}, BG mean: {bg_mean:.4f}, '
            title_text += f'Discrimination: {discrimination:.6f}'
            if image_data is not None and bboxes is not None:
                title_text += f', Objects: {len(bboxes)}'
            
            fig.suptitle(title_text, fontsize=12)
        
        plt.tight_layout()
        
        # Create directory if needed
        _os.makedirs(_os.path.dirname(save_path) if _os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature visualization to {save_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")

def _clip_routing_mode() -> str:
    """How far the CLIP similarity map takes part in token selection.

    MTS_CLIP_ROUTING:
      unset / 0 / off   'off'    selection is the confidence head alone (default)
      train             'train'  similarity joins selection during TRAINING only;
                                 at inference the router falls back to the
                                 confidence head. This is variant (i),
                                 object-level semantics: the category label is
                                 ground truth and is a constant ("object") on the
                                 test splits, so it may guide learning but can
                                 never be read at test time.
      1 / both          'both'   similarity joins selection at training AND
                                 inference. This is variant (ii), query-level
                                 semantics: the referring expression is part of
                                 the user prompt, so it is available at test time.

    Default OFF: with MTS_CLIP_ROUTING unset every path below is byte-identical to
    the confidence-head-only router, so blind training and all existing
    revised-window jobs are unaffected.
    """
    v = _os.environ.get("MTS_CLIP_ROUTING", "").strip().lower()
    if v in ("", "0", "off", "false", "no"):
        return "off"
    if v in ("train", "train_only", "training"):
        return "train"
    return "both"


def _clip_routing_enabled() -> bool:
    """True when the CLIP branch is needed at all (build the modules, run them)."""
    return _clip_routing_mode() != "off"


def _clip_text_source() -> str:
    """Which text conditions the CLIP branch: 'label' or 'query'.

    'label' -- variant (i), object-level semantics. Built from the GT objects'
        `label` field via the frozen CLIP-ViT-B/32 text encoder + cross-attention.
        75 categories on the training split, so a real signal there; ground truth,
        and a single constant "object" on both test splits, hence train-only.
    'query' -- variant (ii), query-level semantics. The full referring expression
        from the user prompt, through the same encoder plus the learned projector
        that aligns it with the visual features. Available at inference.

    Default: 'label' when only ENABLE_CLIP_LOSS=1 (preserves the already-trained
    clip arm exactly) or when routing is train-only; 'query' for full routing.
    """
    mode = _clip_routing_mode()
    return _os.environ.get("MTS_CLIP_TEXT_SOURCE", "query" if mode == "both" else "label")


class TextGuidedVisualProjector(nn.Module):
    def __init__(self, visual_dim, text_dim):
        super().__init__()
        # CLIP components: load if ENABLE_CLIP_LOSS=1 (train the branch) or
        # MTS_CLIP_ROUTING=1 (use the branch for selection, incl. at inference)
        self.enable_clip_loss = bool(_os.environ.get("ENABLE_CLIP_LOSS", "")) or _clip_routing_enabled()

        self.visual_proj = nn.Linear(visual_dim, text_dim)

        if self.enable_clip_loss:
            self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
            self.text_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
            
            # Freeze CLIP text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            
            # cross-attention
            self.cross_attention = nn.MultiheadAttention(text_dim, num_heads=8)
            
            # 输出头 (per-token text similarity logits)
            self.similarity_head = nn.Linear(text_dim, 1)
            
            # Initialize similarity_head with small positive bias for better starting point
            with torch.no_grad():
                if self.similarity_head.bias is not None:
                    self.similarity_head.bias.fill_(0.0)
        else:
            # Placeholder: CLIP components not needed when disabled
            self.text_encoder = None
            self.text_tokenizer = None
            self.cross_attention = None
            self.similarity_head = None
        
        # Counter for saving visualizations
        self._attn_vis_counter = 0

        # `additional_target: visual.token_multiscale_processor` makes PEFT checkpoint
        # this whole module, which dragged the frozen 63M-param CLIP text encoder into
        # every adapter (338MB vs 81MB for the blind arm). It is never trained and is
        # rebuilt from `openai/clip-vit-base-patch32` in __init__, so drop it on save.
        # PEFT loads with strict=False (peft/utils/save_and_load.py), and older, fatter
        # checkpoints still load fine -- extra keys are simply consumed.
        # Escape hatch: MTS_SAVE_CLIP_TEXT_ENCODER=1.
        self._register_state_dict_hook(TextGuidedVisualProjector._drop_frozen_text_encoder)

    @staticmethod
    def _drop_frozen_text_encoder(module, state_dict, prefix, local_metadata):
        if _os.environ.get("MTS_SAVE_CLIP_TEXT_ENCODER", "0") == "1":
            return state_dict
        for key in [k for k in state_dict if k.startswith(f"{prefix}text_encoder.")]:
            del state_dict[key]
        return state_dict

    def _visualize_attention(self, attention_weights, text_prompts, mask_target=None):
        """
        可视化cross-attention map来诊断collapse
        (Only called in debug mode, safe to use CPU conversions)
        
        MultiheadAttention returns: [batch, query_len, key_len]
        - query = visual_features: [128, batch, 512] → query_len=128
        - key = text_embeddings: [text_len, batch, 512] → key_len=text_len
        So attention_weights: [batch, visual_len=128, text_len=?]
        mask_target: [seq_len] - GT mask where 1=ROI, 0=BG
        """
        try:
            
            # 先打印shape信息
            logger.info(f"[Attention Debug] attention_weights shape: {attention_weights.shape}")
            
            # attention_weights shape: [batch, visual_len=128, text_len=?]
            if attention_weights.dim() == 3:
                attn = attention_weights.squeeze(0)  # [visual_len=128, text_len=?]
            else:
                attn = attention_weights  # Already 2D
            
            attn_np = attn.detach().cpu().float().numpy()  # [128, ?]
            
            # 获取GT mask
            mask_np = None
            if mask_target is not None:
                mask_np = mask_target.detach().cpu().float().numpy()  # [128,]
                logger.info(f"[Attention Debug] GT mask - ROI: {mask_np.sum()}/{len(mask_np)}, ratio: {mask_np.mean():.2%}")
            
            # 解码text tokens来显示实际内容
            try:
                if isinstance(text_prompts, dict) and 'input_ids' in text_prompts:
                    # 获取tokenizer
                    tokenizer = self.text_encoder.config._name_or_path  # 这只是路径
                    # 直接显示token ids
                    token_ids = text_prompts['input_ids'][0].cpu().tolist()
                    text_info = f"Token IDs: {token_ids[:10]}..." if len(token_ids) > 10 else f"Token IDs: {token_ids}"
                else:
                    text_info = f"Text prompts type: {type(text_prompts)}"
            except Exception as e:
                text_info = f"Failed to decode: {e}"
            
            logger.info(f"[Attention Debug] attn_np shape: {attn_np.shape}, {text_info}")
            
            # 限制可视化数量
            if self._attn_vis_counter >= 20:
                return
            
            # 根据是否有mask决定子图数量
            num_plots = 4 if mask_np is not None else 3
            fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
            if num_plots == 3:
                axes = list(axes)
            
            # attn_np: [visual_len, text_len]
            num_visual, num_text = attn_np.shape
            logger.info(f"[Attention Debug] Plotting: {num_visual} visual tokens × {num_text} text tokens")
            
            # 1. Attention heatmap
            im = axes[0].imshow(attn_np, cmap='viridis', aspect='auto', interpolation='nearest')
            axes[0].set_xlabel(f'Text Token Index (Total: {num_text})', fontsize=14)
            axes[0].set_ylabel(f'Visual Token Index (Total: {num_visual})', fontsize=14)
            axes[0].set_title(f'Cross-Attention Map\n({num_visual} visual → {num_text} text)\n{text_info}', fontsize=16)
            axes[0].tick_params(axis='both', which='major', labelsize=12)
            # 显示实际的坐标刻度
            if num_text <= 10:
                axes[0].set_xticks(range(num_text))
            plt.colorbar(im, ax=axes[0])
            
            # 2. Attention distribution per visual token (看是否所有visual token都attend到相同text位置)
            # 每条线代表一个visual token对4个text token的attention权重
            for i in range(min(20, num_visual)):  # 只画前20个避免过于拥挤
                axes[1].plot(attn_np[i], alpha=0.2, linewidth=0.5, color='gray')
            axes[1].plot(attn_np.mean(axis=0), 'r-', linewidth=3, label='Mean across all visual', marker='o')
            axes[1].set_xlabel('Text Token Position', fontsize=14)
            axes[1].set_ylabel('Attention Weight', fontsize=14)
            axes[1].set_title(f'Attention per Text Token\n(Gray=first 20 visual, Red=mean)', fontsize=16)
            axes[1].set_xticks(range(num_text))
            axes[1].tick_params(axis='both', which='major', labelsize=12)
            axes[1].legend(fontsize=12)
            axes[1].grid(True, alpha=0.3)
            
            # 3. Attention variance (每个text token上，不同visual token的variance)
            # 如果variance很低，说明所有visual token对这个text token的attention权重都相同→collapse
            attn_var = attn_np.var(axis=0)  # [text_len] - variance across visual tokens
            axes[2].bar(range(num_text), attn_var, alpha=0.7, color='steelblue')
            axes[2].set_xlabel('Text Token Position', fontsize=14)
            axes[2].set_ylabel('Variance (across visual tokens)', fontsize=14)
            axes[2].set_title(f'Attention Variance per Text Token\n(Low variance = all visual attend similarly)', fontsize=16)
            axes[2].set_xticks(range(num_text))
            axes[2].tick_params(axis='both', which='major', labelsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # 4. GT区域 (ROI) vs 背景 (BG) 的attention对比
            if mask_np is not None and len(mask_np) == num_visual:
                roi_mask = mask_np > 0.5  # [visual_len] boolean
                bg_mask = ~roi_mask
                
                num_roi = roi_mask.sum()
                num_bg = bg_mask.sum()
                
                if num_roi > 0 and num_bg > 0:
                    # 计算ROI和BG的平均attention
                    roi_attn_mean = attn_np[roi_mask].mean(axis=0)  # [text_len]
                    bg_attn_mean = attn_np[bg_mask].mean(axis=0)  # [text_len]
                    
                    # 绘制对比图
                    x = np.arange(num_text)
                    width = 0.35
                    
                    axes[3].bar(x - width/2, roi_attn_mean, width, label=f'ROI ({int(num_roi)} tokens)', 
                               alpha=0.8, color='red')
                    axes[3].bar(x + width/2, bg_attn_mean, width, label=f'BG ({int(num_bg)} tokens)', 
                               alpha=0.8, color='blue')
                    
                    axes[3].set_xlabel('Text Token Position', fontsize=14)
                    axes[3].set_ylabel('Average Attention Weight', fontsize=14)
                    axes[3].set_title(f'GT Region Analysis\nROI vs Background Attention', fontsize=16)
                    axes[3].set_xticks(x)
                    axes[3].tick_params(axis='both', which='major', labelsize=12)
                    axes[3].legend(fontsize=12)
                    axes[3].grid(True, alpha=0.3, axis='y')
                    
                    # 计算ROI和BG的attention差异
                    attn_diff = roi_attn_mean - bg_attn_mean
                    mean_diff = attn_diff.mean()
                    
                    # # 在图上添加差异信息
                    # axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                    # axes[3].text(0.02, 0.98, f'Mean Diff: {mean_diff:.4f}\n'
                    #             f'{"✓ ROI attends more" if mean_diff > 0 else "⚠️ BG attends more"}',
                    #             transform=axes[3].transAxes, 
                    #             verticalalignment='top',
                    #             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    #             fontsize=10)
                else:
                    axes[3].text(0.5, 0.5, 'No ROI or BG tokens', 
                                ha='center', va='center', transform=axes[3].transAxes, fontsize=14)
                    axes[3].set_title('GT Region Analysis\n(No valid data)', fontsize=16)
            
            # 添加统计信息
            mean_var = attn_var.mean()
            max_var = attn_var.max()
            
            title_text = f'Attention Analysis (Batch {self._attn_vis_counter})\n'
            title_text += f'Mean Variance: {mean_var:.6f}, Max Variance: {max_var:.6f}'
            if mask_np is not None:
                roi_ratio = mask_np.mean()
                title_text += f', ROI Ratio: {roi_ratio:.1%}'
            
            fig.suptitle(title_text, fontsize=18, color='red' if mean_var < 0.01 else 'green')
            
            plt.tight_layout()
            
            # 保存
            debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
            _os.makedirs(debug_dir, exist_ok=True)
            save_path = f'{debug_dir}/attention_map_batch{self._attn_vis_counter}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"[Attention Vis] Saved to {save_path}, Mean Var={mean_var:.6f}")
            
            self._attn_vis_counter += 1
            
        except Exception as e:
            logger.warning(f"Failed to visualize attention: {e}")
    
    def forward(self, visual_tokens, text_prompts):
        # 获取文本embedding（支持 dict 或原始字符串/列表），并对齐到编码器设备
        # import pdb;pdb.set_trace()
        def _to_device(batch, device):
            return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        enc_device = next(self.text_encoder.parameters()).device
        toks = _to_device(text_prompts, enc_device)

        text_outputs = self.text_encoder(**toks)
        text_embeddings = text_outputs.last_hidden_state  # (batch, text_seq_len, text_dim), [1, ?, 512]
        
        # visual_proj: 降维，1152→512 把视觉特征对齐到文本维度后做跨注意力
        visual_features = self.visual_proj(visual_tokens)  # (batch, seq_len, text_dim), [1, 128, 512]
        
        # 调试：打印实际shape
        if _os.environ.get("MULTISCALE_DEBUG", ""):
            logger.info(f"[Forward Debug] visual_features: {visual_features.shape}, text_embeddings: {text_embeddings.shape}")
            if 'input_ids' in toks:
                logger.info(f"[Forward Debug] input_ids shape: {toks['input_ids'].shape}, values: {toks['input_ids'][0].tolist()}")
        
        # Cross-attention: visual as query, text as K/V
        attended_visual, attention_weights = self.cross_attention(
            visual_features.transpose(0,1), # [128, 1, 512]
            text_embeddings.transpose(0,1), # [text_len, 1, 512]
            text_embeddings.transpose(0,1) # [text_len, 1, 512]
        )

        # 预测输出 (return raw logits for BCEWithLogitsLoss)
        attended_visual = attended_visual.transpose(0,1)  # (batch, seq_len, dim)
        similarities_logits = self.similarity_head(attended_visual)  # (batch, seq_len, 1) raw logits
        
        # 注意：这里不可视化，因为没有mask_target信息
        # 可视化将在MultiScaleTokenProcessor中完成
        
        return {
            'similarities': similarities_logits,  # Return logits, not probabilities
            'attention_weights': attention_weights,
            'text_prompts': text_prompts  # 保存text_prompts用于后续可视化
        }

## 
class MultiScaleTokenProcessor(nn.Module):
    """
    Multi-scale token processor that implements adaptive resolution processing.
    
    The processor works in two phases:
    1. Initial low-res sweep (base at 224*224) - entire image encoded at coarse scale
    2. Selective high-res sampling - applied only to regions with critical small objects
    """
    
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        embed_dim: int = 1152,
        in_channels: int = 3,
        scale_levels: int = 2,
        conf_thresh: float = 0.5,
        scale_thresh: float = 0.8,
        base_resolution: int = 224,
        high_res_scale: float = 2.0,
        patch_embed: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__()
        self.patch_size = patch_size # 14 = patch size
        self.embed_dim = embed_dim # 1152 = embedding dimension
        self.in_channels = in_channels # 3 = number of channels in the input image
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.scale_levels = scale_levels # 2 = coarse + fine scales
        self.conf_thresh = conf_thresh # 0.5 = minimum confidence score for a region to be considered valid for processing
        self.sims_thresh = scale_thresh # 0.8 = scale importance threshold for a region to be considered valid for processing
        self.base_resolution = base_resolution # 224 = initial processing resolution for the entire image (pixels)
        self.high_res_scale = high_res_scale # 2.0 = scale factor between base and original image resolution
        self.scale_layer = kwargs.get('scale_layer', None)  # Store which transformer layer to use for features
        

        self._last_text_prompt = None
        self.text_visual_projector = TextGuidedVisualProjector(visual_dim=self.embed_dim, text_dim=512)
        
        # # Use pre-trained weights from the original patch embedding
        if patch_embed is not None:
            self.encoder_patch_proj = nn.ModuleList([
                copy.deepcopy(patch_embed.proj) for _ in range(scale_levels)
            ])
        else:
            # Random initialization for each scale
            self.encoder_patch_proj = nn.ModuleList([
                nn.Conv3d(in_channels, embed_dim, kernel_size=[1, patch_size, patch_size], 
                         stride=[1, patch_size, patch_size], bias=False)
                for _ in range(scale_levels)
            ])
        
        # scale prediction MLP (3 layers, output dim=1 for confidence only)
        self.scale_head = self._build_mlp(embed_dim, embed_dim, 1, num_layers=3)
        
        with torch.no_grad():
            if isinstance(self.scale_head, nn.Sequential):
                last_layer = self.scale_head[-1]
                if isinstance(last_layer, nn.Linear) and last_layer.bias is not None:
                    # Set bias to 0.0 for confidence
                    # confidence: sigmoid(0.0) = 0.5 (neutral start, will be trained)
                    last_layer.bias[0].fill_(0.0)
        
        # Temperature for differentiable sigmoid-based thresholding
        # Training: use soft approximation (temperature > 0)
        # Inference: use hard threshold (temperature → 0)
        self.temperature = 1.0
        
        # Gradient monitoring
        self._grad_stats = {}
        self._grad_print_counter = 0  # Print every N batches

    
    def print_gradient_stats(self):
        """Print gradient statistics (call this during training)"""
        if not self._grad_stats:
            print("  [Gradient] No gradients captured (may indicate: no backward pass yet, or frozen parameters)")
            return
        
        print("  [Gradient Statistics]")
        for name, stats in self._grad_stats.items():
            print(f"    {name}: mean={stats['mean']:7.5f}, std={stats['std']:7.5f}, "
                  f"norm={stats['norm']:7.3f}, range=[{stats['min']:6.3f}, {stats['max']:6.3f}]")
        
        # Print confidence predictor (scale_head) gradients
        print("\n  [Confidence Predictor - scale_head]")
        has_scale_head_grad = False
        for i, layer in enumerate(self.scale_head):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad = layer.weight.grad
                print(f"    Layer {i} weight: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                      f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
                has_scale_head_grad = True
            if hasattr(layer, 'bias') and layer.bias is not None and layer.bias.grad is not None:
                grad = layer.bias.grad
                print(f"    Layer {i} bias: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                      f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
        if not has_scale_head_grad:
            print("    No gradients (frozen or not computed)")
        
        # Print CLIP text-visual projector gradients
        print("\n  [CLIP Text-Visual Projector]")
        has_projector_grad = False
        
        # Visual projection
        if hasattr(self.text_visual_projector.visual_proj, 'weight') and self.text_visual_projector.visual_proj.weight.grad is not None:
            grad = self.text_visual_projector.visual_proj.weight.grad
            print(f"    visual_proj.weight: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                  f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
            has_projector_grad = True
        
        # Cross-attention (query, key, value projections)
        attn = self.text_visual_projector.cross_attention
        if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight.grad is not None:
            grad = attn.in_proj_weight.grad
            print(f"    cross_attn.in_proj: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                  f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
            has_projector_grad = True
        
        if hasattr(attn, 'out_proj') and hasattr(attn.out_proj, 'weight') and attn.out_proj.weight.grad is not None:
            grad = attn.out_proj.weight.grad
            print(f"    cross_attn.out_proj: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                  f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
            has_projector_grad = True
        
        # Similarity head
        if hasattr(self.text_visual_projector.similarity_head, 'weight') and self.text_visual_projector.similarity_head.weight.grad is not None:
            grad = self.text_visual_projector.similarity_head.weight.grad
            print(f"    similarity_head.weight: mean={grad.mean().item():7.5f}, std={grad.std().item():7.5f}, "
                  f"norm={grad.norm().item():7.3f}, range=[{grad.min().item():6.3f}, {grad.max().item():6.3f}]")
            has_projector_grad = True
        
        if not has_projector_grad:
            print("    No gradients (frozen or not computed)")
    
    def differentiable_threshold(self, values: torch.Tensor, threshold: float, 
                                temperature: float = None, hard: bool = None) -> torch.Tensor:
        """
        Differentiable threshold using Straight-Through Estimator (STE).
        
        Forward pass: Hard threshold (discrete)
        Backward pass: Soft sigmoid (continuous gradients)
        
        Args:
            values: Input tensor (e.g., confidence scores)
            threshold: Threshold value
            temperature: Temperature for soft approximation (default: self.temperature)
            hard: Use hard threshold in forward (default: not self.training)
        
        Returns:
            Boolean mask (hard) or soft probabilities (training)
        """
        if temperature is None:
            temperature = self.temperature
        if hard is None:
            hard = not self.training
        
        # Soft approximation: sigmoid((x - threshold) / temperature)
        # When temp→0, this becomes a step function
        # When temp=1, this is a smooth sigmoid
        soft = torch.sigmoid((values - threshold) / temperature)
        
        if hard:
            # Straight-Through Estimator (STE):
            # Forward: hard threshold
            # Backward: gradient of soft approximation
            hard_mask = (values >= threshold).float()  # Use >= to include threshold=0 case
            # soft - soft.detach()
            # 前向：值为0
            # 反向：∂hard_mask/∂values + (∂soft/∂values - 0) = ∂soft/∂values
            return hard_mask + (soft - soft.detach())
        else:
            # Pure soft (for debugging)
            return soft
    
    def _build_mlp(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Module:
        """Build MLP network matching SAT model architecture"""
        assert num_layers >= 1, "num_layers must be at least 1"
        
        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)
        
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for i, (n, k) in enumerate(zip([input_dim] + h, h + [output_dim])):
            layers.append(nn.Linear(n, k))
            if i < num_layers - 1:  # No activation after last layer
                layers.append(nn.ReLU())
        
        return nn.Sequential(*layers)
    
    def get_scale_map(self, tokens: torch.Tensor, debug_ms: bool = False) -> torch.Tensor:
        """
        Predict scale importance map from tokens
        
        Args:
            tokens: Input tokens [num_patches, embed_dim]
            debug_ms: Whether to print debug information
            
        Returns:
            scale_map: [num_patches, 1] - confidence scores (logits)
        """
        # Print feature distribution BEFORE MLP (only if debug enabled)
        # import pdb; pdb.set_trace()
        if debug_ms:
            print(f"[NORM DEBUG] Input tokens - mean: {tokens.mean().item():.6f}, std: {tokens.std().item():.6f}, min: {tokens.min().item():.6f}, max: {tokens.max().item():.6f}")
            hist, bins = torch.histogram(tokens.float().cpu(), bins=50)
            print(f"[NORM DEBUG] Histogram - min: {bins[0]:.6f}, max: {bins[-1]:.6f}, mean: {tokens.mean().item():.6f}, std: {tokens.std().item():.6f}")

        # 去掉norm
        # scale_map_logits = self.scale_head(normalized_tokens)  # [num_patches, 1] - raw logits

        # MLP prediction (3-layer: Linear→ReLU→Linear→ReLU→Linear)，无norm，logits
        scale_map_logits = self.scale_head(tokens)  # [num_patches, 1] - confidence logits
        
        # Register gradient hook on output tensor (only in debug mode to avoid GPU sync)
        if debug_ms and self.training and scale_map_logits.requires_grad:
            def capture_grad(grad):
                if grad is not None:
                    self._grad_stats['scale_logits'] = {
                        'mean': grad.mean().item(),
                        'std': grad.std().item(),
                        'max': grad.max().item(),
                        'min': grad.min().item(),
                        'norm': grad.norm().item()
                    }
                return grad
            scale_map_logits.register_hook(capture_grad)
        
        # if debug_ms:
        #     print(f"[NORM DEBUG] After MLP (logits) - mean: {scale_map_logits.mean().item():.6f}, std: {scale_map_logits.std().item():.6f}, min: {scale_map_logits.min().item():.6f}, max: {scale_map_logits.max().item():.6f}")
        
        return scale_map_logits

    def _budgeted_reallocation(self, conf_prob, sims_prob, conf_binary, stored_lr_grid_thw,
                               budget):
        """Let the similarity map choose WHICH cells go high-res, not HOW MANY.

        The plain union `max(conf_mask, sims_mask)` is unusable: the similarity head is
        informative but *uncalibrated* -- its sigmoid sits above `sims_thresh` almost
        everywhere -- so the union selected 100% of cells (`kept=1.0000`) and the arm
        trained on a router that had stopped selecting. Thresholding fails on a
        miscalibrated score; *ranking* does not, because sigmoid is monotone in the
        logit. So we rank instead of threshold, and cap the count.

        Per image: convert each head's score to its within-image percentile rank (which
        makes two differently-calibrated heads commensurable), take the elementwise max
        (keeping the OR semantics of the original design -- a cell is routed if EITHER
        head ranks it high), and keep the top k.

        `budget='conf'` sets k to the number of cells the confidence head would have
        selected on its own, so the arm is compute-matched to the query-agnostic
        baseline image by image and any accuracy difference comes from the allocation
        rather than from spending more. A float in (0,1] sets a fixed ratio instead.

        Straight-through, like `differentiable_threshold`: the forward pass is the exact
        top-k (ranks are integer-valued and carry no gradient), while the backward pass
        is taken through `max(conf_prob, sims_prob)` in probability space, so both heads
        still receive gradient. In the revised-window path this is belt-and-braces --
        the mask is consumed as `(need_highres_mask > 0)`, and the heads are trained by
        `loss_conf` / `loss_clip` -- but it keeps the contract of the surrounding code.
        """
        n_imgs = stored_lr_grid_thw.shape[0]
        spans, off = [], 0
        for row in stored_lr_grid_thw.tolist():
            n = int(row[0]) * int(row[1]) * int(row[2])
            spans.append((off, off + n))
            off += n

        if budget == "conf":
            # one sync for the whole batch rather than one per image
            counts = torch.stack([conf_binary[lo:hi].sum() for lo, hi in spans]).tolist()
        else:
            r = float(budget)
            assert 0.0 < r <= 1.0, f"MTS_ROUTING_BUDGET must be 'conf' or in (0,1], got {r}"
            counts = [r * (hi - lo) for lo, hi in spans]

        out = torch.zeros_like(conf_prob)
        for (lo, hi), k in zip(spans, counts):
            n = hi - lo
            k = int(min(max(round(float(k)), 0), n))
            if k == 0 or n == 0:
                continue
            c, s = conf_prob[lo:hi], sims_prob[lo:hi]
            denom = max(n - 1, 1)
            rank_c = torch.argsort(torch.argsort(c)).to(c.dtype) / denom
            rank_s = torch.argsort(torch.argsort(s)).to(s.dtype) / denom
            score = torch.maximum(rank_c, rank_s)

            # Select from top-k indices, not from `score >= tau`: rank ties at the cut
            # would otherwise let k+1 cells through and break the compute match.
            idx = torch.topk(score, k).indices
            hard = torch.zeros_like(c)
            hard[idx] = 1.0

            # Backward path in probability space so gradient still reaches both heads
            # (`argsort` is integer-valued and would sever it). Threshold is the
            # probability of the least-confident selected cell; a 0-dim tensor, so no
            # GPU->CPU sync here.
            combined = torch.maximum(c, s)
            tau = combined[idx].min()
            soft = torch.sigmoid((combined - tau) / self.temperature)
            out[lo:hi] = hard + (soft - soft.detach())
        return out

    def _query_conditioned_sims(self, intermediate_features, stored_lr_grid_thw, instruction):
        """Per-token CLIP similarity logits conditioned on the USER QUERY.

        The legacy branch in forward() conditions on `text_prompt`, which the collator
        decodes from `labels` -- i.e. the GT answer. That is usable as a training
        signal but is ground truth, so it cannot drive selection at inference. This
        path uses `instruction` instead, which exists at train and test time alike.

        Returns flattened logits [sum(t*H*W)] aligned with `intermediate_features`,
        or None when there is no usable instruction.
        """
        proj = self.text_visual_projector
        if proj.text_tokenizer is None or instruction is None or stored_lr_grid_thw is None:
            return None

        texts = instruction if isinstance(instruction, (list, tuple)) else [instruction]
        n_imgs = stored_lr_grid_thw.shape[0]
        sims, start_idx = [], 0
        for img_idx in range(n_imgs):
            row = stored_lr_grid_thw[img_idx]
            num_tokens = int(row[0]) * int(row[1]) * int(row[2])
            end_idx = start_idx + num_tokens
            feats = intermediate_features[start_idx:end_idx]

            # one instruction per sample; every sample here carries exactly one image,
            # so fall back to the last one if a sample ever contributes several
            text = str(texts[img_idx]) if img_idx < len(texts) else str(texts[-1])
            text = text.strip()
            if not text:
                # no query -> contribute nothing rather than a constant-bias map
                sims.append(torch.zeros(num_tokens, device=feats.device, dtype=feats.dtype))
                start_idx = end_idx
                continue

            toks = proj.text_tokenizer(
                [text], padding=True, truncation=True, max_length=77, return_tensors='pt'
            )
            toks = {k: v.to(device=feats.device) for k, v in toks.items()}
            out = proj(feats.unsqueeze(0), toks)['similarities']
            sims.append(out.reshape(-1))
            start_idx = end_idx

        return torch.cat(sims, dim=0) if sims else None

    def forward(self,
                intermediate_features=None,
                stored_lr_grid_thw=None,
                text_prompt=None,
                instruction: Optional[str] = None):
        """
        Multi-scale forward pass:
        1. Use intermediate transformer features for scale prediction  
        2. Extract HR embeddings from pixels for selected positions
        3. Create sparse spatial grid with LR tokens at centers and HR tokens at subdivision positions
        
        Args:
            intermediate_features: Features from intermediate transformer layer (level 3) [seq_len, embed_dim]
            stored_lr_grid_thw: Low-resolution grid information [1, 3] -> [T, H_lr, W_lr]
            hr_grid_thw: Preprocessed HR grid information [1, 3] -> [T, H_hr, W_hr]
            
        Returns:
            torch.Tensor: Sparse embeddings with shape [hr_capacity, embed_dim]
                         - LR tokens placed at spatial centers within HR grid
                         - HR tokens placed at subdivision positions
                         - Empty positions filled with zeros
        """
        debug_ms = _os.environ.get("MULTISCALE_DEBUG", "0") == "1"
        
        def _build_lr_mask(bboxes, grid_thw):    
            # 计算总token数 - 向量化，避免.tolist()减少CPU-GPU同步
            grid_sizes = grid_thw.long()  # [num_frames, 3]
            tokens_per_frame_tensor = grid_sizes[:, 0] * grid_sizes[:, 1] * grid_sizes[:, 2]
            total_tokens = int(tokens_per_frame_tensor.sum())
            # Keep tensor form for later use to avoid repeated .tolist() calls
            tokens_per_frame = tokens_per_frame_tensor.tolist()
            
            # 初始化flattened mask - use zeros_like pattern for better efficiency
            template_tensor = intermediate_features[:total_tokens, 0] if total_tokens <= intermediate_features.shape[0] else intermediate_features.new_zeros(total_tokens)
            conf = torch.zeros_like(template_tensor, dtype=intermediate_features.dtype)
            
            P = int(self.patch_size)  # patch size, typically 14
            
            token_start_idx = 0
            for frame_idx, frame_grid in enumerate(grid_thw):
                # Avoid .tolist() to prevent CPU-GPU sync, use tensor indexing instead
                t, H, W = int(frame_grid[0]), int(frame_grid[1]), int(frame_grid[2])
                frame_tokens = tokens_per_frame[frame_idx]
                
                # bbox
                frame_bbox = None
                if frame_idx < len(bboxes) and bboxes[frame_idx] is not None:
                    frame_bbox = bboxes[frame_idx]
                
                # mask - 向量化计算
                if frame_bbox is not None:
                    # 缩放框至LR分辨率
                    Hpx = H * P  # LR pixel height = grid height * patch size
                    Wpx = W * P  # LR pixel width = grid width * patch size
                    
                    # 处理当前帧的bbox
                    if isinstance(frame_bbox, (list, tuple)) and len(frame_bbox) == 4:
                        # bbox已经是LR坐标系（由数据预处理生成），直接使用
                        x1, y1, x2, y2 = map(float, frame_bbox)
                        
                        # 转为整数（四舍五入）
                        x1 = int(math.floor(x1))
                        x2 = int(math.ceil(x2))
                        y1 = int(math.floor(y1))
                        y2 = int(math.ceil(y2))
                        
                        # clip to LR image bounds
                        x1 = max(0, min(x1, Wpx)); x2 = max(0, min(x2, Wpx))
                        y1 = max(0, min(y1, Hpx)); y2 = max(0, min(y2, Hpx))
                        
                        if debug_ms:
                            print(f"[Build LR Mask] Frame {frame_idx}: bbox=[{x1},{y1},{x2},{y2}], LR size=({Hpx}x{Wpx}), grid=({H}x{W})")
                        
                        # 向量化计算：创建token坐标网格
                        # [H, W] grid of token coordinates
                        yy_grid = torch.arange(H, device=conf.device).view(H, 1).expand(H, W)
                        xx_grid = torch.arange(W, device=conf.device).view(1, W).expand(H, W)
                        
                        # 计算每个token覆盖的像素范围
                        yy0 = yy_grid * P
                        yy1 = torch.clamp((yy_grid + 1) * P, max=Hpx)
                        xx0 = xx_grid * P
                        xx1 = torch.clamp((xx_grid + 1) * P, max=Wpx)
                        
                        # 计算交集 (vectorized)
                        ix1 = torch.clamp(xx0, min=x1)
                        iy1 = torch.clamp(yy0, min=y1)
                        ix2 = torch.clamp(xx1, max=x2)
                        iy2 = torch.clamp(yy1, max=y2)
                        
                        # 交集面积
                        inter = torch.clamp(ix2 - ix1, min=0) * torch.clamp(iy2 - iy1, min=0)
                        
                        # [H, W] mask: 1 if overlap > 0, else 0
                        frame_mask = (inter > 0).float().flatten()  # [H*W]
                        
                        # 对于t个时间步，复制mask
                        for _t in range(t):
                            start = token_start_idx + _t * H * W
                            end = start + H * W
                            conf[start:end] = frame_mask
                
                # 移动到下一帧的起始位置
                token_start_idx += frame_tokens
            
            return conf

        # Scale prediction from intermediate transformer features
        scale_map_raw = self.get_scale_map(intermediate_features, debug_ms=debug_ms)  # Returns raw logits (before sigmoid)
        scale_map = scale_map_raw.sigmoid()  # Apply sigmoid for threshold checks
        
        # Register gradient hook and save for monitoring (only in debug mode)
        if debug_ms:
            self._last_scale_map = scale_map
            
            if self.training and scale_map_raw.requires_grad:
                def capture_scale_map_grad(grad):
                    if grad is not None:
                        self._grad_stats['scale_map_logits'] = {
                            'mean': grad.mean().item(),
                            'std': grad.std().item(),
                            'max': grad.max().item(),
                            'min': grad.min().item(),
                            'norm': grad.norm().item()
                        }
                    return grad
                scale_map_raw.register_hook(capture_scale_map_grad)
            
        # 2) Compute auxiliary loss for supervisions
        mask_target = None
        sims_prob_raw = None
        enable_clip_loss = bool(_os.environ.get("ENABLE_CLIP_LOSS", ""))
        clip_routing_mode = _clip_routing_mode()          # 'off' | 'train' | 'both'
        clip_routing = clip_routing_mode != "off"
        clip_text_source = _clip_text_source()

        if (text_prompt is not None and enable_clip_loss and clip_text_source == "label"
                and self.text_visual_projector.text_tokenizer is not None):
            # 多图
            all_bboxes = []
            all_labels = []
            all_similarities = []
            
            # 保存attention信息和第一个图像的bbox/labels用于可视化
            saved_attention_weights = None
            saved_text_prompts = None
            saved_img_start_idx = 0
            first_img_bboxes = []
            first_img_labels = []

            # 动态提取每个图像的特征
            start_idx = 0
            for img_idx, (prompt, frame_grid) in enumerate(zip(text_prompt, stored_lr_grid_thw)):
                # 计算当前图像的token数量 - avoid .tolist() for better performance
                t, H, W = int(frame_grid[0]), int(frame_grid[1]), int(frame_grid[2])
                num_tokens = t * H * W
                end_idx = start_idx + num_tokens
                
                # 当前图像的特征
                current_img_features = intermediate_features[start_idx:end_idx]
                
                # 解析文本提示
                body = re.sub(r"^```\w*\n|```\s*$", "", str(prompt).strip(), flags=re.S)
                objects = json.loads(body) if body else []
                
                # 收集当前图像的所有objects并构建multi-token text sequence
                curr_img_descriptions = []
                
                for obj in objects:
                    bbox = obj.get("bbox_2d", None)
                    label = obj.get("label", None)
                    
                    if bbox is not None and label is not None:
                        all_bboxes.append(bbox)
                        all_labels.append(label)
                        
                        # 保存第一个图像的bboxes和labels用于可视化
                        if img_idx == 0:
                            first_img_bboxes.append(bbox)
                            first_img_labels.append(label)
                        
                        # 扩充每个object的描述词汇
                        curr_img_descriptions.append(f"a {label}")
                        
                        # 通用描述词（增加更多变体）
                        label_lower = label.lower()
                        extra_descriptions = []
                        if 'hand' in label_lower or 'arm' in label_lower or 'finger' in label_lower:
                            extra_descriptions = ["hand", "human hand", "arms", "fingers", f"the {label}"]
                        elif 'person' in label_lower or 'human' in label_lower or 'body' in label_lower:
                            extra_descriptions = ["human", "person", "human body", "individual", f"the {label}"]
                        else:
                            # 对于其他物体，添加更多描述
                            extra_descriptions = ["object", f"the {label}", f"{label}", "item", "thing"]
                        
                        # 随机选择2-4个额外描述词（增加数量）
                        if extra_descriptions:
                            num_extra = min(random.randint(2, 4), len(extra_descriptions))
                            selected_extra = random.sample(extra_descriptions, num_extra)
                            curr_img_descriptions.extend(selected_extra)
                
                # 将所有描述组合成一个rich text prompt
                if curr_img_descriptions:
                    # 构建形式："an image with hand, human hand, arms, person, object"
                    combined_text = "an image with " + ", ".join(curr_img_descriptions)
                    
                    toks = self.text_visual_projector.text_tokenizer(
                        [combined_text], 
                        padding=True, 
                        truncation=True,
                        max_length=77,
                        return_tensors='pt'
                    )
                    
                    # 调试：打印text prompt和token数量
                    if _os.environ.get("MULTISCALE_DEBUG", ""):
                        num_tokens = toks['input_ids'].shape[1]
                        logger.info(f"[Multi-Token Text] '{combined_text}' → {num_tokens} tokens")
                        logger.info(f"[Token IDs] {toks['input_ids'][0].tolist()}")
                    
                    toks = {k: v.to(device=intermediate_features.device) for k, v in toks.items()} 
                    result = self.text_visual_projector(
                        current_img_features.unsqueeze(0), toks  # 使用当前图像特征
                    )
                    all_similarities.append(result['similarities'])
                    
                    # 保存第一个图像的attention信息用于可视化
                    if img_idx == 0 and saved_attention_weights is None:
                        saved_attention_weights = result.get('attention_weights', None)
                        saved_text_prompts = result.get('text_prompts', None)
                        saved_img_start_idx = start_idx
                
                # 下一个图像
                start_idx = end_idx

            out = torch.cat(all_similarities, dim=1) if all_similarities else None
            
            # Register gradient hook on CLIP similarity logits (only in debug mode)
            if debug_ms and self.training and out is not None and out.requires_grad:
                def capture_clip_grad(grad):
                    if grad is not None:
                        self._grad_stats['clip_logits'] = {
                            'mean': grad.mean().item(),
                            'std': grad.std().item(),
                            'max': grad.max().item(),
                            'min': grad.min().item(),
                            'norm': grad.norm().item()
                        }
                    return grad
                out.register_hook(capture_clip_grad)
            
            # Keep on GPU, no conversion unless debug
            sims_prob_raw = out.squeeze(0).squeeze(-1) if torch.is_tensor(out) else None  # Keep as logits
            sims_prob = sims_prob_raw.sigmoid() if sims_prob_raw is not None else None
            mask_target = _build_lr_mask(all_bboxes, stored_lr_grid_thw) # 多图: flattened 
            
            # 可视化attention map（包含GT区域分析）- only in debug mode
            if debug_ms and saved_attention_weights is not None and mask_target is not None:
                # 提取第一个图像的mask - avoid .tolist() for better performance  
                t, H, W = int(stored_lr_grid_thw[0, 0]), int(stored_lr_grid_thw[0, 1]), int(stored_lr_grid_thw[0, 2])
                first_img_tokens = t * H * W
                first_img_mask = mask_target[saved_img_start_idx:saved_img_start_idx + first_img_tokens]
                
                self.text_visual_projector._visualize_attention(
                    saved_attention_weights, 
                    saved_text_prompts,
                    first_img_mask
                )
        elif text_prompt is not None:
            # CLIP loss disabled, but still build mask_target for confidence loss
            all_bboxes = []
            for prompt in text_prompt:
                body = re.sub(r"^```\w*\n|```\s*$", "", str(prompt).strip(), flags=re.S)
                objects = json.loads(body) if body else []
                for obj in objects:
                    bbox = obj.get("bbox_2d", None)
                    if bbox is not None:
                        all_bboxes.append(bbox)
            
            if all_bboxes:
                mask_target = _build_lr_mask(all_bboxes, stored_lr_grid_thw)

        # [2026-07-26] Query-conditioned CLIP branch. Runs instead of the label-
        # conditioned block above whenever the text source is 'query' (the default
        # as soon as MTS_CLIP_ROUTING=1). Available at inference because it needs
        # only the user instruction, never `labels`. When ENABLE_CLIP_LOSS=1 the
        # loss block below then supervises *these* logits against mask_target, so
        # train and test condition the selector on the same signal.
        if (clip_routing or enable_clip_loss) and clip_text_source == "query":
            sims_prob_raw = self._query_conditioned_sims(
                intermediate_features, stored_lr_grid_thw, instruction
            )

        # [FIX 2026-07-23] this loss block used to be nested inside the
        # `elif text_prompt is not None:` branch, so ENABLE_CLIP_LOSS=1 took the
        # first branch and left `loss_total` unassigned (UnboundLocalError).
        # Backup of the old layout: multiscale_processor_fast_backup_0723.py
        # 3) Auxiliary losses using Focal Loss for class imbalance
        if mask_target is not None:
            if debug_ms:
                # Visualize features with mask overlay (only in debug mode)
                vis_counter = getattr(self, '_vis_counter', 0)
                if vis_counter < 20:  # Only save first 20 batches to avoid clutter
                  try:
                    scale_layer_num = getattr(self, 'scale_layer', 'unknown')
                    debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                    save_path = f'{debug_dir}/features_batch{vis_counter}_layer{scale_layer_num}.png'
                    
                    # 直接从文件读取已保存的图片（避免跨进程实例问题）
                    first_img_data = None
                    img_base_path = f'{debug_dir}/input_images/img{vis_counter}_base'
                    # 找到匹配的base图片文件
                    matching_files = glob.glob(f'{img_base_path}_*.png')
                    if matching_files:
                        base_img_path = matching_files[0]
                        base_img = Image.open(base_img_path)
                        size_match = re.search(r'_(\d+)x(\d+)\.png$', base_img_path)
                        if size_match:
                            base_w, base_h = int(size_match.group(1)), int(size_match.group(2))
                            first_img_data = {
                                'base': base_img,
                                'base_size': (base_w, base_h),
                                'orig_size': (base_w, base_h),  # 简化处理
                            }
                            print(f"[Debug] Loaded image from file: {base_img_path}")
                    
                    # Debug logging (use print to ensure output)
                    print(f"\n{'='*60}")
                    print(f"[Vis Debug - Batch {vis_counter}]")
                    print(f"  first_img_data: {first_img_data is not None}")
                    print(f"  first_img_bboxes: {len(first_img_bboxes) if first_img_bboxes else 0}")
                    print(f"  first_img_labels: {len(first_img_labels) if first_img_labels else 0}")
                    if _IMAGE_PROCESSOR:
                        print(f"  _debug_image_data length: {len(getattr(_IMAGE_PROCESSOR, '_debug_image_data', []))}")
                    print(f"{'='*60}\n")
                    
                    # 绘制输入图片+bbox+mask
                    print(f"[Check] Attempting to visualize: img_data={first_img_data is not None}, "
                          f"bboxes={len(first_img_bboxes) if first_img_bboxes else 0}, "
                          f"mask={mask_target is not None}")
                    
                    if first_img_data is not None and first_img_bboxes and mask_target is not None:
                        try:
                            
                            # 获取第一个图像的mask
                            t, H, W = map(int, stored_lr_grid_thw[0].tolist())
                            first_img_tokens = t * H * W
                            first_img_mask = mask_target[:first_img_tokens].detach().cpu().float().numpy()
                            mask_2d = first_img_mask[:H*W].reshape(H, W)
                            
                            mask_sum = mask_2d.sum()
                            print(f"[Vis] Grid: {H}x{W}, Mask positive: {mask_sum:.0f}/{H*W} ({100*mask_sum/(H*W):.1f}%)")
                            
                            # 1. 低分辨率图片 + mask (使用PIL直接绘制)
                            base_img = first_img_data.get('base').copy().convert('RGBA')
                            base_w, base_h = first_img_data.get('base_size', base_img.size)
                            
                            # Upsample mask to base image size
                            mask_upsampled_lr = zoom(mask_2d, (base_h/H, base_w/W), order=0)
                            
                            # Create green mask overlay with transparency
                            mask_overlay_lr = Image.new('RGBA', (base_w, base_h), (0, 0, 0, 0))
                            overlay_arr = np.array(mask_overlay_lr)
                            overlay_arr[:, :, 1] = 255  # Green channel
                            overlay_arr[:, :, 3] = (mask_upsampled_lr * 153).astype(np.uint8)  # Alpha (0.6 * 255 = 153)
                            mask_overlay_lr = Image.fromarray(overlay_arr, 'RGBA')
                            
                            # Composite images
                            result_lr = Image.alpha_composite(base_img, mask_overlay_lr)
                            
                            debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                            save_path_lr = f'{debug_dir}/input_mask_lr_batch{vis_counter}.png'
                            result_lr.save(save_path_lr)
                            print(f"[Saved] LR image + mask → {save_path_lr} (mask sum: {mask_sum:.0f}/{H*W}, overlay alpha: {overlay_arr[:,:,3].max()})")
                            
                            # 2. 高分辨率图片 + mask (使用PIL直接绘制)
                            debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                            hr_img_path = glob.glob(f'{debug_dir}/input_images/img{vis_counter}_original_hr*.png')
                            if hr_img_path:
                                hr_img = Image.open(hr_img_path[0]).convert('RGBA')
                                hr_w, hr_h = hr_img.size
                                
                                # Upsample mask to HR image size
                                mask_upsampled_hr = zoom(mask_2d, (hr_h/H, hr_w/W), order=0)
                                
                                # Create green mask overlay with transparency
                                mask_overlay_hr = Image.new('RGBA', (hr_w, hr_h), (0, 0, 0, 0))
                                overlay_arr_hr = np.array(mask_overlay_hr)
                                overlay_arr_hr[:, :, 1] = 255  # Green channel
                                overlay_arr_hr[:, :, 3] = (mask_upsampled_hr * 153).astype(np.uint8)  # Alpha (0.6 * 255 = 153)
                                mask_overlay_hr = Image.fromarray(overlay_arr_hr, 'RGBA')
                                
                                # Composite images
                                result_hr = Image.alpha_composite(hr_img, mask_overlay_hr)
                                
                                save_path_hr = f'{debug_dir}/input_mask_hr_batch{vis_counter}.png'
                                result_hr.save(save_path_hr)
                                print(f"[Saved] HR image + mask → {save_path_hr} (mask sum: {mask_sum:.0f}/{H*W}, overlay alpha: {overlay_arr_hr[:,:,3].max()}, nonzero: {(overlay_arr_hr[:,:,3]>0).sum()})")
                            
                        except Exception as e:
                            print(f"[Error] Failed to visualize input+mask: {e}")
                            traceback.print_exc()
                    
                    # 保存2D confidence probability图
                    try:
                        t, H, W = map(int, stored_lr_grid_thw[0].tolist())
                        conf_2d = scale_map[:H*W, 0].float().reshape(H, W).detach().cpu().numpy()
                        
                        fig, ax = plt.subplots(1, 1, figsize=(W*0.8, H*0.8))
                        im = ax.imshow(conf_2d, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
                        
                        # 绘制灰色网格线
                        for i in range(H + 1):
                            ax.axhline(y=i - 0.5, color='gray', linewidth=1, alpha=0.5)
                        for j in range(W + 1):
                            ax.axvline(x=j - 0.5, color='gray', linewidth=1, alpha=0.5)
                        
                        # 在每个格子中心写入概率值（白色大号字体）
                        for i in range(H):
                            for j in range(W):
                                text = ax.text(j, i, f'{conf_2d[i, j]:.2f}',
                                             ha='center', va='center',
                                             color='white', fontsize=14, fontweight='bold')
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.axis('off')
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                        
                        debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                        save_path_conf = f'{debug_dir}/confidence_2d_batch{vis_counter}.png'
                        plt.savefig(save_path_conf, dpi=150, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"[Saved] 2D confidence → {save_path_conf}")
                    except Exception as e:
                        print(f"[Error] Failed to save 2D confidence: {e}")
                    
                    visualize_features_with_mask(
                        intermediate_features, 
                        mask_target, 
                        stored_lr_grid_thw,
                        save_path=save_path,
                        title=f'Input Features (scale_layer={scale_layer_num})',
                        batch_idx=0,
                        image_data=first_img_data,
                        bboxes=first_img_bboxes if first_img_bboxes else None,
                        labels=first_img_labels if first_img_labels else None
                    )
                    self._vis_counter = vis_counter + 1
                  except Exception as _vis_err:
                    print(f"[Debug Vis] Skipped visualization: {_vis_err}")
                    self._vis_counter = vis_counter + 1

            def contrastive_focal_loss(logits, targets, temperature=0.1, margin=1.0, pos_weight=5.0):
                """
                对比学习 + Focal Loss
                拉近ROI样本，推远背景样本
                """
                # 1. 基础BCE
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                
                # 2. 对比项：最大化ROI和BG的差异
                pos_mask = targets > 0.5
                neg_mask = ~pos_mask
                n_pos =  pos_mask.sum()
                
                # 如果没有正样本或负样本，返回0 loss（有梯度）
                if n_pos == 0 or neg_mask.sum() == 0:
                    if debug_ms:
                        print(f"  [Contrastive Loss] No pos or neg samples, skipping (pos={n_pos.item()}, neg={neg_mask.sum().item()})")
                    return logits.sum() * 0.0
                
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    # ROI应该高，BG应该低
                    pos_logits = logits[pos_mask]
                    neg_logits = logits[neg_mask]
                    
                    # Contrastive: 鼓励 pos_logits > neg_logits + margin
                    pos_mean = pos_logits.mean()
                    neg_mean = neg_logits.mean()
                    diff = pos_mean - neg_mean
                    
                    # Hinge loss: max(0, margin - (pos_mean - neg_mean))
                    contrast_loss = F.relu(margin - diff)
                    
                    # 3. 正样本额外权重
                    weighted_bce = torch.zeros_like(bce)
                    weighted_bce[pos_mask] = bce[pos_mask] * pos_weight
                    weighted_bce[neg_mask] = bce[neg_mask] * 1.0
                    
                    bce_loss_val = weighted_bce.mean()
                    
                    # Increase contrast loss weight to encourage stronger separation
                    contrast_weight = float(_os.environ.get("CONTRAST_WEIGHT", "5.0"))  # default 5x stronger
                    total_loss = bce_loss_val + contrast_weight * contrast_loss
                    
                    # Debug output
                    if debug_ms:
                        pos_prob = torch.sigmoid(pos_mean)
                        neg_prob = torch.sigmoid(neg_mean)
                        weighted_contrast = contrast_weight * contrast_loss
                        print(f"  [Contrastive Loss] BCE: {bce_loss_val:.4f}, Contrast: {contrast_loss:.4f}×{contrast_weight:.1f}={weighted_contrast:.4f}")
                        print(f"  [Logits] ROI: {pos_mean:.4f}, BG: {neg_mean:.4f}, Diff: {diff:.4f} (target: >{margin:.1f})")
                        print(f"  [Probs]  ROI: {pos_prob:.4f}, BG: {neg_prob:.4f}")
                    
                    return total_loss

            # Focal Loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
            # Focuses on hard examples, down-weights easy ones
            def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, is_logits=True, reduction='mean', pos_only=False):
                """
                Unified Focal Loss that supports both logits and probabilities
                Args:
                    inputs: [N] either logits (if is_logits=True) or probabilities (if is_logits=False)
                    targets: [N] binary targets (0 or 1)
                    is_logits: whether inputs are logits or probabilities
                    pos_only: if True, only compute loss on positive samples (targets==1), ignore background
                """
                # import pdb; pdb.set_trace()
                if is_logits:
                    # 原始logits版本
                    bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        inputs, targets, reduction='none'
                    )
                    probs = torch.sigmoid(inputs)
                else:
                    # 概率版本
                    probs = torch.clamp(inputs, 1e-7, 1-1e-7)
                    bce_loss = - (targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))
                
                p_t = probs * targets + (1 - probs) * (1 - targets)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                focal_weight = (1 - p_t) ** gamma
                
                # import pdb; pdb.set_trace()
                loss = alpha_t * focal_weight * bce_loss
                
                # 选项：只计算正样本loss
                if pos_only:
                    pos_mask = targets > 0.5
                    if pos_mask.sum() == 0:
                        # 返回一个有梯度的零loss（通过inputs计算得到）
                        # Use sum() * 0 instead of * 0.0 to preserve gradient flow
                        return inputs.sum() * 0.0
                    loss = loss[pos_mask]

                if debug_ms:
                    if targets.sum() > 0:
                        roi_probs = inputs[targets == 1].mean().sigmoid()
                        bg_probs = inputs[targets == 0].mean().sigmoid()
                        print(f"  ROI prob: {roi_probs:.4f}, BG prob: {bg_probs:.4f}")
                        print(f"  Discrimination: {(roi_probs - bg_probs).abs():.6f}")
                        if pos_only:
                            print(f"  [POS_ONLY MODE] Only computing loss on {pos_mask.sum().item()} positive samples")

                if reduction == 'mean':
                    if pos_only:
                        # pos_only模式下已经过滤了，直接mean
                        return loss.mean()
                    else:
                        # 原来的归一化方式
                        num_pos = targets.sum().clamp(min=1.0)
                        return loss.sum() / num_pos
                else:
                    return loss.sum()

            def dice_loss(logits, targets, smooth=1.0, squared=False, class_weights=None):
                """
                Pure Dice Loss for binary segmentation
                
                Dice Coefficient: DC = (2 * |X ∩ Y|) / (|X| + |Y|)
                Dice Loss: 1 - DC
                
                Args:
                    logits: [N] raw logits (before sigmoid)
                    targets: [N] binary targets (0 or 1)
                    smooth: smoothing constant to avoid division by zero
                    squared: if True, use squared denominator (more stable gradients)
                    class_weights: [2] weights for [negative_class, positive_class]
                
                Returns:
                    dice_loss: scalar loss value
                """
                probs = torch.sigmoid(logits)
                
                # Calculate intersection and cardinalities
                intersection = (probs * targets).sum()
                
                if squared:
                    # Squared Dice: denominator uses squared terms
                    pred_sum = (probs ** 2).sum()
                    target_sum = (targets ** 2).sum()
                else:
                    # Standard Dice
                    pred_sum = probs.sum()
                    target_sum = targets.sum()
                
                # Dice coefficient
                dice_coef = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
                
                # Dice loss
                loss = 1 - dice_coef
                
                # Apply class weighting if provided
                if class_weights is not None:
                    # Calculate per-class dice
                    pos_mask = targets > 0.5
                    neg_mask = ~pos_mask
                    
                    # Positive class dice
                    if pos_mask.sum() > 0:
                        pos_inter = (probs[pos_mask] * targets[pos_mask]).sum()
                        pos_pred = probs[pos_mask].sum()
                        pos_target = targets[pos_mask].sum()
                        pos_dice = (2. * pos_inter + smooth) / (pos_pred + pos_target + smooth)
                    else:
                        # Use zeros_like pattern instead of explicit device specification
                        pos_dice = torch.zeros_like(logits[0:1], dtype=torch.float32)[0]
                    
                    # Negative class dice
                    if neg_mask.sum() > 0:
                        neg_probs = 1 - probs[neg_mask]
                        neg_targets = 1 - targets[neg_mask]
                        neg_inter = (neg_probs * neg_targets).sum()
                        neg_pred = neg_probs.sum()
                        neg_target = neg_targets.sum()
                        neg_dice = (2. * neg_inter + smooth) / (neg_pred + neg_target + smooth)
                    else:
                        # Use zeros_like pattern instead of explicit device specification
                        neg_dice = torch.zeros_like(logits[0:1], dtype=torch.float32)[0]
                    
                    # Weighted combination
                    weighted_loss = class_weights[0] * (1 - neg_dice) + class_weights[1] * (1 - pos_dice)
                    
                    if debug_ms:
                        print(f"  [Dice Loss] Pos: {pos_dice:.4f}, Neg: {neg_dice:.4f}, "
                              f"Weighted: {weighted_loss:.4f} (weights: {class_weights})")
                    
                    return weighted_loss
                
                if debug_ms:
                    print(f"  [Dice Loss] Dice Coef: {dice_coef:.4f}, Loss: {loss:.4f}")
                
                return loss

            def tversky_loss(logits, targets, alpha=0.7, beta=0.3, smooth=1.0):
                """
                Tversky Loss - generalization of Dice Loss
                
                Tversky Index: TI = TP / (TP + α*FN + β*FP)
                
                When α=β=0.5, reduces to Dice coefficient
                α>β: penalize False Negatives more (improve Recall)
                α<β: penalize False Positives more (improve Precision)
                
                Args:
                    logits: [N] raw logits
                    targets: [N] binary targets
                    alpha: weight for False Negatives (missed ROI)
                    beta: weight for False Positives (false alarms)
                    smooth: smoothing constant
                
                Recommended settings:
                    - For low Recall: α=0.7, β=0.3 (penalize FN more)
                    - For low Precision: α=0.3, β=0.7 (penalize FP more)
                    - Balanced: α=0.5, β=0.5 (equivalent to Dice)
                """
                probs = torch.sigmoid(logits)
                
                # True Positives
                tp = (probs * targets).sum()
                
                # False Negatives (missed ROI: target=1, pred=0)
                fn = (targets * (1 - probs)).sum()
                
                # False Positives (false alarms: target=0, pred=1)
                fp = ((1 - targets) * probs).sum()
                
                # Tversky index
                tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
                
                loss = 1 - tversky_index
                
                if debug_ms:
                    recall_approx = tp / (tp + fn + 1e-7)
                    precision_approx = tp / (tp + fp + 1e-7)
                    print(f"  [Tversky Loss] TI: {tversky_index:.4f}, Loss: {loss:.4f}")
                    print(f"    TP: {tp:.1f}, FN: {fn:.1f}, FP: {fp:.1f}")
                    print(f"    Recall≈{recall_approx:.3f}, Precision≈{precision_approx:.3f}")
                
                return loss

            def combo_loss(logits, targets, alpha=0.5, beta=0.5, smooth=1.0):
                """
                Combo Loss = α*BCE + β*Dice
                
                Combines classification (BCE) and overlap (Dice) objectives
                
                Args:
                    alpha: weight for BCE
                    beta: weight for Dice
                """
                # BCE component
                bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
                
                # Dice component
                probs = torch.sigmoid(logits)
                intersection = (probs * targets).sum()
                union = probs.sum() + targets.sum()
                dice = (2. * intersection + smooth) / (union + smooth)
                dice_loss_val = 1 - dice
                
                # Combine
                loss = alpha * bce + beta * dice_loss_val
                
                if debug_ms:
                    print(f"  [Combo Loss] BCE: {bce:.4f}, Dice: {dice_loss_val:.4f}, "
                          f"Total: {loss:.4f} (α={alpha}, β={beta})")
                
                return loss

            def dice_bce_loss(logits, targets, smooth=1.0, dice_weight=0.5):
                """
                Dice Loss + BCE (legacy wrapper, use combo_loss instead)
                常用于分割任务，对不平衡数据有效
                """
                return combo_loss(logits, targets, 
                                 alpha=1-dice_weight, 
                                 beta=dice_weight, 
                                 smooth=smooth)
            
            # import pdb; pdb.set_trace()
            # 降低gamma以增强梯度信号 (2.0→1.0 或 0.5)
            # gamma=2.0: 标准Focal Loss，压制容易样本
            # gamma=1.0: 中等聚焦
            # gamma=0.5: 轻度聚焦，更强梯度
            focal_gamma = float(_os.environ.get("FOCAL_GAMMA", "2.0"))
            focal_alpha = float(_os.environ.get("FOCAL_ALPHA", "0.75"))  # 0.75 for imbalanced data (20% pos, 80% neg)
            # loss_conf = focal_loss(scale_map_raw.squeeze(-1), mask_target, alpha=focal_alpha, gamma=focal_gamma, is_logits=True, pos_only=False)

            # Use contrastive focal loss to explicitly separate ROI and BG
            margin = float(_os.environ.get("CONTRAST_MARGIN", "2.0"))  # ROI应该比BG高至少2.0 logits
            pos_weight = float(_os.environ.get("POS_WEIGHT", "5.0"))  # 正样本权重
            
            loss_conf = contrastive_focal_loss(
                scale_map_raw.squeeze(-1), 
                mask_target, 
                margin=margin, 
                pos_weight=pos_weight
            )
        
            
            # loss_clip = focal_loss(sims_prob_raw, mask_target, alpha=focal_alpha, gamma=focal_gamma, is_logits=True, pos_only=False)
            
            # CLIP loss: only compute if ENABLE_CLIP_LOSS=1
            loss_clip = None
            enable_clip_loss = bool(_os.environ.get("ENABLE_CLIP_LOSS", ""))

            # sims_prob_raw is None when the query-conditioned branch got no usable
            # instruction; skip the loss for that batch instead of crashing.
            if enable_clip_loss and sims_prob_raw is not None:
                # Choose loss function via environment variable
                loss_type = _os.environ.get("CLIP_LOSS_TYPE", "dice")  # Options: dice, tversky, combo, contrastive, focal
                
                if loss_type == "dice":
                    loss_clip = dice_loss(sims_prob_raw.squeeze(-1), mask_target, smooth=1.0)
                elif loss_type == "tversky":
                    alpha = float(_os.environ.get("TVERSKY_ALPHA", "0.7"))
                    beta = float(_os.environ.get("TVERSKY_BETA", "0.3"))
                    loss_clip = tversky_loss(sims_prob_raw.squeeze(-1), mask_target, alpha=alpha, beta=beta)
                elif loss_type == "combo":
                    alpha_combo = float(_os.environ.get("COMBO_ALPHA", "0.5"))
                    beta_combo = float(_os.environ.get("COMBO_BETA", "0.5"))
                    loss_clip = combo_loss(sims_prob_raw.squeeze(-1), mask_target, alpha=alpha_combo, beta=beta_combo)
                elif loss_type == "contrastive":
                    loss_clip = contrastive_focal_loss(
                        sims_prob_raw.squeeze(-1), 
                        mask_target, 
                        margin=margin, 
                        pos_weight=pos_weight
                    )
                elif loss_type == "focal":
                    loss_clip = focal_loss(sims_prob_raw.squeeze(-1), mask_target, 
                                          alpha=focal_alpha, gamma=focal_gamma, is_logits=True, pos_only=False)
                else:
                    loss_clip = dice_loss(sims_prob_raw.squeeze(-1), mask_target, smooth=1.0) 

            loss_total = loss_conf + (loss_clip if isinstance(loss_clip, torch.Tensor) else 0.0)
            
            if debug_ms:
                # Safe conversion to float for printing (only in debug mode to avoid GPU sync)
                conf_val = loss_conf.item() if isinstance(loss_conf, torch.Tensor) else float(loss_conf)
                clip_str = f"{loss_clip.item():.4f}" if isinstance(loss_clip, torch.Tensor) else "N/A"
                total_val = loss_total.item() if isinstance(loss_total, torch.Tensor) else float(loss_total)
                num_pos = (mask_target > 0.5).sum().item()
                print(f"[MTS Loss] conf={conf_val:.4f}, clip={clip_str}, total={total_val:.4f}, target_pos_ratio={num_pos}/{mask_target.numel()} ({100*num_pos/mask_target.numel():.1f}%)")
                
                # Store stats for optional inspection (only in debug mode)
                setattr(self, "_last_supervision_losses", {
                    'loss_conf': float(loss_conf.detach().cpu()) if isinstance(loss_conf, torch.Tensor) else float(loss_conf),
                    'loss_clip': float(loss_clip.detach().cpu()) if isinstance(loss_clip, torch.Tensor) else None,
                    'loss_total': float(loss_total.detach().cpu()) if isinstance(loss_total, torch.Tensor) else float(loss_total),
                })
        else:
            loss_total = intermediate_features.sum() * 0.0

        # STE (Straight-Through Estimator)
        # Both training and inference: return binary mask {0, 1}
        # Training: gradients flow through soft sigmoid (via STE)
        # Inference: pure hard threshold
        # Force hard=True to always get binary mask (STE ensures gradient flow)
        valid_conf_binary = self.differentiable_threshold(scale_map.squeeze(-1), self.conf_thresh, hard=True)

        # [2026-07-26] The CLIP similarity map used to be hardcoded to zeros here
        # ("DISABLED: Only use Confidence module"), which made need_highres_mask
        # identical to the confidence head in both training and inference -- the
        # clip arm could only ever act through the auxiliary loss reshaping shared
        # weights, never through semantics-aware selection.
        #   mode 'both'  -> variant (ii), query-level: active at train AND test
        #   mode 'train' -> variant  (i), object-level: active while training only,
        #                   because the category label is GT and is the constant
        #                   "object" on the test splits
        #   mode 'off'   -> zeros, byte-identical to the previous behaviour
        _routing_active = clip_routing_mode == "both" or (
            clip_routing_mode == "train" and self.training
        )
        if _routing_active and sims_prob_raw is not None:
            sims_mask_binary = self.differentiable_threshold(
                sims_prob_raw.squeeze(-1).sigmoid(), self.sims_thresh, hard=True
            )
        else:
            sims_mask_binary = torch.zeros_like(valid_conf_binary)


        if debug_ms:
            mode_str = "TRAINING (STE)" if self.training else "EVAL (hard)"
            print(f"[MTS Mode] {mode_str}, temp={self.temperature:.2f}")
            print(f"[MTS Thresh] conf_thresh={self.conf_thresh}")
            print(f"  scale_map range=[{scale_map.min().item():.3f}, {scale_map.max().item():.3f}]")
            print(f"  binary_mask: {valid_conf_binary.sum().item()}/{valid_conf_binary.numel()} ({100*valid_conf_binary.mean().item():.1f}%) selected")

        # Latency profiling: force an EXACT activated ratio by taking the top-k patches
        # by router confidence, per sample. Sweeping conf_thresh cannot hit a target
        # ratio (the tau->ratio map is image dependent), so the latency-vs-ratio curve
        # needs this. Default off; when unset nothing below changes.
        _force_ratio = _os.environ.get("MTS_FORCE_RATIO", "")
        if _force_ratio:
            r = float(_force_ratio)
            assert 0.0 < r <= 1.0, f"MTS_FORCE_RATIO must be in (0, 1], got {r}"
            conf_flat = scale_map.squeeze(-1).detach()
            forced = torch.zeros_like(valid_conf_binary)
            # per-sample so a batch cannot borrow another image's confidence budget
            if stored_lr_grid_thw is not None:
                spans, off = [], 0
                for row in stored_lr_grid_thw.tolist():
                    n = int(row[0]) * int(row[1]) * int(row[2])
                    spans.append((off, off + n))
                    off += n
            else:
                spans = [(0, conf_flat.numel())]
            for lo, hi in spans:
                n = hi - lo
                k = max(1, int(round(r * n)))
                idx = torch.topk(conf_flat[lo:hi], k=min(k, n)).indices
                forced[lo:hi][idx] = 1.0
            valid_conf_binary = forced

        # Ablation: Use GT mask instead of predicted mask
        USE_GT_MASK = _os.environ.get("USE_GT_MASK", "0") == "1"
        if USE_GT_MASK and mask_target is not None:
            need_highres_mask = mask_target
            if debug_ms:
                print(f"[GT MASK MODE] Using ground truth mask for high-res selection")
                print(f"  GT mask: {need_highres_mask.sum().item()}/{need_highres_mask.numel()} ({100*need_highres_mask.mean().item():.1f}%) selected")
        else:
            # Logical OR using max (both are binary {0, 1})
            need_highres_mask = torch.max(valid_conf_binary, sims_mask_binary)

            # [2026-07-28] The OR above is unbounded, and with an uncalibrated
            # similarity head it degenerated to selecting everything. MTS_ROUTING_BUDGET
            # replaces "OR of two thresholds" with "OR of two rankings, capped":
            #   'conf'   -> k = the confidence head's own per-image count, i.e. exactly
            #               the blind baseline's token budget, reallocated
            #   float r  -> k = r * cells
            # Unset -> untouched, so every existing job keeps its behaviour.
            _budget = _os.environ.get("MTS_ROUTING_BUDGET", "").strip().lower()
            if (_budget and _routing_active and sims_prob_raw is not None
                    and stored_lr_grid_thw is not None):
                need_highres_mask = self._budgeted_reallocation(
                    conf_prob=scale_map.squeeze(-1),
                    sims_prob=sims_prob_raw.squeeze(-1).sigmoid(),
                    conf_binary=valid_conf_binary,
                    stored_lr_grid_thw=stored_lr_grid_thw,
                    budget=_budget,
                )


        # Compute and stash routing metrics when SAVE_ROUTING_METRICS=1
        _save_metrics = _os.environ.get("SAVE_ROUTING_METRICS", "0") == "1"
        if (_save_metrics or debug_ms) and mask_target is not None:
            tp = ((need_highres_mask == 1) & (mask_target == 1)).sum().item()
            fp = ((need_highres_mask == 1) & (mask_target == 0)).sum().item()
            fn = ((need_highres_mask == 0) & (mask_target == 1)).sum().item()
            n_total = len(mask_target)
            n_pred = need_highres_mask.sum().item()
            n_true = (mask_target > 0.5).sum().item()
            setattr(self, '_last_routing_metrics', {
                'tp': tp, 'fp': fp, 'fn': fn,
                'n_pred': int(n_pred), 'n_true': int(n_true), 'n_total': int(n_total),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'token_ratio': n_pred / n_total if n_total > 0 else 0.0,
            })

            # [2026-07-23] SAVE_ROUTING_METRICS only *published* the dict; nothing read it
            # during training (sft/trainer.py:139 runs in prediction_step only), so a
            # collapsed router was invisible until eval. Print it periodically instead.
            if _save_metrics:
                self._routing_print_counter = getattr(self, '_routing_print_counter', 0) + 1
                _every = int(_os.environ.get("MTS_METRICS_EVERY", "50"))
                _rank0 = True
                try:
                    import torch.distributed as _dist
                    if _dist.is_available() and _dist.is_initialized():
                        _rank0 = _dist.get_rank() == 0
                except Exception:
                    pass
                if _rank0 and self._routing_print_counter % _every == 0:
                    _iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
                    print(f"[MTS routing] fwd={self._routing_print_counter} "
                          f"IoU={_iou:.4f} P={tp / (tp + fp) if (tp + fp) > 0 else 0.0:.4f} "
                          f"R={tp / (tp + fn) if (tp + fn) > 0 else 0.0:.4f} "
                          f"kept={n_pred / n_total if n_total > 0 else 0.0:.4f} "
                          f"(pred={n_pred}/{n_total}, gt={n_true})", flush=True)

        # Compute accuracy metrics (only in debug mode to avoid GPU sync)
        if debug_ms and mask_target is not None:
            # Similarity metrics (using binary mask)
            tp_sims = ((valid_conf_binary == 1) & (mask_target == 1)).sum().item()
            fp_sims = ((valid_conf_binary == 1) & (mask_target == 0)).sum().item()
            fn_sims = ((valid_conf_binary == 0) & (mask_target == 1)).sum().item()
            
            precision_sims = tp_sims / (tp_sims + fp_sims) if (tp_sims + fp_sims) > 0 else 0
            recall_sims = tp_sims / (tp_sims + fn_sims) if (tp_sims + fn_sims) > 0 else 0
            f1_sims = 2 * precision_sims * recall_sims / (precision_sims + recall_sims) if (precision_sims + recall_sims) > 0 else 0
            
            # Combined mask metrics
            tp = ((need_highres_mask == 1) & (mask_target == 1)).sum().item()
            fp = ((need_highres_mask == 1) & (mask_target == 0)).sum().item()
            fn = ((need_highres_mask == 0) & (mask_target == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            n_pred = need_highres_mask.sum().item()
            n_true = (mask_target > 0.5).sum().item()
            n_total = len(mask_target)
            acc = ((need_highres_mask == mask_target).float().mean().item())
            
            print(f"[MTS Mask] Pred={n_pred}/{n_total}({100*n_pred/n_total:.1f}%), True={n_true}({100*n_true/n_total:.1f}%), Acc={acc:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
            mask_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
            bbox_rate = n_true / n_total if n_total > 0 else 0
            print(f"[MTS Rate] bbox/tokens={bbox_rate:.4f} ({n_true}/{n_total}), Pred-GT_IoU={mask_iou:.4f}")

            if _save_metrics:
                _metrics_file = _os.environ.get("ROUTING_METRICS_FILE", "routing_debug_metrics.jsonl")
                with open(_metrics_file, "a") as _mf:
                    _mf.write(json.dumps({
                        "tp": tp, "fp": fp, "fn": fn,
                        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                        "mask_iou": round(mask_iou, 4), "acc": round(acc, 4),
                        "n_pred": int(n_pred), "n_true": int(n_true), "n_total": int(n_total),
                        "token_ratio": round(n_pred / n_total, 4) if n_total > 0 else 0,
                        "bbox_rate": round(bbox_rate, 4),
                    }) + "\n")
            
            # 高分辨率图片上叠加选择mask (only in debug mode)
            vis_counter = getattr(self, '_vis_counter', 0) - 1  # 使用已递增的counter
            if vis_counter >= 0 and vis_counter < 20:
                try:
                    debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                    hr_img_path = glob.glob(f'{debug_dir}/input_images/img{vis_counter}_original_hr*.png')
                    if hr_img_path:
                        hr_img = Image.open(hr_img_path[0])
                        hr_w, hr_h = hr_img.size
                        
                        t, H, W = map(int, stored_lr_grid_thw[0].tolist())
                        selection_2d = need_highres_mask[:H*W].float().reshape(H, W).detach().cpu().numpy()
                        
                        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                        ax.imshow(hr_img)
                        
                        P_hr = int(self.patch_size / self.high_res_scale)
                        
                        for i in range(0, hr_h, P_hr):
                            ax.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.3)
                        for j in range(0, hr_w, P_hr):
                            ax.axvline(x=j, color='lightgray', linewidth=0.5, alpha=0.3)
                        
                        P_lr = int(self.patch_size)
                        LR_h_px = H * P_lr
                        LR_w_px = W * P_lr
                        scale_x = hr_w / LR_w_px
                        scale_y = hr_h / LR_h_px
                        
                        for h_idx in range(H):
                            for w_idx in range(W):
                                x1_lr = w_idx * P_lr
                                y1_lr = h_idx * P_lr
                                x2_lr = (w_idx + 1) * P_lr
                                y2_lr = (h_idx + 1) * P_lr
                                
                                x1_hr = int(x1_lr * scale_x)
                                y1_hr = int(y1_lr * scale_y)
                                x2_hr = int(x2_lr * scale_x)
                                y2_hr = int(y2_lr * scale_y)
                                
                                if selection_2d[h_idx, w_idx] > 0.5:
                                    color = np.array([1.0, 1.0, 0.0, 0.5])  # Yellow
                                else:
                                    color = np.array([1.0, 0.8, 0.6, 0.3])  # Light orange
                                
                                overlay = np.zeros((y2_hr - y1_hr, x2_hr - x1_hr, 4))
                                overlay[:, :] = color
                                ax.imshow(overlay, extent=[x1_hr, x2_hr, y2_hr, y1_hr], aspect='auto')
                        
                        ax.axis('off')
                        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
                        
                        save_path_hr = f'{debug_dir}/hr_selection_batch{vis_counter}.png'
                        plt.savefig(save_path_hr, dpi=150, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        print(f"[Saved] HR selection overlay → {save_path_hr}")
                except Exception as e:
                    print(f"[Error] Failed to save HR selection overlay: {e}")
                    traceback.print_exc()
            
            # 保存原始数据到npz文件 (only in debug mode)
            vis_counter = getattr(self, '_vis_counter', 0) - 1
            if vis_counter >= 0 and vis_counter < 20:
                try:
                    t, H, W = map(int, stored_lr_grid_thw[0].tolist())
                    save_data = {
                        'confidence_2d': scale_map[:H*W, 0].float().reshape(H, W).detach().cpu().numpy(),
                        'highres_selection': need_highres_mask[:H*W].float().reshape(H, W).detach().cpu().numpy(),
                        'gt_mask': mask_target[:H*W].float().reshape(H, W).detach().cpu().numpy() if mask_target is not None else None,
                    }
                    
                    # 添加similarity和text_prompts（如果存在）
                    if sims_prob_raw is not None:
                        save_data['similarity_2d'] = sims_prob[:H*W].float().reshape(H, W).detach().cpu().numpy()
                    
                    if text_prompt is not None and len(text_prompt) > 0:
                        save_data['text_prompt'] = str(text_prompt[0])
                    
                    debug_dir = _os.environ.get("DEBUG_VIS_DIR", "debug_vis")
                    save_path_data = f'{debug_dir}/raw_data_batch{vis_counter}.npz'
                    np.savez_compressed(save_path_data, **save_data)
                    print(f"[Saved] Raw data → {save_path_data}")
                except Exception as e:
                    print(f"[Error] Failed to save raw data: {e}")
                    traceback.print_exc()
        
        # 2D map printing (only in debug mode)
        if debug_ms and stored_lr_grid_thw is not None:
            def _print_2d_map(arr, h, w, decimals=2, symbols=None):
                """Helper to print 2D array in grid format"""
                print(f"  Cols→", " ".join(f"{i:>4d}" for i in range(min(w, 16))))
                for row_idx in range(h):
                    row_str = f"Row {row_idx:2d}:"
                    for col_idx in range(min(w, 16)):
                        val = arr[row_idx, col_idx]
                        if symbols is not None:
                            # Use symbol mapping (e.g., 0→'.', 1→'H')
                            symbol = symbols.get(int(val), '?')
                            row_str += f"   {symbol}"
                        else:
                            # Print numeric value
                            row_str += f" {val:4.{decimals}f}"
                    if w > 16:
                        row_str += f" ... (showing first 16/{w} cols)"
                    print(row_str)
            
            # print first image - optimize tensor access
            batch_size = stored_lr_grid_thw.shape[0]
            t, h, w = int(stored_lr_grid_thw[0, 0]), int(stored_lr_grid_thw[0, 1]), int(stored_lr_grid_thw[0, 2])
            total_tokens = t * h * w
            
            if scale_map.shape[0] >= total_tokens:
                # Reshape to 2D grid (use first temporal slice for simplicity)
                # Convert to float32 first to avoid BFloat16 issues with numpy
                conf_2d = scale_map[:h*w, 0].float().reshape(h, w).detach().cpu().numpy()
                
                print(f"\n{'='*60}")
                if batch_size > 1:
                    print(f"[DEBUG] 2D Visualization - Image 0/{batch_size-1} (LR grid: {h}×{w})")
                else:
                    print(f"[DEBUG] 2D Visualization (LR grid: {h}×{w})")
                print(f"{'='*60}")
                
                # 1. Ground truth mask 
                if mask_target is not None:
                    mask_2d = mask_target[:h*w].float().reshape(h, w).detach().cpu().numpy()
                    print(f"\n[1] Ground Truth Mask (from bbox):")
                    _print_2d_map(mask_2d, h, w, decimals=0)
                
                # 2. Predicted confidence
                print(f"\n[2] Predicted Confidence (threshold={self.conf_thresh}):")
                _print_2d_map(conf_2d, h, w)
                
                # 3. Predicted similarity 
                if sims_prob_raw is not None and sims_prob is not None:
                    sim_2d = sims_prob[:h*w].float().reshape(h, w).detach().cpu().numpy()
                    print(f"\n[3] Predicted CLIP Similarity (threshold={self.sims_thresh}):")
                    _print_2d_map(sim_2d, h, w)
                    
                    # Show which positions exceed threshold
                    above_thresh = (sims_prob[:h*w] > self.sims_thresh).sum().item()
                    print(f"  → Positions above threshold: {above_thresh}/{h*w}")
                
                # 4. Final decision mask (binary)
                decision_2d = need_highres_mask[:h*w].float().reshape(h, w).detach().cpu().numpy()
                print(f"\n[4] Final HR Decision (conf OR similarity):")
                _print_2d_map(decision_2d, h, w, decimals=0, symbols={0: '.', 1: 'H'})
                print(f"  → Total HR positions: {need_highres_mask[:h*w].sum().item():.0f}/{h*w}")
                
                print(f"{'='*60}\n")
                
                # Print summary for all images
                if batch_size > 1:
                    print(f"\n[DEBUG] Multi-Image Summary (total {batch_size} images):")
                    for img_idx in range(batch_size):
                        t_i, h_i, w_i = map(int, stored_lr_grid_thw[img_idx].tolist())
                        print(f"  Image {img_idx}: grid [{t_i}, {h_i}, {w_i}] = {t_i*h_i*w_i} tokens")
                    print(f"  Total tokens: {scale_map.shape[0]}")
                    print()

        low_conf_mask = torch.zeros_like(valid_conf_binary)  # 全部设为0，表示没有低置信度token
        # ======================================================

        # Print gradient statistics (only in debug mode)
        if debug_ms:
            self._grad_print_counter += 1
            # Print every batch (gradients from previous backward pass)
            if self._grad_print_counter % 1 == 0:
                if self._grad_stats:
                    print(f"\n[Batch {self._grad_print_counter}] Selector Gradient Statistics:")
                    self.print_gradient_stats()
                
                # Print confidence prediction statistics
                if hasattr(self, '_last_scale_map') and self._last_scale_map is not None:
                    conf_pred = self._last_scale_map.squeeze(-1)  # Already sigmoid-ed
                    print(f"[Batch {self._grad_print_counter}] Confidence Predictions:")
                    print(f"  Mean: {conf_pred.mean().item():.4f} | Std: {conf_pred.std().item():.4f}")
                    print(f"  Min: {conf_pred.min().item():.4f} | Max: {conf_pred.max().item():.4f}")
                    print(f"  >0.5: {(conf_pred > 0.5).sum().item()}/{conf_pred.numel()} ({100*(conf_pred > 0.5).float().mean().item():.1f}%)")

        return loss_total, need_highres_mask, low_conf_mask