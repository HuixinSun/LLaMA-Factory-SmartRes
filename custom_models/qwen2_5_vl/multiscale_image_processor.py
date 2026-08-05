"""
Multi-scale Image Processor for Qwen2.5-VL

This module implements multi-scale image processing at the preprocessing level,
before patch tokenization. It replaces the single-scale image processor with
adaptive resolution processing that can handle multiple scales efficiently.
"""

import math
import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from PIL import Image, ImageDraw, ImageFont

from transformers.image_utils import (
    ImageInput,
    PILImageResampling,
    make_flat_list_of_images,
    valid_images,
)
from transformers.utils import TensorType
from transformers.feature_extraction_utils import BatchFeature

# Import the base processor
import sys
import os
from transformers import Qwen2VLImageProcessor

from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    make_list_of_images,
    to_numpy_array,
    validate_preprocess_arguments
)

from transformers.video_utils import VideoInput
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)

# Set up logger for debugging
logger = logging.getLogger(__name__)


def add_size_annotation(image: Image.Image, label_text: str) -> Image.Image:
    """
    Add size annotation text to image (non-destructive copy)
    
    Args:
        image: PIL Image to annotate
        label_text: Text to display (e.g., "1920x1080")
    
    Returns:
        New PIL Image with annotation
    """
    # Create a copy to avoid modifying original
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    
    # Try to load a nice font, fallback to default
    try:
        # Try common font paths
        font_size = max(20, int(min(img_copy.size) * 0.03))  # Adaptive font size
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Position: top-left corner with padding
    padding = 10
    x = padding
    y = padding
    
    # Draw background rectangle
    bg_rect = [x - 5, y - 5, x + text_width + 5, y + text_height + 5]
    draw.rectangle(bg_rect, fill='black', outline='white', width=2)
    
    # Draw text
    draw.text((x, y), label_text, fill='yellow', font=font)
    
    return img_copy


def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_by_token_ratio(
    height: int, 
    width: int, 
    token_ratio: float = 0.5, 
    patch_size: int = 14, 
    merge_size: int = 2, 
    factor: int = 28,
    max_pixels: int = 14 * 14 * 4 * 1280
) -> Tuple[int, int]:
    """
    Resize image to achieve a target token ratio with max_pixels constraint.
    
    This function scales the image dimensions such that the resulting token count
    is approximately token_ratio times the original token count. Since tokens are
    proportional to (dimension²), we scale dimensions by sqrt(token_ratio).
    
    Steps:
    1. Round original dimensions to nearest factor multiple (standard processing)
    2. Apply max_pixels limit if exceeded (same logic as smart_resize)
    3. Scale dimensions by sqrt(token_ratio) to achieve target token count
    4. Round UP to nearest factor multiple to ensure divisibility
    
    Args:
        height: Original image height
        width: Original image width
        token_ratio: Target ratio of tokens (e.g., 0.5 for 50% tokens, 0.1 for 10% tokens)
        patch_size: Size of each patch (default: 14)
        merge_size: Merge size for grouping patches (default: 2)
        factor: Dimensions must be divisible by this (default: 28 = patch_size * merge_size)
        max_pixels: Maximum allowed pixels (default: 14 * 14 * 4 * 1280)
    
    Returns:
        Tuple[int, int]: (new_height, new_width) both divisible by factor
    
    Example:
        >>> # For a 1920x1080 image, get 50% tokens
        >>> h, w = smart_resize_by_token_ratio(1080, 1920, token_ratio=0.5)
        >>> # Returns (784, 1372) - dimensions scaled by sqrt(0.5) ≈ 0.707
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} and width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    
    # Step 1: Round to nearest factor multiple (standard processing)
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    
    # Step 2: Apply max_pixels limit if exceeded (same logic as smart_resize)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    
    # Step 3: Scale by sqrt(token_ratio) to achieve target token count
    scale_factor = math.sqrt(token_ratio)
    
    # Step 4: Apply scaling and round UP to nearest factor multiple
    new_h = math.ceil(h_bar * scale_factor / factor) * factor
    new_w = math.ceil(w_bar * scale_factor / factor) * factor
    
    return new_h, new_w


class MultiScaleImageProcessor(Qwen2VLImageProcessor):
    """
    Multi-scale image processor for Qwen2.5-VL that implements adaptive resolution processing.
    
    This processor analyzes images to determine which regions require high-resolution processing
    and which can be processed at lower resolution, optimizing both quality and computational efficiency.
    """
    
    def __init__(
        self,
        # Standard parameters from parent class
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        # Multi-scale specific parameters
        use_multi_scale: bool = True,
        scale_levels: int = 2,
        conf_thresh: float = 0.5,
        scale_thresh: float = 0.5,
        base_resolution: int = 1920,
        high_res_scale: float = 2.0,

        **kwargs,
    ) -> None:
        # Initialize parent class
        super().__init__(
            do_resize=do_resize,
            size=size,
            resample=resample,
            do_rescale=do_rescale,
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_convert_rgb=do_convert_rgb,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            merge_size=merge_size,
            **kwargs
        )
        
        # Multi-scale parameters
        self.use_multi_scale = use_multi_scale
        self.scale_levels = scale_levels
        self.conf_thresh = conf_thresh
        self.scale_thresh = scale_thresh
        self.base_resolution = base_resolution
        self.high_res_scale = high_res_scale
        
    
    def _process_multi_scale(
        self,
        image: Image.Image,
        do_resize: bool,
        size: Dict[str, int],
        resample: PILImageResampling,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: List[float],
        image_std: List[float],
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        data_format,
        do_convert_rgb: bool,
        input_data_format,
    ) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        """
        preprocess one image with the multi-scale approach.

        Logic:
        - Produce base patches at the base resolution using the parent pipeline.
        - Build per-image grid as [T, H, W] keeping H and W from the base grid.
        - Treat any high-resolution additions as extra temporal slices: only T grows
          (T = T_base + T_hr). This keeps spatial indexing and RoPE valid.
        - Expose processed frames (pre-patchify) via the returned frames tensor for
          token→box cropping by the SAT token processor.
        
        Returns:
        - patches: list[np.ndarray] of flat pixel patches (length equals T×H×W)
        - image_grid_thw: np.ndarray shaped (1, 3) with [T, H, W] as described above
        - frames: np.ndarray shaped (num_frames, C, H, W) of processed frames before patchify
        """
        # Step 1: Process original image as HR (100% token, use max_pixels from config)
        hr_patches, hr_grid_thw = self._preprocess(image, do_resize=do_resize, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size, data_format=data_format or ChannelDimension.FIRST, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format)
        
        # Step 2: Calculate LR size using token_ratio logic (convert_resolution_token_ratio.py)
        # Get original image dimensions
        orig_w, orig_h = image.size if isinstance(image, Image.Image) else (get_image_size(image, channel_dim=input_data_format)[::-1])
        
        # Calculate h_bar and w_bar (aligned to factor=28)
        factor = patch_size * merge_size  # 14 * 2 = 28
        max_pixels = self.max_pixels or 12845056  # Use max_pixels from config (default to full res)
        h_bar = round(orig_h / factor) * factor
        w_bar = round(orig_w / factor) * factor
        
        # If exceeds max_pixels, scale down
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((orig_h * orig_w) / max_pixels)
            h_bar = math.floor(orig_h / beta / factor) * factor
            w_bar = math.floor(orig_w / beta / factor) * factor
        
        # Use high_res_scale as token_ratio (e.g., 0.5 means 50% tokens)
        token_ratio = self.high_res_scale
        scale_factor = math.sqrt(token_ratio)
        
        # Calculate new LR dimensions
        lr_h = math.ceil(h_bar * scale_factor / factor) * factor
        lr_w = math.ceil(w_bar * scale_factor / factor) * factor
        
        # Ensure minimum size
        lr_h = max(factor, lr_h)
        lr_w = max(factor, lr_w)
        
        # Calculate actual scale for bbox adjustment
        scale_w = lr_w / orig_w
        scale_h = lr_h / orig_h
        
        # Verify token counts
        new_grid_h = lr_h // patch_size
        new_grid_w = lr_w // patch_size
        lr_tokens = (new_grid_h // merge_size) * (new_grid_w // merge_size)
        
        grid_h = h_bar // patch_size
        grid_w = w_bar // patch_size
        hr_tokens = (grid_h // merge_size) * (grid_w // merge_size)
        
        # Step 3: Create and process LR image
        base_image = image.resize((lr_w, lr_h), resample) if isinstance(image, Image.Image) else resize(image, size=(lr_h, lr_w), resample=resample, input_data_format=input_data_format)
        base_patches, base_grid_thw = self._preprocess(base_image, do_resize=False, size=size, resample=resample, do_rescale=do_rescale, rescale_factor=rescale_factor, do_normalize=do_normalize, image_mean=image_mean, image_std=image_std, patch_size=patch_size, temporal_patch_size=temporal_patch_size, merge_size=merge_size, data_format=data_format or ChannelDimension.FIRST, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format)
        
        base_w, base_h = lr_w, lr_h
        
        # resize_ratio for bbox scaling (LR relative to original)
        resize_ratio = np.array([
            float(scale_h),  # sy = Hlow / Horig
            float(scale_w),  # sx = Wlow / Worig
        ], dtype=np.float32)
        
        # import pdb; pdb.set_trace()
        import os as _os
        if _os.environ.get("VIS_PREPROCESS", ""):
            debug_dir = _os.environ.get("VIS_PREPROCESS", "debug_vis")
            if not hasattr(self, '_debug_img_counter'):
                self._debug_img_counter = 0
                _os.makedirs(f'{debug_dir}/input_images', exist_ok=True)
            
            # 只保存前20张图片
            if self._debug_img_counter < 20:
                counter = self._debug_img_counter
                debug_dir = _os.environ.get("VIS_PREPROCESS", "debug_vis")
                
                # Get original dimensions
                orig_w, orig_h = image.size
                
                # Add annotations to images
                hr_annotated = add_size_annotation(image, f"HR: {orig_w}×{orig_h}\nTokens: {hr_tokens}")
                base_annotated = add_size_annotation(base_image, f"LR: {base_w}×{base_h}\nTokens: {lr_tokens}\nRatio: {token_ratio:.2f}")
                
                # 保存带标注的图片
                hr_annotated.save(f'{debug_dir}/input_images/img{counter:03d}_1_original_hr_{orig_w}x{orig_h}.png')
                base_annotated.save(f'{debug_dir}/input_images/img{counter:03d}_2_resized_lr_{base_w}x{base_h}.png')
                
                # Also save scale info as text file
                with open(f'{debug_dir}/input_images/img{counter:03d}_info.txt', 'w') as f:
                    f.write(f"Image {counter} Scale Information\n")
                    f.write(f"{'='*50}\n")
                    f.write(f"Original (HR): {orig_w} × {orig_h} = {orig_w * orig_h:,} pixels\n")
                    f.write(f"Resized (LR):  {base_w} × {base_h} = {base_w * base_h:,} pixels\n")
                    f.write(f"\nToken Ratio: {token_ratio:.4f} (target)\n")
                    f.write(f"HR Tokens: {hr_tokens}\n")
                    f.write(f"LR Tokens: {lr_tokens}\n")
                    f.write(f"Actual Token Ratio: {lr_tokens / hr_tokens:.4f}\n")
                    f.write(f"\nScale Factors:\n")
                    f.write(f"  scale_w: {scale_w:.4f}\n")
                    f.write(f"  scale_h: {scale_h:.4f}\n")
                    f.write(f"\nResize Ratio (for bbox):\n")
                    f.write(f"  resize_ratio: [{resize_ratio[0]:.4f}, {resize_ratio[1]:.4f}]\n")
                
                self._debug_img_counter += 1
        
        if not hasattr(self, '_debug_image_data'):
            self._debug_image_data = []
        
        if len(self._debug_image_data) >= 20:
            self._debug_image_data.pop(0)  # 删除最旧的
        
        img_data = {
            'original': image.copy(),
            'base': base_image.copy(),
            'hr': image.copy(),
            'base_size': (base_w, base_h),
        }
        self._debug_image_data.append(img_data)

        return base_patches, base_grid_thw, hr_patches, hr_grid_thw, resize_ratio
    
    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        resample: PILImageResampling = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        patch_size: Optional[int] = None,
        temporal_patch_size: Optional[int] = None,
        merge_size: Optional[int] = None,
        do_convert_rgb: Optional[bool] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            vision_info (`List[Dict]`, *optional*):
                Optional list of dictionaries containing additional information about vision inputs.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Size of the image after resizing. `shortest_edge` and `longest_edge` keys must be present.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            patch_size (`int`, *optional*, defaults to `self.patch_size`):
                The spatial patch size of the vision encoder.
            temporal_patch_size (`int`, *optional*, defaults to `self.temporal_patch_size`):
                The temporal patch size of the vision encoder.
            merge_size (`int`, *optional*, defaults to `self.merge_size`):
                The merge size of the vision encoder to llm encoder.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if do_rescale and is_scaled_image(images[0]):
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        if patches.shape[0] % temporal_patch_size != 0:
            repeats = np.repeat(
                patches[-1][np.newaxis], temporal_patch_size - (patches.shape[0] % temporal_patch_size), axis=0
            )
            patches = np.concatenate([patches, repeats], axis=0)
        channel = patches.shape[1]
        grid_t = patches.shape[0] // temporal_patch_size
        grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
        patches = patches.reshape(
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size, # number of vertical merge groups
            merge_size,
            patch_size,
            grid_w // merge_size, # number of horizontal merge groups
            merge_size,
            patch_size,
        )
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flatten_patches = patches.reshape(
            grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size
        )

        return flatten_patches, (grid_t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        videos: Optional[List] = None,
        do_resize: bool = True,
        size: Optional[Dict[str, int]] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1/255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        data_format=None,
        do_convert_rgb: bool = True,
        input_data_format=None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> BatchFeature:
        """
        Override the parent preprocess method to implement multi-scale processing for all images at once.
        """
        # original
        min_pixels = min_pixels if min_pixels is not None else self.min_pixels
        max_pixels = max_pixels if max_pixels is not None else self.max_pixels
        
        if size is not None:
            if "shortest_edge" not in size or "longest_edge" not in size:
                raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
            min_pixels = size["shortest_edge"]
        elif min_pixels is not None and max_pixels is not None:
            # backward compatibility: override size with min_pixels and max_pixels if they are provided
            size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
        else:
            size = {**self.size}

        do_resize = do_resize if do_resize is not None else self.do_resize

        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        patch_size = patch_size if patch_size is not None else self.patch_size
        temporal_patch_size = temporal_patch_size if temporal_patch_size is not None else self.temporal_patch_size
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        if images is not None:
            images = make_flat_list_of_images(images)

        if images is not None and not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
        )
        
        validate_preprocess_arguments(
            rescale_factor=rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
            do_resize=do_resize,
            size=size,
            resample=resample,
        )

        # Set defaults
        size = size or self.size
        image_mean = image_mean or self.image_mean
        image_std = image_std or self.image_std
        
        all_patches: List[np.ndarray] = []
        grid_list: List[np.ndarray] = []
        all_hr_patches: List[np.ndarray] = []
        hr_grid_list: List[np.ndarray] = []
        all_resize_ratio: List[np.ndarray] = []


        if not hasattr(self, '_debug_image_data'):
            self._debug_image_data = []
        self._current_batch_start_idx = len(self._debug_image_data)
        
        # Multi-scale processing: 处理所有图像
        data = {}
        if images is not None:
            # If multi-scale is disabled, use parent class's standard processing
            if not self.use_multi_scale:
                return super().preprocess(
                    images=images,
                    videos=videos,
                    do_resize=do_resize,
                    size=size,
                    min_pixels=min_pixels,
                    max_pixels=max_pixels,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                    return_tensors=return_tensors,
                )
            
            pixel_values = []
            for img in images:
                pixel_values = []
                base_patches, base_grid_thw, hr_patches, hr_grid_thw, resize_ratio = self._process_multi_scale(
                    image=img,
                    do_resize=do_resize,
                    size=size or self.size,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean or self.image_mean,
                    image_std=image_std or self.image_std,
                    patch_size=patch_size,
                    temporal_patch_size=temporal_patch_size,
                    merge_size=merge_size,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )

                all_patches.extend(base_patches) # 
                
                # When conf_thresh=0, all LR tokens will be replaced with HR tokens in modeling,
                # so we need to return HR grid to generate correct number of placeholders.
                # Otherwise tokenizer will generate placeholders based on LR grid (too few),
                # causing prefix length mismatch and sequence length errors.
                # if self.conf_thresh == 0:
                #     grid_list.append(hr_grid_thw)  # Use HR grid for correct placeholder count
                # else:
                grid_list.append(base_grid_thw)  # Use LR grid for mixed LR+HR case
                all_hr_patches.extend(hr_patches)
                hr_grid_list.append(hr_grid_thw)
                all_resize_ratio.append(resize_ratio)
        
            # HR frames are exposed via attributes and picked up by the mm plugin.
            self._last_processed_hr_frames = all_hr_patches
            self._last_processed_hr_grid_thw = hr_grid_list
            self._last_processed_hr_resize_ratio = all_resize_ratio
        
            # 拼接所有图像的patches: [(128,1176), (192,1176)] -> (320, 1176)
            pixel_values = np.concatenate(all_patches, axis=0) if len(all_patches) > 0 else np.array([])
            # 堆叠所有grid: [[1,8,16], [1,12,16]] -> shape (2, 3)
            image_grid_thw = np.array(grid_list, dtype=np.int64)

            # 返回 BatchFeature
            data.update({
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            })

        return BatchFeature(data=data, tensor_type=return_tensors)
    
    def to_dict(self):
        """
        Override to_dict to exclude non-serializable debug data and temporary processing data
        """
        output = super().to_dict()
        # Remove attributes that contain non-JSON-serializable objects
        non_serializable_attrs = [
            '_debug_image_data',  # PIL Image objects
            '_debug_img_counter',  # Not needed in config
            '_current_batch_start_idx',  # Runtime state
            '_last_processed_hr_frames',  # numpy arrays
            '_last_processed_hr_grid_thw',  # numpy arrays
            '_last_processed_hr_resize_ratio',  # lists/tuples
        ]
        for attr in non_serializable_attrs:
            output.pop(attr, None)
        return output