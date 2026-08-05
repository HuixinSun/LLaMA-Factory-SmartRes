# Copyright 2024 The Qwen Team and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import PretrainedConfig

# Define rope_config_validation locally to avoid import issues
def rope_config_validation(config):
    """
    Validate the `rope_scaling` configuration.
    """
    if config.rope_scaling is None:
        return

    if not isinstance(config.rope_scaling, dict) or len(config.rope_scaling) != 2:
        raise ValueError(
            "`rope_scaling` must be a dictionary with exactly two fields, `type` and `factor`, "
            f"got {config.rope_scaling}"
        )
    rs = config.rope_scaling
    if rs is None:
        return
    # Accept multimodal RoPE configurations
    rope_type = rs.get("rope_type", None)
    rope_type_alt = rs.get("type", None)
    if rope_type == "mrope" or rope_type_alt == "mrope":
        return
    rope_scaling_type = rope_type_alt
    rope_scaling_factor = rs.get("factor", None)
    if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
        raise ValueError(
            f"`rope_scaling`'s type field must be one of ['linear', 'dynamic', 'mrope'], got {rope_scaling_type}"
        )
    if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")


class Qwen2_5_VLVisionConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=32,
        embed_dim=1280,
        hidden_size=3584,
        hidden_act="gelu",
        initializer_range=0.02,
        intermediate_size=15360,
        num_heads=16,
        num_attention_heads=16,
        num_channels=3,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        **kwargs,
    ):
        self.depth = depth
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        
        # Multiscale attributes
        self.fullatt_block_indexes = kwargs.get('fullatt_block_indexes', [])
        self.use_multi_scale = kwargs.get('use_multi_scale', False)
        self.scale_levels = kwargs.get('scale_levels', 2)
        self.conf_thresh = kwargs.get('conf_thresh', 0.5)
        self.scale_thresh = kwargs.get('scale_thresh', 0.8)
        self.base_resolution = kwargs.get('base_resolution', 224)
        self.scale_layer = kwargs.get('scale_layer', 15)  
        self.max_pixels = kwargs.get('max_pixels', None)
        self.window_size = kwargs.get('window_size', 224)  # Default window size (should be >= spatial_merge_size * patch_size)
        self.out_hidden_size = kwargs.get('out_hidden_size', self.hidden_size)  # Usually same as hidden_size

        super().__init__(**kwargs)


class Qwen2_5_VLTextConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"

    def __init__(
        self,
        vocab_size=152064,
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.rope_scaling = rope_scaling

        if self.rope_scaling is not None:
            rope_config_validation(self)

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class Qwen2_5_VLConfig(PretrainedConfig):
    model_type = "qwen2_5_vl"
    is_composition = True
    sub_configs = {"vision_config": Qwen2_5_VLVisionConfig, "text_config": Qwen2_5_VLTextConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=151655,
        video_token_id=151656,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        super().__init__(**kwargs)


__all__ = ["Qwen2_5_VLConfig", "Qwen2_5_VLTextConfig", "Qwen2_5_VLVisionConfig"] 