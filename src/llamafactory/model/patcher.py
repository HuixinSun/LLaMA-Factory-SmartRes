# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Any

import torch
from peft import PeftModel
from transformers import GenerationMixin, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.modeling_utils import is_fsdp_enabled

from ..extras import logging
from ..extras.misc import infer_optim_dtype
from ..extras.packages import is_transformers_version_greater_than
from .model_utils.attention import configure_attn_implementation, print_attn_implementation
from .model_utils.checkpointing import prepare_model_for_training
from .model_utils.embedding import resize_embedding_layer
from .model_utils.kv_cache import configure_kv_cache
from .model_utils.longlora import configure_longlora
from .model_utils.moe import add_z3_leaf_module, configure_moe
from .model_utils.packing import configure_packing
from .model_utils.quantization import configure_quantization
from .model_utils.rope import configure_rope
from .model_utils.valuehead import prepare_valuehead_model
from .model_utils.visual import autocast_projector_dtype, configure_visual_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedTokenizer, ProcessorMixin
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import ModelArguments


logger = logging.get_logger(__name__)


def patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.model_max_length is not None and tokenizer.model_max_length < model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length  # enlarge the tokenizer max length

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning_rank0("New special tokens have been added, changed `resize_vocab` to True.")


def patch_processor(
    processor: "ProcessorMixin",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
) -> None:
    """Replace the image processor with our MultiScaleImageProcessor and plumb pixel_frames.

    Keeps tokenizer and video settings intact.
    """
    setattr(processor, "tokenizer", tokenizer)
    setattr(processor, "image_max_pixels", model_args.image_max_pixels)
    setattr(processor, "image_min_pixels", model_args.image_min_pixels)
    setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
    setattr(processor, "crop_to_patches", model_args.crop_to_patches)
    setattr(processor, "video_max_pixels", model_args.video_max_pixels)
    setattr(processor, "video_min_pixels", model_args.video_min_pixels)
    setattr(processor, "video_fps", model_args.video_fps)
    setattr(processor, "video_maxlen", model_args.video_maxlen)
    setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)

    # Swap in our MultiScaleImageProcessor so base_resolution + pixel_frames are used
    # First check if the existing processor is already a MultiScaleImageProcessor
    import sys
    custom_models_path = '/mnt/rdata4_6/huixin/LLaMA-Factory-main/custom_models'
    if custom_models_path not in sys.path:
        sys.path.insert(0, custom_models_path)
    from qwen2_5_vl.multiscale_image_processor import MultiScaleImageProcessor  # type: ignore
    
    old_ip = getattr(processor, 'image_processor', None)
    is_already_multiscale = isinstance(old_ip, MultiScaleImageProcessor)
    use_multi_scale_flag = getattr(model_args, 'use_multi_scale', False)
    
    # Set use_multi_scale on processor so mm_plugin.py can read it
    setattr(processor, 'use_multi_scale', use_multi_scale_flag)
    
    # Also store other multi-scale config on processor for mm_plugin.py to use if needed
    if use_multi_scale_flag:
        setattr(processor, 'base_resolution', getattr(model_args, 'base_resolution', 224))
        setattr(processor, 'high_res_scale', getattr(model_args, 'high_res_scale', 2.0))
    
    # If use_multi_scale is False but processor is already MultiScaleImageProcessor,
    # we need to update its use_multi_scale attribute
    if not use_multi_scale_flag and is_already_multiscale:
        setattr(old_ip, 'use_multi_scale', False)
        print(f"[INFO] Disabled multi-scale processing for existing MultiScaleImageProcessor")
    
    # Only create/swap MultiScaleImageProcessor if use_multi_scale is True
    elif use_multi_scale_flag:
        kwargs = {
            'patch_size': getattr(old_ip, 'patch_size', getattr(model_args, 'patch_size', 14)),
            'temporal_patch_size': getattr(old_ip, 'temporal_patch_size', getattr(model_args, 'temporal_patch_size', 2)),
            'merge_size': getattr(old_ip, 'merge_size', getattr(model_args, 'spatial_merge_size', 2)),
            'min_pixels': getattr(old_ip, 'min_pixels', getattr(model_args, 'image_min_pixels', None)),
            'max_pixels': getattr(old_ip, 'max_pixels', getattr(model_args, 'image_max_pixels', None)), # full resolution
            'use_multi_scale': getattr(model_args, 'use_multi_scale', True),
            'scale_levels': getattr(model_args, 'scale_levels', 2),
            'conf_thresh': getattr(model_args, 'conf_thresh', 0.5),
            'scale_thresh': getattr(model_args, 'scale_thresh', 0.8),
            'base_resolution': getattr(model_args, 'base_resolution', 224),
            'high_res_scale': getattr(model_args, 'high_res_scale', 3.0),
        }
        
        
        ms_image_processor = MultiScaleImageProcessor(**kwargs)
        
        ## swap
        setattr(processor, 'image_processor', ms_image_processor)
        
        # Set global image processor for visualization
        from qwen2_5_vl.multiscale_processor_fast import set_global_image_processor
        set_global_image_processor(ms_image_processor)



def patch_config(
    config: "PretrainedConfig",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    init_kwargs: dict[str, Any],
    is_trainable: bool,
) -> None:

    # import pdb; pdb.set_trace()
    if model_args.compute_dtype is None:  # priority: bf16 > fp16 > fp32
        if model_args.infer_dtype != "auto" and not is_trainable:
            model_args.compute_dtype = getattr(torch, model_args.infer_dtype)
        else:
            model_args.compute_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))

    # Configure FastV token pruning for Qwen2.5-VL (baseline comparison, ported from -update)
    if getattr(model_args, "use_fastv", False) and getattr(config, "model_type", None) == "qwen2_5_vl":
        setattr(config, "use_fastv", True)
        setattr(config, "fastv_k", getattr(model_args, "fastv_k", 2))
        setattr(config, "fastv_r", getattr(model_args, "fastv_r", 0.5))
        setattr(config, "fastv_inplace", getattr(model_args, "fastv_inplace", True))
        # FastV params must also live on text_config (read by Qwen2_5_VLTextModel)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_fastv", True)
            setattr(config.text_config, "fastv_k", config.fastv_k)
            setattr(config.text_config, "fastv_r", config.fastv_r)
            setattr(config.text_config, "fastv_inplace", config.fastv_inplace)
        logger.info_rank0(
            f"FastV enabled: k={config.fastv_k}, r={config.fastv_r}, inplace={config.fastv_inplace}"
        )

    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_longlora(config, model_args, is_trainable)
    configure_quantization(config, tokenizer, model_args, init_kwargs)
    configure_moe(config, model_args, is_trainable)
    configure_visual_model(config)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [("fp16", torch.float16), ("bf16", torch.bfloat16), ("fp32", torch.float32)]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)

    if getattr(config, "model_type", None) == "minicpmo":
        setattr(config, "init_audio", True)
        setattr(config, "init_tts", False)

    if getattr(config, "model_type", None) == "kimi_vl" and is_trainable:
        setattr(config.text_config, "topk_method", "greedy")

    architectures = getattr(config, "architectures", None) or []
    if "InternVLChatModel" in architectures:
        raise ValueError(
            "Please download the internvl models in a Hugging Face–compatible format "
            "(for example, https://huggingface.co/OpenGVLab/InternVL3-8B-hf)."
        )

    if "LlavaLlamaForCausalLM" in architectures:
        raise ValueError("Please download llava models with hf-compatible format: https://huggingface.co/llava-hf")

    if getattr(config, "model_type", None) == "internlm3" and not is_transformers_version_greater_than("4.47.1"):
        raise RuntimeError("InternLM3 model requires transformers>=4.47.1, please upgrade it.")

    # deepspeed zero3 is not compatible with low_cpu_mem_usage
    init_kwargs["low_cpu_mem_usage"] = model_args.low_cpu_mem_usage and (not is_deepspeed_zero3_enabled())

    # do not cast data type of the model deepspeed zero3 without qlora
    if not (is_deepspeed_zero3_enabled() and model_args.quantization_bit is None):
        init_kwargs["torch_dtype"] = model_args.compute_dtype

        if init_kwargs["low_cpu_mem_usage"] and not is_fsdp_enabled():  # fsdp does not need device map
            if "device_map" not in init_kwargs and model_args.device_map:
                init_kwargs["device_map"] = model_args.device_map  # device map requires low_cpu_mem_usage=True

            if init_kwargs.get("device_map", None) == "auto":
                init_kwargs["offload_folder"] = model_args.offload_folder


def patch_qwen25vl_multiscale_vision(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """
    Patch Qwen2.5-VL model with multi-scale vision processing capability
    This now works at the image preprocessing level;
    """

    import os

    # Token-order variant: "legacy" (default, byte-identical to the trained checkpoints)
    # or "revised" (2026-07-23 window->native / raster-aligned fix, see
    # custom_models/qwen2_5_vl/ms_forward_revised_window.py)
    _token_order = os.environ.get("MTS_TOKEN_ORDER", "legacy").strip().lower()

    if _token_order == "smartres":
        # Drive the RELEASED smartres package instead of the in-tree forwards, so the
        # release repo's code and its converted checkpoint (whose router lives at
        # `visual.router`) are what actually gets evaluated. Installed before PEFT
        # attaches the adapter, so `modules_to_save=['visual.router']` resolves.
        from smartres import install_smartres

        install_smartres(
            model,
            tau=getattr(model_args, "conf_thresh", 0.5),
            router_layer=getattr(model_args, "scale_layer", 30),
            encode_snap=os.environ.get("MTS_SPARSE_SNAP", "window"),
        )
        print(f"[INFO] SmartRes (released package) tau={getattr(model_args, 'conf_thresh', 0.5)} "
              f"router_layer={getattr(model_args, 'scale_layer', 30)} "
              f"encode_snap={os.environ.get('MTS_SPARSE_SNAP', 'window')}")
        return

    if _token_order in ("revised_dense_vec", "dense_vec", "v5"):
        # Full HR encode (identical to `revised`) with the per-patch Python assembly loop
        # replaced. Should be BIT-identical to `revised`, so no retraining is needed.
        from qwen2_5_vl.ms_forward_revised_window_dense_vec import ms_forward
    elif _token_order in ("revised_sparse_win", "sparse_win", "v6"):
        # revised_sparse_vec with the encode set snapped up to whole 112px attention
        # windows instead of 2x2 merge units, so the 28 window-attention blocks see the
        # neighbourhood they were pretrained on. Same token sequence, closer features,
        # ~78% of HR patches encoded instead of ~50%.
        os.environ["MTS_SPARSE_SNAP"] = "window"
        from qwen2_5_vl.ms_forward_revised_window_sparse_vec import ms_forward
    elif _token_order in ("revised_sparse_vec", "sparse_vec", "v4"):
        # Same output as revised_sparse, but the per-patch Python assembly loop (and its
        # ~8.5k GPU->CPU syncs per image) is replaced by ownership + prefix-sum scatter.
        from qwen2_5_vl.ms_forward_revised_window_sparse_vec import ms_forward
    elif _token_order in ("revised_sparse", "sparse", "v3"):
        # Encodes HR only where the router asked for it (real encoder speedup);
        # NOT numerically equivalent to the dense variants -- needs its own training.
        from qwen2_5_vl.ms_forward_revised_window_sparse import ms_forward
    elif _token_order in ("revised", "native", "v2"):
        from qwen2_5_vl.ms_forward_revised_window import ms_forward
    else:
        from qwen2_5_vl.modeling_qwen2_5_vl_fast import ms_forward
    from qwen2_5_vl.multiscale_processor_fast import MultiScaleTokenProcessor

    # Locate vision tower on model.visual 
    visual = getattr(model, "visual", None)

    use_teva = getattr(model_args, "teva_baseline", False)

    # Init token processor (TEVA baseline or learned MTS router)
    if getattr(visual, "token_multiscale_processor", None) is None:
        vc = visual.config
        if use_teva:
            from teva_baseline.modeling_teva_baseline import TEVATokenProcessor
            visual.token_multiscale_processor = TEVATokenProcessor(
                patch_size=getattr(vc, "patch_size", 14),
                temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
                merge_size=getattr(vc, "spatial_merge_size", 2),
                embed_dim=getattr(vc, "hidden_size", 1280),
                n_patches=256,
                base_resolution=getattr(model_args, "base_resolution", 224),
                high_res_scale=getattr(model_args, "high_res_scale", 2.0),
            )
            print("[INFO] Using TEVA baseline processor (bbox-based selection, no learned router)")
        else:
            visual.token_multiscale_processor = MultiScaleTokenProcessor(
                patch_size=getattr(vc, "patch_size", 14),
                temporal_patch_size=getattr(vc, "temporal_patch_size", 2),
                merge_size=getattr(vc, "spatial_merge_size", 2),
                embed_dim=getattr(vc, "hidden_size", 1280),
                in_channels=getattr(vc, "in_channels", 3),
                # get from model_args
                scale_levels=getattr(model_args, "scale_levels", 2),
                conf_thresh=getattr(model_args, "conf_thresh", 0.5),
                scale_thresh=getattr(model_args, "scale_thresh", 0.8),
                base_resolution=getattr(model_args, "base_resolution", 224),
                high_res_scale=getattr(model_args, "high_res_scale", 2.0),
                scale_layer=getattr(model_args, "scale_layer", 15), 
            )
        
        # Store scale_layer for use in forward pass
        scale_layer = getattr(model_args, "scale_layer", 15)
        

        # A monkey-patch that replaces visual.forward with _visual_forward.
        def _visual_forward(self, x, grid_thw, *args, **kwargs):
            # 保留原始forward接口, 从key word arguments读取 
            # vis
            frames_hr = kwargs.get('pixel_frames_hr', None)
            hr_grid_thw = kwargs.get('hr_grid_thw', None)
            # text
            text_prompt = kwargs.get('text_prompt', None)
            instruction = kwargs.get('instruction', None)

            # multiscale forward
            out = ms_forward(self, x, grid_thw, 
                             frames_hr=frames_hr, 
                             hr_grid_thw=hr_grid_thw, 
                             text_prompt=text_prompt, 
                             instruction=instruction,
                             scale_layer=scale_layer)
    
            # 保持原始输出
            return out

        # 定义任何调用visual.forward(...)都通过这个wrapper
        visual.forward = MethodType(_visual_forward, visual)


def patch_model(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    is_trainable: bool,
    add_valuehead: bool,
) -> None:
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if getattr(model.config, "model_type", None) not in ["minicpmv", "minicpmo"] and "GenerationMixin" not in str(
        model.generate.__func__
    ):
        model.generate = MethodType(GenerationMixin.generate, model)

    if add_valuehead:
        prepare_valuehead_model(model)

    if model_args.resize_vocab:
        resize_embedding_layer(model, tokenizer)

    # Patch Qwen2.5-VL with multi-scale vision processing
    if model_args.use_multi_scale:
        patch_qwen25vl_multiscale_vision(model, model_args)

    if is_trainable:
        if getattr(model.config, "model_type", None) == "gemma3n":
            setattr(model_args, "disable_gradient_checkpointing", True)

        prepare_model_for_training(model, model_args)
        autocast_projector_dtype(model, model_args)
        add_z3_leaf_module(model)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)

    try:
        model.add_model_tags(["llama-factory"])
    except Exception:
        logger.warning_rank0("Cannot properly tag the model.")


def patch_valuehead_model(model: "AutoModelForCausalLMWithValueHead") -> None:
    def tie_weights(self: "AutoModelForCausalLMWithValueHead") -> None:
        if isinstance(self.pretrained_model, PreTrainedModel):
            self.pretrained_model.tie_weights()

    def get_input_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_input_embeddings()

    def get_output_embeddings(self: "AutoModelForCausalLMWithValueHead") -> torch.nn.Module:
        if isinstance(self.pretrained_model, PreTrainedModel):
            return self.pretrained_model.get_output_embeddings()

    def create_or_update_model_card(self: "AutoModelForCausalLMWithValueHead", output_dir: str) -> None:
        if isinstance(self.pretrained_model, PeftModel):
            self.pretrained_model.create_or_update_model_card(output_dir)

    ignore_modules = [name for name, _ in model.named_parameters() if "pretrained_model" in name]
    setattr(model, "_keys_to_ignore_on_save", ignore_modules)
    setattr(model, "tie_weights", MethodType(tie_weights, model))
    setattr(model, "get_input_embeddings", MethodType(get_input_embeddings, model))
    setattr(model, "get_output_embeddings", MethodType(get_output_embeddings, model))
    setattr(model, "create_or_update_model_card", MethodType(create_or_update_model_card, model))
