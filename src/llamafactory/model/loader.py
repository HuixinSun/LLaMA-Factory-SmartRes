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

import sys
sys.path.insert(0, '/mnt/rdata4_6/huixin/LLaMA-Factory-main')

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


def _get_patched_processor(processor, model_args):
    """Get potentially replaced processor for multi-scale processing."""
    # Multi-scale processing is now handled directly in the model patcher
    return processor


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)
        # Handle potential processor replacement for multi-scale
        processor = _get_patched_processor(processor, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            # Prefer custom Qwen2.5-VL implementation to ensure pixel_frames support and multiscale hooks
            model_type = getattr(config, "model_type", None)
            if getattr(model_args, "use_fastv", False) and model_type == "qwen2_5_vl":
                # FastV baseline: prunes vision tokens INSIDE the LLM at layer k, so the
                # vision tower still runs at full resolution. Mutually exclusive with MTS.
                logger.info_rank0("Loading FastV model for Qwen2.5-VL...")
                _fastv_dir = os.path.abspath(
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../custom_models/qwen2_5_vl_fastv")
                )
                if _fastv_dir not in sys.path:
                    sys.path.insert(0, _fastv_dir)
                from modeling_qwen2_5_vl_fastv import Qwen2_5_VLForConditionalGeneration as _FastVQwen25VL

                logger.info_rank0(f"  FastV model path: {_fastv_dir}")
                if model_args.train_from_scratch:
                    model = _FastVQwen25VL(config)
                else:
                    model = _FastVQwen25VL.from_pretrained(**init_kwargs)
                logger.info_rank0("FastV model loaded successfully")
            elif model_type in ("qwen2_5_vl", "qwen2_vl"):
                try:
                    from custom_models.qwen2_5_vl.modeling_qwen2_5_vl_fast import (
                        Qwen2_5_VLForConditionalGeneration as _CustomQwen25VL,
                    )
                    if model_args.train_from_scratch:
                        model = _CustomQwen25VL.from_config(config)
                    else:
                        model = _CustomQwen25VL.from_pretrained(**init_kwargs)
                except Exception as e:
                    logger.warning_rank0(
                        f"Falling back to HF AutoModel for {model_type} due to custom import error: {e}"
                    )
            if model is None:
                if type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                    load_class = AutoModelForVision2Seq
                elif type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                    load_class = AutoModelForImageTextToText
                elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                    load_class = AutoModelForSeq2SeqLM
                elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                    load_class = AutoModelForTextToWaveform
                else:
                    load_class = AutoModelForCausalLM

                if model_args.train_from_scratch:
                    model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
                else:
                    model = load_class.from_pretrained(**init_kwargs)
                    if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                        model = model.thinker  # use part of Omni model

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        
    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    # Freeze CLIP text encoder in MTS processor after LoRA initialization
    # This ensures CLIP remains frozen even when using additional_target
    if is_trainable:
        freeze_count = 0
        for name, param in model.named_parameters():
            # Match both original_module and modules_to_save paths
            if ('token_multiscale_processor' in name and 
                'text_visual_projector.text_encoder' in name):
                param.requires_grad = False
                freeze_count += 1
        
        if freeze_count > 0:
            logger.info_rank0(f"Frozen {freeze_count} CLIP text encoder parameters in MTS processor")

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    # Always emit a concise selector-trainable summary on rank 0
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        sel_params = [
            (name, param.requires_grad)
            for name, param in model.named_parameters()
            if "visual.token_multiscale_processor" in name
        ]
        logger.info_rank0(f"Selector params (requires_grad): {sel_params}")

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
