# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


def _extract_assistant_response(text: str) -> str:
    """Everything after the last `assistant\\n` (same rule as save_predictions)."""
    import re

    match = re.search(r"assistant\s*\n(.*)", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _extract_bbox(text: str) -> Optional[list]:
    import re

    match = re.search(r'"bbox_2d":\s*\[([^\]]+)\]', text)
    if match:
        coords = [round(float(x), 2) for x in re.findall(r"\d+\.?\d*", match.group(1))[:4]]
        return coords if len(coords) == 4 else None
    return None


def _calc_iou(box1: Optional[list], box2: Optional[list]) -> float:
    if not box1 or not box2:
        return 0.0
    inter = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter + 1e-10
    return inter / union


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")
        
        # # DEBUG: Check input_ids before generation
        # import pdb; pdb.set_trace()
        # # inputs["input_ids"] - 原始输入token序列
        # # 检查: 1) shape是否正确 2) image_pad token数量 3) 是否有异常token
        
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        
        # Capture per-sample routing metrics from the processor
        if os.environ.get("SAVE_ROUTING_METRICS", "0") == "1":
            visual = getattr(getattr(model, "model", model), "visual", None)
            if visual is None:
                visual = getattr(getattr(getattr(model, "module", model), "model", model), "visual", None)
            if visual is not None:
                mts = getattr(visual, 'token_multiscale_processor', None)
                metrics = getattr(mts, '_last_routing_metrics', None) if mts is not None else None
                if metrics is not None:
                    if not hasattr(self, '_routing_metrics_acc'):
                        self._routing_metrics_acc = []
                    self._routing_metrics_acc.append(metrics)

        if generated_tokens is not None and self.args.predict_with_generate:
            prompt_len = inputs["input_ids"].size(-1)
            if generated_tokens.size(-1) > prompt_len:
                # 只填充原始prompt长度的部分
                generated_tokens[:, :prompt_len] = self.processing_class.pad_token_id
                print(f"[PREDICT] Masked prompt: prompt_len={prompt_len}, generated_len={generated_tokens.size(-1)}")
                # print(self.processing_class.decode(generated_tokens[0][prompt_len:]))
            else:
                generated_tokens[:, :] = self.processing_class.pad_token_id
                print(f"[PREDICT] WARNING: generated_len ({generated_tokens.size(-1)}) <= prompt_len ({prompt_len})")
            generated_tokens = generated_tokens.contiguous()

            # Stream this batch out NOW instead of only at the end of the split.
            if os.environ.get("STREAM_PREDICTIONS", "1") != "0":
                self._stream_predictions(inputs, generated_tokens, labels)

        return loss, generated_tokens, labels

    def _stream_predictions(self, inputs: dict, generated_tokens: "torch.Tensor", labels) -> None:
        r"""Append the current batch's decoded bboxes to `predictions_stream.jsonl`.

        `generated_predictions.jsonl` is only written by save_predictions() after the
        whole split has been generated -- on the 9892/12796-sample splits that is 15-19h
        with nothing on disk, and a walltime kill loses everything. This file is written
        (and flushed) per batch with the same {"prompt", "predict", "label"} keys, so
        eval_scripts/egointenion/eval_with_image_path_v4.py can score it mid-run, plus
        the parsed boxes and per-sample IoU. One file per rank when world_size > 1.

        Best-effort: any error here disables streaming instead of killing the eval.
        Set STREAM_PREDICTIONS=0 to turn it off.
        """
        if getattr(self, "_stream_broken", False):
            return
        try:
            tok = self.processing_class
            pad_id = tok.pad_token_id
            if getattr(self, "_stream_file", None) is None:
                os.makedirs(self.args.output_dir, exist_ok=True)
                name = "predictions_stream.jsonl"
                if self.args.world_size > 1:
                    name = f"predictions_stream_rank{self.args.process_index}.jsonl"
                path = os.path.join(self.args.output_dir, name)
                self._stream_file = open(path, "a", encoding="utf-8")
                self._stream_n = 0
                logger.info_rank0(f"[stream] writing predictions as they are generated to {path}")

            preds = generated_tokens.detach().cpu().numpy()
            prompts = inputs["input_ids"].detach().cpu().numpy()
            labs = labels.detach().cpu().numpy() if labels is not None else None
            for i in range(len(preds)):
                keep = preds[i][preds[i] != pad_id]
                pred_txt = _extract_assistant_response(tok.decode(keep.tolist(), skip_special_tokens=True)) if keep.size else ""
                prompt_txt = tok.decode(prompts[i].tolist(), skip_special_tokens=False)
                if labs is not None:
                    lab = np.where(labs[i] != IGNORE_INDEX, labs[i], pad_id)
                    lab_txt = tok.decode(lab[lab != pad_id].tolist(), skip_special_tokens=True)
                else:
                    lab_txt = ""
                pred_box, gt_box = _extract_bbox(pred_txt), _extract_bbox(lab_txt)
                row = {
                    "idx": self._stream_n,
                    "prompt": prompt_txt,
                    "predict": pred_txt,
                    "label": lab_txt,
                    "pred_bbox": pred_box,
                    "gt_bbox": gt_box,
                    "iou": round(_calc_iou(pred_box, gt_box), 4),
                }
                # STREAM_DEBUG=1: also keep the undecoded picture of the row, so a
                # prompt/generation split that lands in the wrong place is visible.
                if os.environ.get("STREAM_DEBUG", "0") == "1" and self._stream_n < 5:
                    row["dbg"] = {
                        "input_ids_len": int(prompts[i].shape[-1]),
                        "generated_len": int(preds[i].shape[-1]),
                        "n_nonpad_in_generated": int(keep.size),
                        "predict_no_extract": tok.decode(keep.tolist(), skip_special_tokens=True),
                        "predict_with_specials": tok.decode(preds[i].tolist(), skip_special_tokens=False)[-3000:],
                    }
                self._stream_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._stream_n += 1
            self._stream_file.flush()
        except Exception as err:
            self._stream_broken = True
            logger.warning_rank0(f"[stream] disabled after error: {err}")

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        # import pdb; pdb.set_trace()
        
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        # Decode each sample individually, then extract assistant response
        import re
        
        def extract_assistant_response(text):
            """Extract only the assistant's response after 'assistant\\n'"""
            # Find the last 'assistant\n' and take everything after it
            match = re.search(r'assistant\s*\n(.*)', text, re.DOTALL)
            if match:
                return match.group(1).strip()
            return text  # fallback to original if no match
        
        # Decode predictions individually (avoid batch padding issues with MTS)
        decoded_preds = []
        for i in range(len(preds)):
            # Remove pad tokens for this sample
            non_pad_indices = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(non_pad_indices) > 0:
                sample_tokens = preds[i][non_pad_indices]
            else:
                sample_tokens = np.array([self.processing_class.pad_token_id])
            
            # Decode this sample
            decoded = self.processing_class.decode(sample_tokens, skip_special_tokens=skip_special_tokens)
            # Extract only assistant response
            decoded = extract_assistant_response(decoded)
            decoded_preds.append(decoded)
        
        # import pdb; pdb.set_trace()
        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        def extract_bbox(text):
            match = re.search(r'"bbox_2d":\s*\[([^\]]+)\]', text)
            if match:
                coords = [round(float(x), 2) for x in re.findall(r'\d+\.?\d*', match.group(1))[:4]]
                return coords if len(coords) == 4 else None
            return None
        
        def calc_iou(box1, box2):
            if not box1 or not box2: return 0.0
            inter = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0])) * max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter + 1e-10
            return inter / union

        # Write full predictions
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                pred_box, label_box = extract_bbox(pred), extract_bbox(label)
                iou = calc_iou(pred_box, label_box)
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
        
        # Write bbox metrics only
        bbox_metrics_file = output_prediction_file.replace("generated_predictions.jsonl", "bbox_metrics.jsonl")
        with open(bbox_metrics_file, "w", encoding="utf-8") as f:
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_box, label_box = extract_bbox(pred), extract_bbox(label)
                iou = calc_iou(pred_box, label_box)
                f.write(json.dumps({"pred_bbox": pred_box, "gt_bbox": label_box, "iou": round(iou, 4)}, ensure_ascii=False) + "\n")

        # Write per-sample routing metrics (computed by the processor itself)
        if hasattr(self, '_routing_metrics_acc') and self._routing_metrics_acc:
            routing_file = output_prediction_file.replace("generated_predictions.jsonl", "routing_metrics.jsonl")

            fg_recalls, fg_precisions, token_ratios = [], [], []
            with open(routing_file, "w", encoding="utf-8") as f:
                for idx, m in enumerate(self._routing_metrics_acc):
                    m["sample_idx"] = idx
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
                    fg_recalls.append(m["recall"])
                    fg_precisions.append(m["precision"])
                    token_ratios.append(m["token_ratio"])

            n = len(fg_recalls)
            valid = [(r, p, t) for r, p, t in zip(fg_recalls, fg_precisions, token_ratios) if t > 0]
            n_valid = len(valid)
            mean_recall = sum(r for r, _, _ in valid) / n_valid if n_valid else 0.0
            mean_precision = sum(p for _, p, _ in valid) / n_valid if n_valid else 0.0
            mean_ratio = sum(t for _, _, t in valid) / n_valid if n_valid else 0.0

            logger.info_rank0(
                f"[Routing Metrics] Total={n}, Valid={n_valid}, "
                f"FG-Recall={mean_recall:.4f}, FG-Precision={mean_precision:.4f}, "
                f"Mean Token Ratio={mean_ratio:.4f}"
            )
            summary_file = output_prediction_file.replace("generated_predictions.jsonl", "routing_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "num_samples": n,
                    "num_valid": n_valid,
                    "mean_fg_recall": round(mean_recall, 4),
                    "mean_fg_precision": round(mean_precision, 4),
                    "mean_token_ratio": round(mean_ratio, 4),
                }, f, indent=2)
