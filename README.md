# LLaMA-Factory for SmartRes

The training and generation loop for [SmartRes](https://github.com/HuixinSun/SmartRes),
forked from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). Included as a
submodule of that repository.

## Usage

Set `use_smartres: true` in a config to enable dynamic resolution routing:

```yaml
use_smartres: true
tau: 0.5             # routing threshold, M = STE(S > tau)
router_layer: 30     # vision block the router reads
encode_snap: window  # encode-set granularity: window | unit
lr_budget: 0.10      # r_LR, low-resolution token budget
hr_budget: 0.50      # r_HR, high-resolution token budget
lambda_route: 1.0    # weight of the routing BCE term
lambda_hinge: 5.0    # weight of the margin regulariser
```

```bash
llamafactory-cli train config.yaml
```

## SmartRes's Modifications

| Path | What it does |
|:--|:--|
| `src/llamafactory/hparams/model_args.py` | the SmartRes parameters above |
| `src/llamafactory/data/mm_plugin.py` | swap in the dual-resolution image processor |
| `src/llamafactory/data/collator.py` | carry `pixel_frames_hr` / `hr_grid_thw` through the batch |
| `src/llamafactory/model/patcher.py` | install the `smartres` package onto the vision tower |
| `custom_models/qwen2_5_vl/multiscale_image_processor.py` | build the low- and high-resolution views |
| `custom_models/qwen2_5_vl/modeling_qwen2_5_vl_fast.py` | splice the variable-length visual span into the prompt |

The visual sequence has a routing-dependent length, so `input_ids`, the attention mask and
`cache_position` are rebuilt per sample.

## Install

```bash
pip install -e . --no-deps
```

Use `--no-deps`. Dependencies are pinned in SmartRes's `env/requirements.txt`; resolving
them here upgrades `transformers` past the version the released checkpoint was trained
with.

## License

Apache 2.0, as upstream.
