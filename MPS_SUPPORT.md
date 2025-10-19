# Metal Performance Shaders (MPS) GPU Support

## Overview

Your JointBERT codebase now supports Apple's Metal Performance Shaders (MPS) backend for GPU acceleration on Apple Silicon Macs! 🚀

## What Changed

### Files Modified

1. **trainer.py** - Updated device selection to support MPS
2. **predict.py** - Updated device selection to support MPS
3. **utils.py** - Updated seed setting to handle MPS correctly

### Device Priority

The code now follows this priority order for device selection:
1. **MPS (Metal)** - Apple GPU (M1/M2/M3 chips)
2. **CUDA** - NVIDIA GPU
3. **CPU** - Fallback if no GPU available

## Verification

✅ **MPS Support Status**: ENABLED and WORKING
- PyTorch Version: 2.9.0
- MPS Built: ✓
- MPS Available: ✓
- Device Selected: **mps**

You can verify this anytime by running:
```bash
python3 test_mps_support.py
```

## Usage

### Training
When you train your model, it will automatically use the Metal GPU:

```bash
python3 main.py --task atis \
                --model_type bert \
                --model_dir atis_model \
                --do_train --do_eval
```

You'll see this log message:
```
Using device: mps
```

### Prediction
Similarly, predictions will automatically use the Metal GPU:

```bash
python3 predict.py --input_file sample_pred_in.txt \
                   --output_file sample_pred_out.txt \
                   --model_dir atis_model
```

### Disabling GPU (Force CPU)
If you need to run on CPU for any reason:

```bash
# For training
python3 main.py --task atis --model_type bert --model_dir atis_model --do_train --no_cuda

# For prediction
python3 predict.py --input_file sample_pred_in.txt --output_file sample_pred_out.txt --model_dir atis_model --no_cuda
```

## Performance Benefits

With Metal GPU acceleration, you can expect:
- **Significantly faster training** (typically 3-10x faster than CPU)
- **Faster inference/prediction**
- **Better power efficiency**
- **Ability to train larger models**

## Technical Details

### Device Selection Logic
```python
if torch.backends.mps.is_available() and torch.backends.mps.is_built() and not args.no_cuda:
    device = "mps"
elif torch.cuda.is_available() and not args.no_cuda:
    device = "cuda"
else:
    device = "cpu"
```

### Seed Setting
- MPS uses the same seed mechanism as CPU
- The seed is set via `torch.manual_seed(args.seed)`
- This ensures reproducible results across different runs

## Compatibility

- ✅ **macOS with Apple Silicon** (M1/M2/M3) - MPS enabled
- ✅ **Linux/Windows with NVIDIA GPU** - CUDA enabled (unchanged)
- ✅ **Any system without GPU** - CPU fallback (unchanged)

The changes are fully backward compatible and won't affect systems without MPS support.

## Notes

- The `--no_cuda` flag now also disables MPS (it's actually a "no GPU" flag)
- Log messages will clearly indicate which device is being used
- All existing functionality remains unchanged
