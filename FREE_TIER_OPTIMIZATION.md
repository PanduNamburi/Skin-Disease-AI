# Free Tier Optimization - Lazy PyTorch Loading

## What Was Changed

To optimize for Render's free tier (512MB RAM), PyTorch imports have been made **lazy** - they only load when actually needed for predictions, not at Django startup.

### Before (Heavy Startup):
- PyTorch imported at module level → ~200-300MB memory used immediately
- Models loaded at startup → Additional memory usage
- Worker timeouts and 502 errors

### After (Lightweight Startup):
- PyTorch imports only when prediction is made
- Django starts quickly with minimal memory
- Models load lazily when first prediction is requested
- Much lower memory footprint at startup

## Changes Made

### 1. Lazy Import Function
Added `_lazy_import_torch()` that imports PyTorch only when called:
```python
def _lazy_import_torch():
    """Lazy import PyTorch modules only when needed."""
    global _torch, _torch_nn, _torchvision_transforms, _torchvision_models
    if _torch is None:
        import torch
        import torch.nn as nn
        from torchvision import transforms, models
        # ... cache imports
    return _torch, _torch_nn, _torchvision_transforms, _torchvision_models
```

### 2. Updated All Functions
All functions that use PyTorch now call `_lazy_import_torch()`:
- `build_dl_transform()` - lazy loads transforms
- `get_device()` - lazy loads torch
- `load_dl_model()` - lazy loads models, torch, nn
- `predict_with_dl()` - lazy loads torch, transforms
- `SimpleCNN` - lazy loads nn when instantiated

### 3. Lazy-Loaded Globals
- `DEVICE` → `get_device_cached()` - computed when first needed
- `DL_TRANSFORM` → `get_dl_transform()` - computed when first needed

## Benefits

1. **Faster Startup**: Django starts in seconds instead of timing out
2. **Lower Memory**: Startup uses ~50-100MB instead of 300-400MB
3. **Better Reliability**: Less likely to hit memory limits
4. **Same Functionality**: Predictions work exactly the same, just load on-demand

## Testing

The website should now:
- ✅ Start successfully on Render free tier
- ✅ Load homepage quickly
- ✅ Load models only when making predictions
- ✅ Handle multiple requests without memory issues

## Deployment

After pushing these changes:
1. Render will automatically redeploy
2. Check logs for successful startup
3. Test the website - it should load without 502 errors
4. First prediction may take a few seconds (model loading), subsequent ones are fast

## Memory Usage

- **Startup**: ~50-100MB (Django + dependencies)
- **After first prediction**: ~200-300MB (Django + PyTorch + model)
- **Free tier limit**: 512MB
- **Headroom**: ~200MB for handling requests

## Notes

- Models are cached after first load (same process)
- Each worker loads models independently
- First prediction per worker will be slower (model loading)
- Subsequent predictions are fast (model cached)
