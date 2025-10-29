# Batch Image Generator

Gradio app for batch text-to-image with:
- Large live preview
- Streaming batch generation with rolling ZIP
- Character consistency via IP-Adapter XL (2 refs)
- Model presets (FLUX, SDXL, Playground, Juggernaut, RealVis, SD1.5)
- Images per prompt (1–10), styles, resolutions, ratios, steps/guidance, seed

## Quickstart (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
# Choose one torch build:
# - NVIDIA CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# - Windows AMD (DirectML): pip install torch-directml
pip install -r requirements.txt
python entry.py --fast
```

## Quickstart (Linux)
```bash
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
# Choose one torch build:
# - CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# - ROCm: pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -r requirements.txt
python entry.py --fast
```

## Colab
- Upload repo files or clone the repo
- `!pip -q install -r requirements.txt`
- Install your preferred PyTorch build (CUDA/ROCm/DirectML)
- `!python entry.py --share --fast`

## Notes
- Use the Prompts tab to paste or upload a .txt file (one prompt per line).
- Downloads tab updates as images are generated.
- Use model helper notes to pick steps/guidance quickly.