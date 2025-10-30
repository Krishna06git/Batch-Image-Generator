# app.py
import os
import gc
import time
import zipfile
import tempfile
import threading
import argparse
from typing import List, Tuple, Optional, Dict, Any

import torch
from PIL import Image
import gradio as gr
import yaml

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

def select_device_and_dtype():
    try:
        import torch_directml  # optional on Windows AMD
        return torch_directml.device(), torch.float16, True
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return device, torch.bfloat16, False
        except Exception:
            pass
        return device, torch.float16, False
    return device, torch.float32, False

DEVICE, DTYPE, USE_DML = select_device_and_dtype()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

def load_models_yaml(yaml_path: str = "models.yaml"):
    paths_to_try = [yaml_path, os.path.join("app", "config", "models.yaml")]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Support both dict or list style yaml
            models = data.get("models") if isinstance(data, dict) else data
            result = []
            for model in models:
                if isinstance(model, dict) and "name" in model and "path" in model:
                    result.append({"name": model["name"], "path": model["path"], "description": model.get("description", "")})
            return result
    return []

MODEL_YAML_ENTRIES = load_models_yaml()
if not MODEL_YAML_ENTRIES:
    raise RuntimeError("No models found in models.yaml! Please add models with HuggingFace paths.")

MODEL_OPTIONS = [x["name"] for x in MODEL_YAML_ENTRIES]
MODEL_NAME2PATH = {x["name"]: x["path"] for x in MODEL_YAML_ENTRIES}
DEFAULT_MODEL_ID = MODEL_NAME2PATH.get(MODEL_OPTIONS[0])

STYLE_PRESETS = {
    "none": "",
    "realistic": "photorealistic highly detailed face perfect anatomy natural skin texture professional photography sharp focus detailed eyes realistic proportions 8k uhd dslr",
    "anime": "anime style detailed anime artwork vibrant colors clean lines detailed background studio ghibli inspired smooth shading",
    "3d cartoon": "3d render Pixar style soft lighting smooth surfaces vibrant colors professional 3d modeling subsurface scattering",
    "sketch": "pencil sketch detailed line art crosshatching monochrome artistic sketch fine lines shading details",
    "painting": "oil painting classical art style rich colors impasto brush strokes canvas texture detailed",
    "chibi": "chibi style cute and adorable big head small body proportions kawaii bright colors",
    "ghibli": "studio ghibli style whimsical and dreamy soft pastel colors painterly background magical atmosphere",
}
RES_OPTIONS = {"480p": 480, "720p": 720, "1080p": 1080, "2k": 1440, "4k": 2160}
RATIO_OPTIONS = {"1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "16:9": (16, 9), "9:16": (9, 16)}

def round_to_multiple(value: int, multiple: int) -> int:
    return int(round(value / multiple) * multiple)

def compute_size(target_long_side: int, ratio_str: str, base_multiple: int = 8) -> Tuple[int, int]:
    rw, rh = RATIO_OPTIONS[ratio_str]
    if rw >= rh:
        width = target_long_side
        height = int(target_long_side * rh / rw)
    else:
        height = target_long_side
        width = int(target_long_side * rw / rh)
    width = max(base_multiple, round_to_multiple(width, base_multiple))
    height = max(base_multiple, round_to_multiple(height, base_multiple))
    return width, height

def build_styled_prompt(prompt: str, style_key: str) -> str:
    style = STYLE_PRESETS.get(style_key, "").strip()
    return f"{prompt.strip()}, {style}" if style else prompt.strip()

def parse_prompt_file(file_obj) -> List[str]:
    if file_obj is None:
        return []
    try:
        if hasattr(file_obj, "name"):
            with open(file_obj.name, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            if hasattr(file_obj, "read"):
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
            else:
                content = str(file_obj)
    except Exception:
        return []
    lines = [ln.strip() for ln in content.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith("#")]

def append_image_to_zip(zip_path: str, img_path: str):
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(img_path):
            zf.write(img_path, arcname=os.path.basename(img_path))

def create_zip_container() -> str:
    zip_dir = tempfile.mkdtemp(prefix="zip_")
    zip_path = os.path.join(zip_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED):
        pass
    return zip_path

_HEARTBEAT_STOP = False
def _heartbeat():
    while not _HEARTBEAT_STOP:
        print(f"[heartbeat] {time.strftime('%H:%M:%S')}")
        time.sleep(45)

PIPE = None
PIPE_MODEL_ID = None
IP_ADAPTER_READY = False
CHAR_BANK: Dict[str, Dict[str, Any]] = {}

def del_pipe():
    global PIPE, PIPE_MODEL_ID, IP_ADAPTER_READY
    if PIPE is not None:
        del PIPE
    PIPE = None
    PIPE_MODEL_ID = None
    IP_ADAPTER_READY = False
    gc.collect()
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

def load_pipeline(model_id: str):
    global PIPE, PIPE_MODEL_ID, IP_ADAPTER_READY
    if PIPE is not None and PIPE_MODEL_ID == model_id:
        return
    del_pipe()

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True,
    )

    try:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        except Exception:
            pass

    if not USE_DML:
        for fn in [
            getattr(pipe, "enable_xformers_memory_efficient_attention", None),
            getattr(pipe, "enable_sdp_attention", None),
            getattr(pipe, "enable_vae_slicing", None),
            getattr(pipe, "enable_vae_tiling", None),
        ]:
            try:
                if fn:
                    fn()
            except Exception:
                pass

    try:
        pipe.to(DEVICE if USE_DML else DEVICE, dtype=None if USE_DML else DTYPE)
    except Exception:
        pipe.to("cpu")

    global IP_ADAPTER_READY
    IP_ADAPTER_READY = False
    try:
        if isinstance(pipe, StableDiffusionXLPipeline):
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter-plus_sdxl.safetensors",
            )
            IP_ADAPTER_READY = True
    except Exception:
        IP_ADAPTER_READY = False

    PIPE = pipe
    PIPE_MODEL_ID = model_id

def default_steps_and_scale(model_id: str, quality_mode: str) -> Tuple[int, float]:
    mid = model_id.lower()
    if "schnell" in mid or "turbo" in mid:
        return (8 if quality_mode == "High" else 6 if quality_mode == "Balanced" else 4), 2.0 if quality_mode != "Fast" else 1.8
    if "flux" in mid:
        return (28 if quality_mode == "High" else 20 if quality_mode == "Balanced" else 14), (7.5 if quality_mode == "High" else 6.5 if quality_mode == "Balanced" else 5.5)
    if any(k in mid for k in ["realvis", "juggernaut", "stable-diffusion-xl", "playground"]):
        return (35 if quality_mode == "High" else 25 if quality_mode == "Balanced" else 18), (7.5 if quality_mode == "High" else 7.0 if quality_mode == "Balanced" else 6.5)
    # SD1.5 family: runwayml sd-v1-5, realistic vision v4, realistic stock photo v2.0
    if any(k in mid for k in ["sd-v1-5", "stable-diffusion-v1-5", "realistic_vision_v4", "realistic-vision", "realisticstockphoto", "realisticstockphoto-v20"]):
        return (30 if quality_mode == "High" else 25 if quality_mode == "Balanced" else 18), (7.0 if quality_mode != "Fast" else 6.0)
    return (25 if quality_mode != "Fast" else 12), (7.0 if quality_mode != "Fast" else 5.0)

def get_model_info(model_id: str) -> str:
    """Returns helpful info about model's recommended settings"""
    mid = model_id.lower()
    if "schnell" in mid or "turbo" in mid:
        return "⚡ Fast model: 4-8 steps, guidance ~2.0. Best for quick drafts and iterations."
    if "flux" in mid and "dev" in mid:
        return "🎨 Premium quality: 20-30 steps, guidance ~7.0. Best overall quality, slower generation."
    if "flux" in mid and "schnell" in mid:
        return "⚡ Fast FLUX: 4-8 steps, guidance ~2.0. Good quality with speed."
    if "playground" in mid:
        return "🏞️ Playground v2: 18-35 steps, guidance ~7.0. Excellent for aesthetics and general use."
    if "juggernaut" in mid:
        return "💪 Juggernaut XL: 25-35 steps, guidance ~7.0. Powerful, detailed outputs."
    if "realvis" in mid:
        return "📸 RealVis XL: 25-35 steps, guidance ~7.5. Optimized for photorealistic images."
    if "stable-diffusion-xl" in mid and "base" in mid:
        return "🌟 SDXL Base: 25-35 steps, guidance ~7.5. Balanced quality and speed."
    if any(k in mid for k in ["sd-v1-5", "stable-diffusion-v1-5"]):
        return "🔧 SD1.5: 18-30 steps, guidance ~7.0. Classic model, wide community support."
    if "realistic" in mid:
        return "📷 Realistic models: 18-30 steps, guidance ~7.0. Specialized for photorealistic outputs."
    return "⚙️ Default: 12-25 steps, guidance ~7.0. Adjust based on quality/speed needs."

def prepare_character_refs(char_entries: List[Tuple[str, Optional[Image.Image], float]]):
    CHAR_BANK.clear()
    for name, img, strength in char_entries:
        if name and img is not None:
            CHAR_BANK[name.strip()] = {"images": [img], "strength": float(max(0.0, min(1.0, strength)))}

@torch.inference_mode()
def generate_images_stream(
    model_id: str,
    prompts: List[str],
    negative_prompt: str,
    target_res: str,
    ratio: str,
    quality_mode: str,
    steps_override: Optional[int],
    guidance_override: Optional[float],
    seed: Optional[int],
    images_per_prompt: int = 1,
    max_images: Optional[int] = None,
):
    if not prompts:
        yield None, None, None, "No prompts provided."
        return

    try:
        load_pipeline(model_id)
    except Exception as e:
        yield None, None, None, f"Failed to load model: {e}"
        return

    generator = None
    if (not USE_DML) and seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(int(seed))

    width, height = compute_size(RES_OPTIONS[target_res], ratio)
    auto_steps, auto_scale = default_steps_and_scale(model_id, quality_mode)
    steps = int(steps_override) if steps_override and int(steps_override) > 0 else auto_steps
    gscale = float(guidance_override) if guidance_override and float(guidance_override) > 0 else auto_scale

    neg_prompt = negative_prompt or "low quality, worst quality, watermark, logo, text, blurry, jpeg artifacts, bad anatomy, extra limbs, deformed, distorted"

    gallery_images: List[Image.Image] = []
    first_png_path = None
    zip_path = create_zip_container()
    total = len(prompts) if not max_images else min(len(prompts), max_images)
    images_per_prompt = max(1, min(10, int(images_per_prompt)))  # Clamp 1-10
    total_images = total * images_per_prompt

    ip_images, ip_scales = [], []
    if IP_ADAPTER_READY and len(CHAR_BANK) > 0 and isinstance(PIPE, StableDiffusionXLPipeline):
        for _, entry in CHAR_BANK.items():
            ip_images.extend(entry["images"])
            ip_scales.append(entry["strength"])
        ip_images = ip_images[:2]
        ip_scales = (ip_scales[:2] or [0.8])

    generated_count = 0
    for prompt_idx, prompt in enumerate(prompts[:total]):
        for img_idx in range(images_per_prompt):
            try:
                # Use different seeds for variety when generating multiple images per prompt
                current_seed = seed
                if seed is None or seed < 0:
                    # Random seed for each image
                    current_seed = None
                elif images_per_prompt > 1:
                    # Offset seed for variety: base_seed + prompt_idx * 1000 + img_idx
                    current_seed = int(seed) + prompt_idx * 1000 + img_idx

                gen = None
                if (not USE_DML) and current_seed is not None and current_seed >= 0:
                    gen = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(current_seed)

                kwargs = dict(
                    prompt=prompt.strip(),
                    negative_prompt=neg_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=gscale,
                )
                if gen is not None:
                    kwargs["generator"] = gen
                if IP_ADAPTER_READY and len(ip_images) > 0 and isinstance(PIPE, StableDiffusionXLPipeline):
                    kwargs["ip_adapter_image"] = ip_images
                    kwargs["ip_adapter_image_embeds"] = None
                    kwargs["ip_adapter_scale"] = ip_scales

            out = PIPE(**kwargs)
            img = out.images[0]

            img_dir = tempfile.mkdtemp(prefix="gen_")
                out_path = os.path.join(img_dir, f"prompt_{prompt_idx+1:04d}_img_{img_idx+1:02d}.png")
            img.save(out_path, "PNG")
            append_image_to_zip(zip_path, out_path)

            if first_png_path is None:
                first_png_path = out_path

                generated_count += 1
            gallery_images.append(img)
            if len(gallery_images) > 16:
                gallery_images = gallery_images[-16:]

                info = f"Generated {generated_count}/{total_images} | Prompt {prompt_idx+1}/{total} | Image {img_idx+1}/{images_per_prompt} | {width}x{height} | steps={steps} | scale={gscale}"
            yield gallery_images, first_png_path, zip_path, info

            del img, out
                if DEVICE == "cuda" and (generated_count % 8 == 0):
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
                yield gallery_images, first_png_path, zip_path, f"Error on prompt {prompt_idx+1}, image {img_idx+1}: {e}"
            continue

    yield gallery_images, first_png_path, zip_path, f"Done. Generated {generated_count} images from {total} prompts."
    return

def ui_generate_single(
    model_id, prompt, negative_prompt, style_preset, target_res, ratio,
    quality_mode, num_steps, guidance_scale, seed, images_per_prompt,
    char1_name, char1_img, char1_strength,
    char2_name, char2_img, char2_strength,
):
    sprompt = build_styled_prompt(prompt, style_preset)
    prepare_character_refs([
        (char1_name, char1_img, char1_strength or 0.8),
        (char2_name, char2_img, char2_strength or 0.8),
    ])
    stream = generate_images_stream(
        model_id=model_id,
        prompts=[sprompt],
        negative_prompt=negative_prompt,
        target_res=target_res,
        ratio=ratio,
        quality_mode=quality_mode,
        steps_override=int(num_steps) if num_steps else None,
        guidance_override=float(guidance_scale) if guidance_scale else None,
        seed=int(seed) if isinstance(seed, (int, float, str)) and str(seed).isdigit() and int(seed) >= 0 else None,
        images_per_prompt=int(images_per_prompt) if images_per_prompt else 1,
    )
    last = None
    for state in stream:
        last = state
    if not last:
        return None, None, None, "No image"
    return last

def ui_generate_batch_stream(
    model_id, prompt_file, negative_prompt, style_preset, target_res, ratio,
    quality_mode, num_steps, guidance_scale, seed, images_per_prompt,
    char1_name, char1_img, char1_strength,
    char2_name, char2_img, char2_strength,
    limit_images
):
    if prompt_file is None:
        yield None, None, None, "Please upload a prompt file."
        return
    prompts = parse_prompt_file(prompt_file)
    if not prompts:
        gr.Warning("No prompts found in file or file could not be read. Please check your file.")
        yield None, None, None, "No prompts found."
        return

    prompts = [build_styled_prompt(p, style_preset) for p in prompts]
    prepare_character_refs([
        (char1_name, char1_img, char1_strength or 0.8),
        (char2_name, char2_img, char2_strength or 0.8),
    ])

    max_images = int(limit_images) if isinstance(limit_images, (int, float, str)) and str(limit_images).isdigit() and int(limit_images) > 0 else None

    stream = generate_images_stream(
        model_id=model_id,
        prompts=prompts,
        negative_prompt=negative_prompt,
        target_res=target_res,
        ratio=ratio,
        quality_mode=quality_mode,
        steps_override=int(num_steps) if num_steps else None,
        guidance_override=float(guidance_scale) if guidance_scale else None,
        seed=int(seed) if isinstance(seed, (int, float, str)) and str(seed).isdigit() and int(seed) >= 0 else None,
        images_per_prompt=int(images_per_prompt) if images_per_prompt else 1,
        max_images=max_images,
    )
    for gallery_images, first_png_path, zip_path, info in stream:
        yield gallery_images, first_png_path, zip_path, info

def build_ui(default_quality_mode: str = "Balanced"):
    custom_css = """
    .main-preview { min-height: 600px !important; }
    .settings-tab { padding: 20px !important; }
    .model-info-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(title="Batch Image Generator - YouTube Visuals", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🎨 Batch Image Generator for YouTube Visuals")
        gr.Markdown("Generate high-quality images with character consistency, streaming batches, and performance modes.")

        # Shared state variables (hidden)
        model_id = gr.State(value=DEFAULT_MODEL_ID)
        style_preset = gr.State(value="none")
        quality_mode = gr.State(value=default_quality_mode)
        target_res = gr.State(value="1080p")
        ratio = gr.State(value="16:9")
        images_per_prompt = gr.State(value=1)
        num_steps = gr.State(value=0)
        guidance_scale = gr.State(value=0.0)
        seed = gr.State(value=-1)
        prompt_text = gr.State(value="")
        prompt_file_state = gr.State(value=None)
        negative_prompt = gr.State(value="deformed, irregular eyes, ugly, bad anatomy, extra limbs, watermark, text, logo, blurry, low quality")
        char1_name = gr.State(value="")
        char1_img = gr.State(value=None)
        char1_strength = gr.State(value=0.8)
        char2_name = gr.State(value="")
        char2_img = gr.State(value=None)
        char2_strength = gr.State(value=0.8)
        limit_images = gr.State(value=0)
        
        # Large Preview Window (always visible)
        with gr.Row():
            with gr.Column(scale=2):
                gallery_main = gr.Gallery(
                    label="🎨 Live Preview",
                    columns=[3],
                    height=700,
                    elem_classes=["main-preview"],
                    show_label=True
                )
                status_info = gr.Markdown("Ready to generate. Configure settings in tabs below and click Generate.", elem_classes=["status-info"])
            
            with gr.Column(scale=1):
                # Action buttons
                gr.Markdown("### 🚀 Generate")
                btn_single = gr.Button("✨ Generate from Prompt", variant="primary", size="lg")
                btn_batch = gr.Button("📝 Generate Batch from File", variant="secondary", size="lg")
                gr.Markdown("---")
                
                # Quick settings summary
                gr.Markdown("### ⚙️ Current Settings")
                settings_summary = gr.Markdown("**Model:** Loading...\n**Style:** none\n**Resolution:** 1080p\n**Ratio:** 16:9")
        
        # Settings Tabs
        with gr.Tabs():
            with gr.Tab("🤖 Models"):
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=MODEL_OPTIONS[0],
                        label="Select AI Model",
                        info="Choose your text-to-image generation model"
                    )
                    model_info = gr.Markdown(get_model_info(DEFAULT_MODEL_ID), elem_classes=["model-info-box"])
                    quality_dropdown = gr.Dropdown(
                        choices=["Fast", "Balanced", "High"],
                        value=default_quality_mode,
                        label="Quality/Speed Mode",
                        info="Fast = quick drafts, Balanced = general use, High = final quality"
                    )
                    
                    def update_model_state(m, qm):
                        return m, qm, get_model_info(MODEL_NAME2PATH.get(m, DEFAULT_MODEL_ID))
                    model_dropdown.change(
                        fn=update_model_state,
                        inputs=[model_dropdown, quality_dropdown],
                        outputs=[model_id, quality_mode, model_info]
                    )
                    quality_dropdown.change(
                        fn=update_model_state,
                        inputs=[model_dropdown, quality_dropdown],
                        outputs=[model_id, quality_mode, model_info]
                    )
            
            with gr.Tab("🎨 Style"):
                style_dropdown = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="none",
                    label="Style Preset",
                    info="Apply artistic style enhancements to your prompts"
                )
                style_description = gr.Markdown("**Style:** none - No style preset applied")
                
                def update_style(s):
                    desc = STYLE_PRESETS.get(s, "")
                    return s, f"**Style:** {s}\n\n{desc if desc else 'No style preset applied'}"
                style_dropdown.change(fn=lambda s: update_style(s)[0], inputs=style_dropdown, outputs=style_preset)
                style_dropdown.change(fn=lambda s: update_style(s)[1], inputs=style_dropdown, outputs=style_description)
            
            with gr.Tab("📐 Resolution"):
                res_dropdown = gr.Dropdown(
                    choices=list(RES_OPTIONS.keys()),
                    value="1080p",
                    label="Resolution (Long Side)",
                    info="Target resolution for the longest side of the image"
                )
                res_preview = gr.Markdown("**Selected:** 1080p (long side)")
                
                def update_res(r):
                    return r, f"**Selected:** {r} (long side = {RES_OPTIONS[r]}px)"
                res_dropdown.change(fn=lambda r: update_res(r)[0], inputs=res_dropdown, outputs=target_res)
                res_dropdown.change(fn=lambda r: update_res(r)[1], inputs=res_dropdown, outputs=res_preview)
            
            with gr.Tab("⚖️ Ratio"):
                ratio_dropdown = gr.Dropdown(
                    choices=list(RATIO_OPTIONS.keys()),
                    value="16:9",
                    label="Aspect Ratio",
                    info="Image aspect ratio"
                )
                w, h = compute_size(1080, "16:9")
                ratio_preview = gr.Markdown(f"**Selected:** 16:9 → Final size: {w}x{h}px (at 1080p long side)")
                
                def update_ratio(r, current_res_val):
                    current_res = current_res_val if current_res_val else "1080p"
                    w, h = compute_size(RES_OPTIONS[current_res], r)
                    return r, f"**Selected:** {r} → Final size: {w}x{h}px (at {current_res} long side)"
                ratio_dropdown.change(
                    fn=lambda r: update_ratio(r, target_res.value)[0],
                    inputs=[ratio_dropdown],
                    outputs=ratio
                )
                ratio_dropdown.change(
                    fn=lambda r: update_ratio(r, target_res.value)[1],
                    inputs=[ratio_dropdown],
                    outputs=ratio_preview
                )
                # Also update when resolution changes
                res_dropdown.change(
                    fn=lambda res: update_ratio(ratio.value, res)[1],
                    inputs=[res_dropdown],
                    outputs=ratio_preview
                )
            
            with gr.Tab("🖼️ Images Per Prompt"):
                images_slider = gr.Slider(
                    1, 10, value=1, step=1,
                    label="Number of Images per Prompt",
                    info="Generate multiple variations from each prompt (1-10)"
                )
                images_info = gr.Markdown("**Current:** 1 image per prompt")
                
                def update_images(n):
                    return int(n), f"**Current:** {int(n)} image(s) per prompt"
                images_slider.change(fn=lambda n: update_images(n)[0], inputs=images_slider, outputs=images_per_prompt)
                images_slider.change(fn=lambda n: update_images(n)[1], inputs=images_slider, outputs=images_info)
            
            with gr.Tab("🔄 Inference Steps"):
                steps_slider = gr.Slider(
                    0, 60, value=0, step=1,
                    label="Inference Steps",
                    info="0 = auto (recommended). Higher = better quality but slower"
                )
                steps_info = gr.Markdown("**Current:** Auto (model will choose optimal steps)")
                
                def update_steps(s):
                    if s == 0:
                        return 0, "**Current:** Auto (model will choose optimal steps based on selected model and quality mode)"
                    return int(s), f"**Current:** {int(s)} steps"
                steps_slider.change(fn=lambda s: update_steps(s)[0], inputs=steps_slider, outputs=num_steps)
                steps_slider.change(fn=lambda s: update_steps(s)[1], inputs=steps_slider, outputs=steps_info)
            
            with gr.Tab("🎯 Guidance Scale"):
                guidance_slider = gr.Slider(
                    0.0, 20.0, value=0.0, step=0.5,
                    label="Guidance Scale (CFG)",
                    info="0 = auto (recommended). How closely to follow the prompt"
                )
                guidance_info = gr.Markdown("**Current:** Auto (model will choose optimal guidance)")
                
                def update_guidance(g):
                    if g == 0:
                        return 0.0, "**Current:** Auto (model will choose optimal guidance based on selected model and quality mode)"
                    return float(g), f"**Current:** {float(g)}"
                guidance_slider.change(fn=lambda g: update_guidance(g)[0], inputs=guidance_slider, outputs=guidance_scale)
                guidance_slider.change(fn=lambda g: update_guidance(g)[1], inputs=guidance_slider, outputs=guidance_info)
            
            with gr.Tab("🌱 Seed"):
                seed_input = gr.Number(
                    value=-1, precision=0,
                    label="Seed",
                    info="-1 or leave blank = random seed. Positive number = reproducible results"
                )
                seed_info = gr.Markdown("**Current:** Random (each generation will be unique)")
                
                def update_seed(s):
                    if s is None or s < 0:
                        return -1, "**Current:** Random (each generation will be unique)"
                    return int(s), f"**Current:** {int(s)} (reproducible)"
                seed_input.change(fn=lambda s: update_seed(s)[0], inputs=seed_input, outputs=seed)
                seed_input.change(fn=lambda s: update_seed(s)[1], inputs=seed_input, outputs=seed_info)
            
            with gr.Tab("📝 Prompts"):
                with gr.Column():
                    gr.Markdown("### Option 1: Type or Paste Prompts")
                    prompt_textbox = gr.Textbox(
                        lines=8,
                        label="Prompt Text",
                        placeholder="Enter your prompt here, or use the file upload option below...\n\nExample: A photorealistic rabbit with natural proportions, sitting in a garden, natural lighting, highly detailed"
                    )
                    gr.Markdown("---")
                    gr.Markdown("### Option 2: Upload Prompt File")
                    gr.Markdown("Upload a `.txt` file with one prompt per line. Lines starting with `#` are ignored.")
                    prompt_file_upload = gr.File(
                        label="Upload Prompt File (.txt)",
                        file_types=[".txt"]
                    )
                    
                    def update_prompts(txt, f):
                        return txt if txt else None, f
                    prompt_textbox.change(fn=lambda t: update_prompts(t, None)[0], inputs=prompt_textbox, outputs=prompt_text)
                    prompt_file_upload.change(fn=lambda f: update_prompts("", f)[1], inputs=prompt_file_upload, outputs=prompt_file_state)
            
            with gr.Tab("👥 Character Consistency"):
                char_enable = gr.Checkbox(
                    value=False,
                    label="Enable Character Consistency",
                    info="Use IP-Adapter XL to maintain character appearance across images"
                )
                
                with gr.Row(visible=False) as char_row1:
                    char1_name_input = gr.Textbox(label="Character 1 Name", placeholder="e.g., Aria (use this name in your prompts)")
                    char1_img_input = gr.Image(type="pil", label="Character 1 Reference Image", height=200)
                    char1_strength_input = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Identity Strength")
                
                with gr.Row(visible=False) as char_row2:
                    char2_name_input = gr.Textbox(label="Character 2 Name", placeholder="e.g., Kian (use this name in your prompts)")
                    char2_img_input = gr.Image(type="pil", label="Character 2 Reference Image", height=200)
                    char2_strength_input = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Identity Strength")
                
                def toggle_chars(enabled):
                    return gr.update(visible=enabled), gr.update(visible=enabled)
                char_enable.change(fn=toggle_chars, inputs=char_enable, outputs=[char_row1, char_row2])
                
                def update_char1(name, img, strength):
                    return name or "", img, float(strength) if strength else 0.8
                char1_name_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                char1_img_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                char1_strength_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                
                def update_char2(name, img, strength):
                    return name or "", img, float(strength) if strength else 0.8
                char2_name_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
                char2_img_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
                char2_strength_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
            
            with gr.Tab("❌ Negative Prompts"):
                negative_textbox = gr.Textbox(
                    lines=4,
                    label="Negative Prompt",
                    value="deformed, irregular eyes, ugly, bad anatomy, extra limbs, watermark, text, logo, blurry, low quality",
                    info="Things to avoid in generated images"
                )
                negative_textbox.change(fn=lambda n: n, inputs=negative_textbox, outputs=negative_prompt)
            
            with gr.Tab("📥 Downloads"):
                gr.Markdown("### Generated Images")
                download_first = gr.File(label="📥 Download First Image (PNG)")
                download_zip = gr.File(label="📦 Download All Images (ZIP)")
                gr.Markdown("---")
                gr.Markdown("**Note:** Downloads update automatically as images are generated. ZIP file grows incrementally for batch operations.")
        
        # Generation handlers using direct inputs
        def generate_single_handler(model_sel, prompt_text_in, neg_prompt_in, style_sel, res_sel, ratio_sel,
                                    quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                                    c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength):
            if not prompt_text_in or not str(prompt_text_in).strip():
                return None, None, None, "⚠️ Please enter a prompt."
            result = ui_generate_single(
                MODEL_NAME2PATH.get(model_sel, DEFAULT_MODEL_ID), str(prompt_text_in), neg_prompt_in or "", style_sel, res_sel, ratio_sel,
                quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength
            )
            if result:
                imgs, first_file, zip_file, info = result
                return imgs, first_file, zip_file, info or "✅ Generation complete!"
            return None, None, None, "❌ Generation failed. Check settings and try again."

        def generate_batch_handler(model_sel, prompt_file_in, neg_prompt_in, style_sel, res_sel, ratio_sel,
                                   quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                                   c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength):
            if prompt_file_in is None:
                yield None, None, None, "⚠️ Please upload a prompt file (.txt)."
                return
                for state in ui_generate_batch_stream(
                MODEL_NAME2PATH.get(model_sel, DEFAULT_MODEL_ID), prompt_file_in, neg_prompt_in or "", style_sel, res_sel, ratio_sel,
                quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength,
                0,
                ):
                    yield state
        
        btn_single.click(
            fn=generate_single_handler,
            inputs=[
                model_dropdown, prompt_textbox, negative_textbox, style_dropdown, res_dropdown, ratio_dropdown,
                quality_dropdown, steps_slider, guidance_slider, seed_input, images_slider,
                char1_name_input, char1_img_input, char1_strength_input,
                char2_name_input, char2_img_input, char2_strength_input,
            ],
            outputs=[gallery_main, download_first, download_zip, status_info],
            queue=True,
        )

            btn_batch.click(
            fn=generate_batch_handler,
            inputs=[
                model_dropdown, prompt_file_upload, negative_textbox, style_dropdown, res_dropdown, ratio_dropdown,
                quality_dropdown, steps_slider, guidance_slider, seed_input, images_slider,
                char1_name_input, char1_img_input, char1_strength_input,
                char2_name_input, char2_img_input, char2_strength_input,
            ],
            outputs=[gallery_main, download_first, download_zip, status_info],
            queue=True,
            )

        with gr.Accordion("💡 Tips & Guidelines", open=False):
            gr.Markdown("""
### Model picks
- FLUX.1-dev: Best quality (20–30 steps, guidance ~7)
- FLUX.1-schnell: Fastest (4–8 steps, guidance ~2)
- SDXL Turbo: Very fast (4–6 steps, guidance ~2)
- RealVis XL V6 / Juggernaut XL: Realistic photos (18–35 steps, guidance ~7)

### Speed/Quality
- Use "Fast" for drafts, "Balanced" for most work, "High" for finals.
- 2k long side supported; streaming ZIP updates as images complete.

### Character Consistency
- Upload clear frontal reference images; use names in prompts.
- Identity strength 0.6–0.9 recommended.
""")
        gr.Markdown("Made for YouTube creators 🎬 | Streams results and supports very large batches.")
    return demo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_name", default="0.0.0.0")
    ap.add_argument("--server_port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    default_mode = os.environ.get("DEFAULT_QUALITY_MODE", "Balanced")
    demo = build_ui(default_mode)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)

if __name__ == "__main__":
    print(f"Using device: {DEVICE} | dtype: {DTYPE} | DirectML: {USE_DML}")
    main()
=======
# app.py
import os
import gc
import time
import zipfile
import tempfile
import threading
import argparse
from typing import List, Tuple, Optional, Dict, Any

import torch
from PIL import Image
import gradio as gr
import yaml

from diffusers import (
    AutoPipelineForText2Image,
    StableDiffusionXLPipeline,
    DPMSolverMultistepScheduler,
)

def select_device_and_dtype():
    try:
        import torch_directml  # optional on Windows AMD
        return torch_directml.device(), torch.float16, True
    except Exception:
        pass
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        try:
            if torch.cuda.is_bf16_supported():
                return device, torch.bfloat16, False
        except Exception:
            pass
        return device, torch.float16, False
    return device, torch.float32, False

DEVICE, DTYPE, USE_DML = select_device_and_dtype()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

def load_models_yaml(yaml_path: str = "models.yaml"):
    paths_to_try = [yaml_path, os.path.join("app", "config", "models.yaml")]
    for path in paths_to_try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Support both dict or list style yaml
            models = data.get("models") if isinstance(data, dict) else data
            result = []
            for model in models:
                if isinstance(model, dict) and "name" in model and "path" in model:
                    result.append({"name": model["name"], "path": model["path"], "description": model.get("description", "")})
            return result
    return []

MODEL_YAML_ENTRIES = load_models_yaml()
if not MODEL_YAML_ENTRIES:
    raise RuntimeError("No models found in models.yaml! Please add models with HuggingFace paths.")

MODEL_OPTIONS = [x["name"] for x in MODEL_YAML_ENTRIES]
MODEL_NAME2PATH = {x["name"]: x["path"] for x in MODEL_YAML_ENTRIES}
DEFAULT_MODEL_ID = MODEL_NAME2PATH.get(MODEL_OPTIONS[0])

STYLE_PRESETS = {
    "none": "",
    "realistic": "photorealistic highly detailed face perfect anatomy natural skin texture professional photography sharp focus detailed eyes realistic proportions 8k uhd dslr",
    "anime": "anime style detailed anime artwork vibrant colors clean lines detailed background studio ghibli inspired smooth shading",
    "3d cartoon": "3d render Pixar style soft lighting smooth surfaces vibrant colors professional 3d modeling subsurface scattering",
    "sketch": "pencil sketch detailed line art crosshatching monochrome artistic sketch fine lines shading details",
    "painting": "oil painting classical art style rich colors impasto brush strokes canvas texture detailed",
    "chibi": "chibi style cute and adorable big head small body proportions kawaii bright colors",
    "ghibli": "studio ghibli style whimsical and dreamy soft pastel colors painterly background magical atmosphere",
}
RES_OPTIONS = {"480p": 480, "720p": 720, "1080p": 1080, "2k": 1440, "4k": 2160}
RATIO_OPTIONS = {"1:1": (1, 1), "2:3": (2, 3), "3:2": (3, 2), "16:9": (16, 9), "9:16": (9, 16)}

def round_to_multiple(value: int, multiple: int) -> int:
    return int(round(value / multiple) * multiple)

def compute_size(target_long_side: int, ratio_str: str, base_multiple: int = 8) -> Tuple[int, int]:
    rw, rh = RATIO_OPTIONS[ratio_str]
    if rw >= rh:
        width = target_long_side
        height = int(target_long_side * rh / rw)
    else:
        height = target_long_side
        width = int(target_long_side * rw / rh)
    width = max(base_multiple, round_to_multiple(width, base_multiple))
    height = max(base_multiple, round_to_multiple(height, base_multiple))
    return width, height

def build_styled_prompt(prompt: str, style_key: str) -> str:
    style = STYLE_PRESETS.get(style_key, "").strip()
    return f"{prompt.strip()}, {style}" if style else prompt.strip()

def parse_prompt_file(file_obj) -> List[str]:
    if file_obj is None:
        return []
    try:
        if hasattr(file_obj, "name"):
            with open(file_obj.name, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            if hasattr(file_obj, "read"):
                content = file_obj.read()
                if isinstance(content, bytes):
                    content = content.decode("utf-8")
            else:
                content = str(file_obj)
    except Exception:
        return []
    lines = [ln.strip() for ln in content.splitlines()]
    return [ln for ln in lines if ln and not ln.lstrip().startswith("#")]

def append_image_to_zip(zip_path: str, img_path: str):
    with zipfile.ZipFile(zip_path, "a", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.exists(img_path):
            zf.write(img_path, arcname=os.path.basename(img_path))

def create_zip_container() -> str:
    zip_dir = tempfile.mkdtemp(prefix="zip_")
    zip_path = os.path.join(zip_dir, "images.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED):
        pass
    return zip_path

_HEARTBEAT_STOP = False
def _heartbeat():
    while not _HEARTBEAT_STOP:
        print(f"[heartbeat] {time.strftime('%H:%M:%S')}")
        time.sleep(45)

PIPE = None
PIPE_MODEL_ID = None
IP_ADAPTER_READY = False
CHAR_BANK: Dict[str, Dict[str, Any]] = {}

def del_pipe():
    global PIPE, PIPE_MODEL_ID, IP_ADAPTER_READY
    if PIPE is not None:
        del PIPE
    PIPE = None
    PIPE_MODEL_ID = None
    IP_ADAPTER_READY = False
    gc.collect()
    try:
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

def load_pipeline(model_id: str):
    global PIPE, PIPE_MODEL_ID, IP_ADAPTER_READY
    if PIPE is not None and PIPE_MODEL_ID == model_id:
        return
    del_pipe()

    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        variant="fp16" if DTYPE == torch.float16 else None,
        use_safetensors=True,
    )

    try:
        from diffusers import EulerAncestralDiscreteScheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    except Exception:
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras=True)
        except Exception:
            pass

    if not USE_DML:
        for fn in [
            getattr(pipe, "enable_xformers_memory_efficient_attention", None),
            getattr(pipe, "enable_sdp_attention", None),
            getattr(pipe, "enable_vae_slicing", None),
            getattr(pipe, "enable_vae_tiling", None),
        ]:
            try:
                if fn:
                    fn()
            except Exception:
                pass

    try:
        pipe.to(DEVICE if USE_DML else DEVICE, dtype=None if USE_DML else DTYPE)
    except Exception:
        pipe.to("cpu")

    global IP_ADAPTER_READY
    IP_ADAPTER_READY = False
    try:
        if isinstance(pipe, StableDiffusionXLPipeline):
            pipe.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter-plus_sdxl.safetensors",
            )
            IP_ADAPTER_READY = True
    except Exception:
        IP_ADAPTER_READY = False

    PIPE = pipe
    PIPE_MODEL_ID = model_id

def default_steps_and_scale(model_id: str, quality_mode: str) -> Tuple[int, float]:
    mid = model_id.lower()
    if "schnell" in mid or "turbo" in mid:
        return (8 if quality_mode == "High" else 6 if quality_mode == "Balanced" else 4), 2.0 if quality_mode != "Fast" else 1.8
    if "flux" in mid:
        return (28 if quality_mode == "High" else 20 if quality_mode == "Balanced" else 14), (7.5 if quality_mode == "High" else 6.5 if quality_mode == "Balanced" else 5.5)
    if any(k in mid for k in ["realvis", "juggernaut", "stable-diffusion-xl", "playground"]):
        return (35 if quality_mode == "High" else 25 if quality_mode == "Balanced" else 18), (7.5 if quality_mode == "High" else 7.0 if quality_mode == "Balanced" else 6.5)
    # SD1.5 family: runwayml sd-v1-5, realistic vision v4, realistic stock photo v2.0
    if any(k in mid for k in ["sd-v1-5", "stable-diffusion-v1-5", "realistic_vision_v4", "realistic-vision", "realisticstockphoto", "realisticstockphoto-v20"]):
        return (30 if quality_mode == "High" else 25 if quality_mode == "Balanced" else 18), (7.0 if quality_mode != "Fast" else 6.0)
    return (25 if quality_mode != "Fast" else 12), (7.0 if quality_mode != "Fast" else 5.0)

def get_model_info(model_id: str) -> str:
    """Returns helpful info about model's recommended settings"""
    mid = model_id.lower()
    if "schnell" in mid or "turbo" in mid:
        return "⚡ Fast model: 4-8 steps, guidance ~2.0. Best for quick drafts and iterations."
    if "flux" in mid and "dev" in mid:
        return "🎨 Premium quality: 20-30 steps, guidance ~7.0. Best overall quality, slower generation."
    if "flux" in mid and "schnell" in mid:
        return "⚡ Fast FLUX: 4-8 steps, guidance ~2.0. Good quality with speed."
    if "playground" in mid:
        return "🏞️ Playground v2: 18-35 steps, guidance ~7.0. Excellent for aesthetics and general use."
    if "juggernaut" in mid:
        return "💪 Juggernaut XL: 25-35 steps, guidance ~7.0. Powerful, detailed outputs."
    if "realvis" in mid:
        return "📸 RealVis XL: 25-35 steps, guidance ~7.5. Optimized for photorealistic images."
    if "stable-diffusion-xl" in mid and "base" in mid:
        return "🌟 SDXL Base: 25-35 steps, guidance ~7.5. Balanced quality and speed."
    if any(k in mid for k in ["sd-v1-5", "stable-diffusion-v1-5"]):
        return "🔧 SD1.5: 18-30 steps, guidance ~7.0. Classic model, wide community support."
    if "realistic" in mid:
        return "📷 Realistic models: 18-30 steps, guidance ~7.0. Specialized for photorealistic outputs."
    return "⚙️ Default: 12-25 steps, guidance ~7.0. Adjust based on quality/speed needs."

def prepare_character_refs(char_entries: List[Tuple[str, Optional[Image.Image], float]]):
    CHAR_BANK.clear()
    for name, img, strength in char_entries:
        if name and img is not None:
            CHAR_BANK[name.strip()] = {"images": [img], "strength": float(max(0.0, min(1.0, strength)))}

@torch.inference_mode()
def generate_images_stream(
    model_id: str,
    prompts: List[str],
    negative_prompt: str,
    target_res: str,
    ratio: str,
    quality_mode: str,
    steps_override: Optional[int],
    guidance_override: Optional[float],
    seed: Optional[int],
    images_per_prompt: int = 1,
    max_images: Optional[int] = None,
):
    if not prompts:
        yield None, None, None, "No prompts provided."
        return

    try:
        load_pipeline(model_id)
    except Exception as e:
        yield None, None, None, f"Failed to load model: {e}"
        return

    generator = None
    if (not USE_DML) and seed is not None and seed >= 0:
        generator = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(int(seed))

    width, height = compute_size(RES_OPTIONS[target_res], ratio)
    auto_steps, auto_scale = default_steps_and_scale(model_id, quality_mode)
    steps = int(steps_override) if steps_override and int(steps_override) > 0 else auto_steps
    gscale = float(guidance_override) if guidance_override and float(guidance_override) > 0 else auto_scale

    neg_prompt = negative_prompt or "low quality, worst quality, watermark, logo, text, blurry, jpeg artifacts, bad anatomy, extra limbs, deformed, distorted"

    gallery_images: List[Image.Image] = []
    first_png_path = None
    zip_path = create_zip_container()
    total = len(prompts) if not max_images else min(len(prompts), max_images)
    images_per_prompt = max(1, min(10, int(images_per_prompt)))  # Clamp 1-10
    total_images = total * images_per_prompt

    ip_images, ip_scales = [], []
    if IP_ADAPTER_READY and len(CHAR_BANK) > 0 and isinstance(PIPE, StableDiffusionXLPipeline):
        for _, entry in CHAR_BANK.items():
            ip_images.extend(entry["images"])
            ip_scales.append(entry["strength"])
        ip_images = ip_images[:2]
        ip_scales = (ip_scales[:2] or [0.8])

    generated_count = 0
    for prompt_idx, prompt in enumerate(prompts[:total]):
        for img_idx in range(images_per_prompt):
            try:
                # Use different seeds for variety when generating multiple images per prompt
                current_seed = seed
                if seed is None or seed < 0:
                    # Random seed for each image
                    current_seed = None
                elif images_per_prompt > 1:
                    # Offset seed for variety: base_seed + prompt_idx * 1000 + img_idx
                    current_seed = int(seed) + prompt_idx * 1000 + img_idx

                gen = None
                if (not USE_DML) and current_seed is not None and current_seed >= 0:
                    gen = torch.Generator(device="cuda" if DEVICE == "cuda" else "cpu").manual_seed(current_seed)

            kwargs = dict(
                prompt=prompt.strip(),
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=gscale,
            )
                if gen is not None:
                    kwargs["generator"] = gen
            if IP_ADAPTER_READY and len(ip_images) > 0 and isinstance(PIPE, StableDiffusionXLPipeline):
                kwargs["ip_adapter_image"] = ip_images
                kwargs["ip_adapter_image_embeds"] = None
                kwargs["ip_adapter_scale"] = ip_scales

            out = PIPE(**kwargs)
            img = out.images[0]

            img_dir = tempfile.mkdtemp(prefix="gen_")
                out_path = os.path.join(img_dir, f"prompt_{prompt_idx+1:04d}_img_{img_idx+1:02d}.png")
            img.save(out_path, "PNG")
            append_image_to_zip(zip_path, out_path)

            if first_png_path is None:
                first_png_path = out_path

                generated_count += 1
            gallery_images.append(img)
            if len(gallery_images) > 16:
                gallery_images = gallery_images[-16:]

                info = f"Generated {generated_count}/{total_images} | Prompt {prompt_idx+1}/{total} | Image {img_idx+1}/{images_per_prompt} | {width}x{height} | steps={steps} | scale={gscale}"
            yield gallery_images, first_png_path, zip_path, info

            del img, out
                if DEVICE == "cuda" and (generated_count % 8 == 0):
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
                yield gallery_images, first_png_path, zip_path, f"Error on prompt {prompt_idx+1}, image {img_idx+1}: {e}"
            continue

    yield gallery_images, first_png_path, zip_path, f"Done. Generated {generated_count} images from {total} prompts."
    return

def ui_generate_single(
    model_id, prompt, negative_prompt, style_preset, target_res, ratio,
    quality_mode, num_steps, guidance_scale, seed, images_per_prompt,
    char1_name, char1_img, char1_strength,
    char2_name, char2_img, char2_strength,
):
    sprompt = build_styled_prompt(prompt, style_preset)
    prepare_character_refs([
        (char1_name, char1_img, char1_strength or 0.8),
        (char2_name, char2_img, char2_strength or 0.8),
    ])
    stream = generate_images_stream(
        model_id=model_id,
        prompts=[sprompt],
        negative_prompt=negative_prompt,
        target_res=target_res,
        ratio=ratio,
        quality_mode=quality_mode,
        steps_override=int(num_steps) if num_steps else None,
        guidance_override=float(guidance_scale) if guidance_scale else None,
        seed=int(seed) if isinstance(seed, (int, float, str)) and str(seed).isdigit() and int(seed) >= 0 else None,
        images_per_prompt=int(images_per_prompt) if images_per_prompt else 1,
    )
    last = None
    for state in stream:
        last = state
    if not last:
        return None, None, None, "No image"
    return last

def ui_generate_batch_stream(
    model_id, prompt_file, negative_prompt, style_preset, target_res, ratio,
    quality_mode, num_steps, guidance_scale, seed, images_per_prompt,
    char1_name, char1_img, char1_strength,
    char2_name, char2_img, char2_strength,
    limit_images
):
    if prompt_file is None:
        yield None, None, None, "Please upload a prompt file."
        return
    prompts = parse_prompt_file(prompt_file)
    if not prompts:
        gr.Warning("No prompts found in file or file could not be read. Please check your file.")
        yield None, None, None, "No prompts found."
        return

    prompts = [build_styled_prompt(p, style_preset) for p in prompts]
    prepare_character_refs([
        (char1_name, char1_img, char1_strength or 0.8),
        (char2_name, char2_img, char2_strength or 0.8),
    ])

    max_images = int(limit_images) if isinstance(limit_images, (int, float, str)) and str(limit_images).isdigit() and int(limit_images) > 0 else None

    stream = generate_images_stream(
        model_id=model_id,
        prompts=prompts,
        negative_prompt=negative_prompt,
        target_res=target_res,
        ratio=ratio,
        quality_mode=quality_mode,
        steps_override=int(num_steps) if num_steps else None,
        guidance_override=float(guidance_scale) if guidance_scale else None,
        seed=int(seed) if isinstance(seed, (int, float, str)) and str(seed).isdigit() and int(seed) >= 0 else None,
        images_per_prompt=int(images_per_prompt) if images_per_prompt else 1,
        max_images=max_images,
    )
    for gallery_images, first_png_path, zip_path, info in stream:
        yield gallery_images, first_png_path, zip_path, info

def build_ui(default_quality_mode: str = "Balanced"):
    custom_css = """
    .main-preview { min-height: 600px !important; }
    .settings-tab { padding: 20px !important; }
    .model-info-box { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 8px;
        color: white;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(title="Batch Image Generator - YouTube Visuals", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# 🎨 Batch Image Generator for YouTube Visuals")
        gr.Markdown("Generate high-quality images with character consistency, streaming batches, and performance modes.")

        # Shared state variables (hidden)
        model_id = gr.State(value=DEFAULT_MODEL_ID)
        style_preset = gr.State(value="none")
        quality_mode = gr.State(value=default_quality_mode)
        target_res = gr.State(value="1080p")
        ratio = gr.State(value="16:9")
        images_per_prompt = gr.State(value=1)
        num_steps = gr.State(value=0)
        guidance_scale = gr.State(value=0.0)
        seed = gr.State(value=-1)
        prompt_text = gr.State(value="")
        prompt_file_state = gr.State(value=None)
        negative_prompt = gr.State(value="deformed, irregular eyes, ugly, bad anatomy, extra limbs, watermark, text, logo, blurry, low quality")
        char1_name = gr.State(value="")
        char1_img = gr.State(value=None)
        char1_strength = gr.State(value=0.8)
        char2_name = gr.State(value="")
        char2_img = gr.State(value=None)
        char2_strength = gr.State(value=0.8)
        limit_images = gr.State(value=0)
        
        # Large Preview Window (always visible)
        with gr.Row():
            with gr.Column(scale=2):
                gallery_main = gr.Gallery(
                    label="🎨 Live Preview",
                    columns=[3],
                    height=700,
                    elem_classes=["main-preview"],
                    show_label=True
                )
                status_info = gr.Markdown("Ready to generate. Configure settings in tabs below and click Generate.", elem_classes=["status-info"])
            
            with gr.Column(scale=1):
                # Action buttons
                gr.Markdown("### 🚀 Generate")
                btn_single = gr.Button("✨ Generate from Prompt", variant="primary", size="lg")
                btn_batch = gr.Button("📝 Generate Batch from File", variant="secondary", size="lg")
                gr.Markdown("---")
                
                # Quick settings summary
                gr.Markdown("### ⚙️ Current Settings")
                settings_summary = gr.Markdown("**Model:** Loading...\n**Style:** none\n**Resolution:** 1080p\n**Ratio:** 16:9")
        
        # Settings Tabs
        with gr.Tabs():
            with gr.Tab("🤖 Models"):
                with gr.Column():
                    model_dropdown = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=MODEL_OPTIONS[0],
                        label="Select AI Model",
                        info="Choose your text-to-image generation model"
                    )
                    model_info = gr.Markdown(get_model_info(DEFAULT_MODEL_ID), elem_classes=["model-info-box"])
                    quality_dropdown = gr.Dropdown(
                        choices=["Fast", "Balanced", "High"],
                        value=default_quality_mode,
                        label="Quality/Speed Mode",
                        info="Fast = quick drafts, Balanced = general use, High = final quality"
                    )
                    
                    def update_model_state(m, qm):
                        return m, qm, get_model_info(MODEL_NAME2PATH.get(m, DEFAULT_MODEL_ID))
                    model_dropdown.change(
                        fn=update_model_state,
                        inputs=[model_dropdown, quality_dropdown],
                        outputs=[model_id, quality_mode, model_info]
                    )
                    quality_dropdown.change(
                        fn=update_model_state,
                        inputs=[model_dropdown, quality_dropdown],
                        outputs=[model_id, quality_mode, model_info]
                    )
            
            with gr.Tab("🎨 Style"):
                style_dropdown = gr.Dropdown(
                    choices=list(STYLE_PRESETS.keys()),
                    value="none",
                    label="Style Preset",
                    info="Apply artistic style enhancements to your prompts"
                )
                style_description = gr.Markdown("**Style:** none - No style preset applied")
                
                def update_style(s):
                    desc = STYLE_PRESETS.get(s, "")
                    return s, f"**Style:** {s}\n\n{desc if desc else 'No style preset applied'}"
                style_dropdown.change(fn=lambda s: update_style(s)[0], inputs=style_dropdown, outputs=style_preset)
                style_dropdown.change(fn=lambda s: update_style(s)[1], inputs=style_dropdown, outputs=style_description)
            
            with gr.Tab("📐 Resolution"):
                res_dropdown = gr.Dropdown(
                    choices=list(RES_OPTIONS.keys()),
                    value="1080p",
                    label="Resolution (Long Side)",
                    info="Target resolution for the longest side of the image"
                )
                res_preview = gr.Markdown("**Selected:** 1080p (long side)")
                
                def update_res(r):
                    return r, f"**Selected:** {r} (long side = {RES_OPTIONS[r]}px)"
                res_dropdown.change(fn=lambda r: update_res(r)[0], inputs=res_dropdown, outputs=target_res)
                res_dropdown.change(fn=lambda r: update_res(r)[1], inputs=res_dropdown, outputs=res_preview)
            
            with gr.Tab("⚖️ Ratio"):
                ratio_dropdown = gr.Dropdown(
                    choices=list(RATIO_OPTIONS.keys()),
                    value="16:9",
                    label="Aspect Ratio",
                    info="Image aspect ratio"
                )
                w, h = compute_size(1080, "16:9")
                ratio_preview = gr.Markdown(f"**Selected:** 16:9 → Final size: {w}x{h}px (at 1080p long side)")
                
                def update_ratio(r, current_res_val):
                    current_res = current_res_val if current_res_val else "1080p"
                    w, h = compute_size(RES_OPTIONS[current_res], r)
                    return r, f"**Selected:** {r} → Final size: {w}x{h}px (at {current_res} long side)"
                ratio_dropdown.change(
                    fn=lambda r: update_ratio(r, target_res.value)[0],
                    inputs=[ratio_dropdown],
                    outputs=ratio
                )
                ratio_dropdown.change(
                    fn=lambda r: update_ratio(r, target_res.value)[1],
                    inputs=[ratio_dropdown],
                    outputs=ratio_preview
                )
                # Also update when resolution changes
                res_dropdown.change(
                    fn=lambda res: update_ratio(ratio.value, res)[1],
                    inputs=[res_dropdown],
                    outputs=ratio_preview
                )
            
            with gr.Tab("🖼️ Images Per Prompt"):
                images_slider = gr.Slider(
                    1, 10, value=1, step=1,
                    label="Number of Images per Prompt",
                    info="Generate multiple variations from each prompt (1-10)"
                )
                images_info = gr.Markdown("**Current:** 1 image per prompt")
                
                def update_images(n):
                    return int(n), f"**Current:** {int(n)} image(s) per prompt"
                images_slider.change(fn=lambda n: update_images(n)[0], inputs=images_slider, outputs=images_per_prompt)
                images_slider.change(fn=lambda n: update_images(n)[1], inputs=images_slider, outputs=images_info)
            
            with gr.Tab("🔄 Inference Steps"):
                steps_slider = gr.Slider(
                    0, 60, value=0, step=1,
                    label="Inference Steps",
                    info="0 = auto (recommended). Higher = better quality but slower"
                )
                steps_info = gr.Markdown("**Current:** Auto (model will choose optimal steps)")
                
                def update_steps(s):
                    if s == 0:
                        return 0, "**Current:** Auto (model will choose optimal steps based on selected model and quality mode)"
                    return int(s), f"**Current:** {int(s)} steps"
                steps_slider.change(fn=lambda s: update_steps(s)[0], inputs=steps_slider, outputs=num_steps)
                steps_slider.change(fn=lambda s: update_steps(s)[1], inputs=steps_slider, outputs=steps_info)
            
            with gr.Tab("🎯 Guidance Scale"):
                guidance_slider = gr.Slider(
                    0.0, 20.0, value=0.0, step=0.5,
                    label="Guidance Scale (CFG)",
                    info="0 = auto (recommended). How closely to follow the prompt"
                )
                guidance_info = gr.Markdown("**Current:** Auto (model will choose optimal guidance)")
                
                def update_guidance(g):
                    if g == 0:
                        return 0.0, "**Current:** Auto (model will choose optimal guidance based on selected model and quality mode)"
                    return float(g), f"**Current:** {float(g)}"
                guidance_slider.change(fn=lambda g: update_guidance(g)[0], inputs=guidance_slider, outputs=guidance_scale)
                guidance_slider.change(fn=lambda g: update_guidance(g)[1], inputs=guidance_slider, outputs=guidance_info)
            
            with gr.Tab("🌱 Seed"):
                seed_input = gr.Number(
                    value=-1, precision=0,
                    label="Seed",
                    info="-1 or leave blank = random seed. Positive number = reproducible results"
                )
                seed_info = gr.Markdown("**Current:** Random (each generation will be unique)")
                
                def update_seed(s):
                    if s is None or s < 0:
                        return -1, "**Current:** Random (each generation will be unique)"
                    return int(s), f"**Current:** {int(s)} (reproducible)"
                seed_input.change(fn=lambda s: update_seed(s)[0], inputs=seed_input, outputs=seed)
                seed_input.change(fn=lambda s: update_seed(s)[1], inputs=seed_input, outputs=seed_info)
            
            with gr.Tab("📝 Prompts"):
                with gr.Column():
                    gr.Markdown("### Option 1: Type or Paste Prompts")
                    prompt_textbox = gr.Textbox(
                        lines=8,
                        label="Prompt Text",
                        placeholder="Enter your prompt here, or use the file upload option below...\n\nExample: A photorealistic rabbit with natural proportions, sitting in a garden, natural lighting, highly detailed"
                    )
                    gr.Markdown("---")
                    gr.Markdown("### Option 2: Upload Prompt File")
                    gr.Markdown("Upload a `.txt` file with one prompt per line. Lines starting with `#` are ignored.")
                    prompt_file_upload = gr.File(
                        label="Upload Prompt File (.txt)",
                        file_types=[".txt"]
                    )
                    
                    def update_prompts(txt, f):
                        return txt if txt else None, f
                    prompt_textbox.change(fn=lambda t: update_prompts(t, None)[0], inputs=prompt_textbox, outputs=prompt_text)
                    prompt_file_upload.change(fn=lambda f: update_prompts("", f)[1], inputs=prompt_file_upload, outputs=prompt_file_state)
            
            with gr.Tab("👥 Character Consistency"):
                char_enable = gr.Checkbox(
                    value=False,
                    label="Enable Character Consistency",
                    info="Use IP-Adapter XL to maintain character appearance across images"
                )
                
                with gr.Row(visible=False) as char_row1:
                    char1_name_input = gr.Textbox(label="Character 1 Name", placeholder="e.g., Aria (use this name in your prompts)")
                    char1_img_input = gr.Image(type="pil", label="Character 1 Reference Image", height=200)
                    char1_strength_input = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Identity Strength")
                
                with gr.Row(visible=False) as char_row2:
                    char2_name_input = gr.Textbox(label="Character 2 Name", placeholder="e.g., Kian (use this name in your prompts)")
                    char2_img_input = gr.Image(type="pil", label="Character 2 Reference Image", height=200)
                    char2_strength_input = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Identity Strength")
                
                def toggle_chars(enabled):
                    return gr.update(visible=enabled), gr.update(visible=enabled)
                char_enable.change(fn=toggle_chars, inputs=char_enable, outputs=[char_row1, char_row2])
                
                def update_char1(name, img, strength):
                    return name or "", img, float(strength) if strength else 0.8
                char1_name_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                char1_img_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                char1_strength_input.change(fn=update_char1, inputs=[char1_name_input, char1_img_input, char1_strength_input], outputs=[char1_name, char1_img, char1_strength])
                
                def update_char2(name, img, strength):
                    return name or "", img, float(strength) if strength else 0.8
                char2_name_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
                char2_img_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
                char2_strength_input.change(fn=update_char2, inputs=[char2_name_input, char2_img_input, char2_strength_input], outputs=[char2_name, char2_img, char2_strength])
            
            with gr.Tab("❌ Negative Prompts"):
                negative_textbox = gr.Textbox(
                    lines=4,
                    label="Negative Prompt",
                    value="deformed, irregular eyes, ugly, bad anatomy, extra limbs, watermark, text, logo, blurry, low quality",
                    info="Things to avoid in generated images"
                )
                negative_textbox.change(fn=lambda n: n, inputs=negative_textbox, outputs=negative_prompt)
            
            with gr.Tab("📥 Downloads"):
                gr.Markdown("### Generated Images")
                download_first = gr.File(label="📥 Download First Image (PNG)")
                download_zip = gr.File(label="📦 Download All Images (ZIP)")
                gr.Markdown("---")
                gr.Markdown("**Note:** Downloads update automatically as images are generated. ZIP file grows incrementally for batch operations.")
        
        # Generation handlers using direct inputs
        def generate_single_handler(model_sel, prompt_text_in, neg_prompt_in, style_sel, res_sel, ratio_sel,
                                    quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                                    c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength):
            if not prompt_text_in or not str(prompt_text_in).strip():
                return None, None, None, "⚠️ Please enter a prompt."
            result = ui_generate_single(
                MODEL_NAME2PATH.get(model_sel, DEFAULT_MODEL_ID), str(prompt_text_in), neg_prompt_in or "", style_sel, res_sel, ratio_sel,
                quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength
            )
            if result:
                imgs, first_file, zip_file, info = result
                return imgs, first_file, zip_file, info or "✅ Generation complete!"
            return None, None, None, "❌ Generation failed. Check settings and try again."

        def generate_batch_handler(model_sel, prompt_file_in, neg_prompt_in, style_sel, res_sel, ratio_sel,
                                   quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                                   c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength):
            if prompt_file_in is None:
                yield None, None, None, "⚠️ Please upload a prompt file (.txt)."
                return
                for state in ui_generate_batch_stream(
                MODEL_NAME2PATH.get(model_sel, DEFAULT_MODEL_ID), prompt_file_in, neg_prompt_in or "", style_sel, res_sel, ratio_sel,
                quality_sel, steps_in, guidance_in, seed_in, images_per_prompt_in,
                c1_name, c1_img, c1_strength, c2_name, c2_img, c2_strength,
                0,
                ):
                    yield state
        
        btn_single.click(
            fn=generate_single_handler,
            inputs=[
                model_dropdown, prompt_textbox, negative_textbox, style_dropdown, res_dropdown, ratio_dropdown,
                quality_dropdown, steps_slider, guidance_slider, seed_input, images_slider,
                char1_name_input, char1_img_input, char1_strength_input,
                char2_name_input, char2_img_input, char2_strength_input,
            ],
            outputs=[gallery_main, download_first, download_zip, status_info],
            queue=True,
        )

            btn_batch.click(
            fn=generate_batch_handler,
            inputs=[
                model_dropdown, prompt_file_upload, negative_textbox, style_dropdown, res_dropdown, ratio_dropdown,
                quality_dropdown, steps_slider, guidance_slider, seed_input, images_slider,
                char1_name_input, char1_img_input, char1_strength_input,
                char2_name_input, char2_img_input, char2_strength_input,
            ],
            outputs=[gallery_main, download_first, download_zip, status_info],
            queue=True,
            )

        with gr.Accordion("💡 Tips & Guidelines", open=False):
            gr.Markdown("""
### Model picks
- FLUX.1-dev: Best quality (20–30 steps, guidance ~7)
- FLUX.1-schnell: Fastest (4–8 steps, guidance ~2)
- SDXL Turbo: Very fast (4–6 steps, guidance ~2)
- RealVis XL V6 / Juggernaut XL: Realistic photos (18–35 steps, guidance ~7)

### Speed/Quality
- Use "Fast" for drafts, "Balanced" for most work, "High" for finals.
- 2k long side supported; streaming ZIP updates as images complete.

### Character Consistency
- Upload clear frontal reference images; use names in prompts.
- Identity strength 0.6–0.9 recommended.
""")
        gr.Markdown("Made for YouTube creators 🎬 | Streams results and supports very large batches.")
    return demo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_name", default="0.0.0.0")
    ap.add_argument("--server_port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    default_mode = os.environ.get("DEFAULT_QUALITY_MODE", "Balanced")
    demo = build_ui(default_mode)

    t = threading.Thread(target=_heartbeat, daemon=True)
    t.start()
    demo.queue()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)

if __name__ == "__main__":
    print(f"Using device: {DEVICE} | dtype: {DTYPE} | DirectML: {USE_DML}")
    main()
>>>>>>> 1757197612d8a511d0210c5335fadc2f8d5a8051
