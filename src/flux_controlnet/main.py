import time
from pathlib import Path

import torch
from diffusers.utils import check_min_version, load_image

from flux_controlnet.controlnet_flux import FluxControlNetModel
from flux_controlnet.pipeline_flux_controlnet_inpaint import (
    FluxControlNetInpaintingPipeline,
)
from flux_controlnet.transformer_flux import FluxTransformer2DModel

check_min_version("0.30.2")

DEVICE = (
    "mps"
    if torch.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)

# TORCH_DTYPE: str = torch.bfloat16
TORCH_DTYPE: str = torch.float16


# Build pipeline
def build():
    print("Loading models")
    top = time.time()
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
        # "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta",
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        torch_dtype=TORCH_DTYPE,
    ).to(DEVICE)
    # pipe.transformer.to(torch.bfloat16)
    # pipe.controlnet.to(torch.bfloat16)
    print(f"models loaded in {time.time() - top:.2f}s")
    return pipe


def run(
    pipe: FluxControlNetInpaintingPipeline,
    image_path: Path,
    mask_path: Path,
    prompt: str,
):
    # Load image and mask
    # size = (768, 768)
    size = (512, 512)
    image = load_image(str(image_path)).convert("RGB").resize(size)
    mask = load_image(str(mask_path)).convert("RGB").resize(size)
    generator = torch.Generator(device=DEVICE).manual_seed(24)

    # Inpaint
    print("Running inpainting")
    top = time.time()
    result = pipe(
        prompt=prompt,
        height=size[1],
        width=size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt="",
        true_guidance_scale=1.0,  # default: 3.5 for alpha and 1.0 for beta
    ).images[0]
    print(f"Inpainting done in {time.time() - top:.2f}s")
    return result


def main():
    image_path = Path("/home/ubuntu/code/DATA/TestSet/1-a-astaged.png").expanduser()
    mask_path = Path(
        "/home/ubuntu/code/DATA/TestSet/1-a-bmasksquare-painting.png"
    ).expanduser()
    assert image_path.exists(), f"Image not found: {image_path}"
    assert mask_path.exists(), f"Mask not found: {mask_path}"
    prompt = "van gogh starry night painting"
    pipe = build()

    result = run(pipe, image_path, mask_path, prompt)
    result.save("flux_inpaint.png")


if __name__ == "__main__":
    main()
