import torch
from diffusers.utils import check_min_version, load_image

from flux_controlnet.controlnet_flux import FluxControlNetModel
from flux_controlnet.pipeline_flux_controlnet_inpaint import (
    FluxControlNetInpaintingPipeline,
)
from flux_controlnet.transformer_flux import FluxTransformer2DModel

check_min_version("0.30.2")


# Build pipeline
def build():
    device = (
        "mps"
        if torch.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    controlnet = FluxControlNetModel.from_pretrained(
        "alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha",
        # torch_dtype=torch.bfloat16,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        # torch_dtype=torch.bfloat16,
    )
    pipe = FluxControlNetInpaintingPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        controlnet=controlnet,
        transformer=transformer,
        # torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.transformer.to(torch.bfloat16)
    pipe.controlnet.to(torch.bfloat16)
    return pipe


def run(pipe, image_path, mask_path, prompt):
    # Load image and mask
    size = (768, 768)
    image = load_image(image_path).convert("RGB").resize(size)
    mask = load_image(mask_path).convert("RGB").resize(size)
    generator = torch.Generator(device="cuda").manual_seed(24)

    # Inpaint
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
    return result


def main():
    image_path = ("~/Documents/Workspace/Presti/Meubles-Perso/canape-1.jpg",)
    mask_path = ("~/Documents/Workspace/Presti/Meubles-Perso/canape-1-mask-1.jpg",)
    prompt = (
        'a person wearing a white shoe, carrying a white bucket with text "FLUX" on it'
    )
    pipe = build()

    result = run(pipe, image_path, mask_path, prompt)
    result.save("flux_inpaint.png")


if __name__ == "__main__":
    main()
