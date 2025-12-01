from typing import Optional

import gradio as gr
import torch
from diffusers import ZImagePipeline

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

_pipe: Optional[ZImagePipeline] = None


THEME_CSS = """
body, .gradio-container {
  background-color: #ffffff;
  color: #000000;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container button,
.gradio-container .prose * {
  background-color: #ffffff;
  color: #000000;
}
"""


def _get_device() -> str:
  if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return "mps"
  if torch.cuda.is_available():
    return "cuda"
  return "cpu"


def load_pipeline() -> ZImagePipeline:
  global _pipe
  if _pipe is not None:
    return _pipe

  device = _get_device()

  kwargs = {"low_cpu_mem_usage": False}
  if device == "cuda":
    kwargs["torch_dtype"] = torch.bfloat16

  pipe = ZImagePipeline.from_pretrained(
    MODEL_ID,
    **kwargs,
  )

  pipe.to(device)

  _pipe = pipe
  return pipe


def generate_image(
  prompt: str,
  steps: int,
  guidance_scale: float,
  seed: int,
  width: int,
  height: int,
):
  if not prompt:
    return None

  pipe = load_pipeline()

  generator = None
  if seed and int(seed) != 0:
    device = pipe.device if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(int(seed))

  result = pipe(
    prompt=prompt,
    height=int(height),
    width=int(width),
    num_inference_steps=int(steps),
    guidance_scale=float(guidance_scale),
    generator=generator,
  )
  image = result.images[0]
  return image


with gr.Blocks(css=THEME_CSS, title="Z-Image-Turbo") as demo:
  with gr.Row():
    output_image = gr.Image(label="Generated Image", show_label=True)

  with gr.Row():
    with gr.Column(scale=3):
      prompt = gr.Textbox(label="Prompt", value="a fuzzy cat")
      run_button = gr.Button("Generate", scale=0)
    with gr.Column(scale=2):
      steps = gr.Slider(1, 32, value=9, step=1, label="Steps")
      guidance_scale = gr.Slider(0, 10, value=0.0, step=0.1, label="Guidance Scale")
    with gr.Column(scale=2):
      seed = gr.Number(label="Seed (0 = random)", value=0, precision=0)
      width = gr.Slider(256, 1536, value=512, step=64, label="Width")
      height = gr.Slider(256, 1536, value=512, step=64, label="Height")

  def _ui_generate(prompt, steps, guidance_scale, seed, width, height):
    return generate_image(prompt, steps, guidance_scale, seed, width, height)

  gr.on(
    triggers=[run_button.click, prompt.submit],
    fn=_ui_generate,
    inputs=[prompt, steps, guidance_scale, seed, width, height],
    outputs=[output_image],
  )


if __name__ == "__main__":
  demo.launch()
