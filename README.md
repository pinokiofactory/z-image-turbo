# Z-Image-Turbo (Gradio)

Simple Gradio web UI for generating images with the Hugging Face model `Tongyi-MAI/Z-Image-Turbo`.

It uses Diffusers and PyTorch under the hood and exposes both an interactive web UI and a simple HTTP API for programmatic usage.

## Install & Run (via Pinokio)

1. Open this project in Pinokio.
2. From the sidebar, run **Install** (this will:
   - create a virtual environment in `env/`,
   - install PyTorch via `torch.js`,
   - install Python dependencies from `app/requirements.txt` using `uv`).
3. After installation finishes, run **Start**.
4. When the Gradio server comes up, Pinokio will capture the web UI URL and show an **Open Web UI** tab.
5. Use **Reset** to delete the `env/` virtual environment if you want a fresh reinstall.
6. Use **Update** to pull the latest changes from git (if this repo is tracked remotely).

## How it Works

- `app/app.py` defines a Gradio `Interface` named `demo` that:
  - loads the `Tongyi-MAI/Z-Image-Turbo` model via Diffusers,
  - exposes a `generate_image` function for text-to-image generation,
  - registers an API endpoint named `generate`.
- `start.js` launches the app with `gradio app.py` from the `app/` folder.
- The Pinokio `start.js` script watches for the first `http://...` URL printed by Gradio, then stores it as a local variable `url` via the `local.set` API.
- `pinokio.js` reads this `url` and shows an **Open Web UI** tab that forwards you directly to the running Gradio app.

## HTTP API

Assume the app is running and Pinokio shows a URL like:

- `http://127.0.0.1:7860`

Gradio exposes an HTTP endpoint for the `generate` API at:

- `POST http://127.0.0.1:7860/run/generate`

The payload format is a JSON object with an `data` array corresponding to the Gradio inputs:

1. `prompt` (string)
2. `negative_prompt` (string, can be empty)
3. `steps` (int)
4. `guidance_scale` (float)
5. `seed` (int, 0 for random)
6. `width` (int)
7. `height` (int)

### Curl

```bash
curl -X POST "http://127.0.0.1:7860/run/generate" \
  -H "Content-Type: application/json" \
  -o output.png \
  -d '{
    "data": [
      "A cozy cabin in the snowy mountains at sunset",
      "",
      25,
      7.5,
      0,
      512,
      512
    ]
  }'
```

This saves the generated image as `output.png`.

### Python

```python
import base64
import io

import requests
from PIL import Image

url = "http://127.0.0.1:7860/run/generate"

payload = {
    "data": [
        "A cozy cabin in the snowy mountains at sunset",
        "",
        25,
        7.5,
        0,
        512,
        512,
    ]
}

resp = requests.post(url, json=payload)
resp.raise_for_status()

image_b64 = resp.json()["data"][0].split(",")[1]
image_bytes = base64.b64decode(image_b64)
img = Image.open(io.BytesIO(image_bytes))
img.save("output.png")
```

### JavaScript

```js
const url = "http://127.0.0.1:7860/run/generate";

async function generate() {
  const payload = {
    data: [
      "A cozy cabin in the snowy mountains at sunset",
      "",
      25,
      7.5,
      0,
      512,
      512,
    ],
  };

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }

  const json = await res.json();
  const imageDataUrl = json.data[0];
  const img = new Image();
  img.src = imageDataUrl;
  document.body.appendChild(img);
}

generate().catch(console.error);
```

