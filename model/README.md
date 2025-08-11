---
license: mit
---


<h1 style="
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:48px;
  font-weight:700;
  line-height:1.25;
  text-align:center;
  margin:0 0 24px;">
  OpenCUA Models
</h1>

<div style="
  display:flex;
  justify-content:center;
  gap:12px;
  flex-wrap:wrap;
  margin-bottom:28px;">
    
  <a href="https://opencua.xlang.ai/" style="display:inline-block;padding:8px 24px;background:#2b2b2b;color:#ffffff;border-radius:36px;text-decoration:none;font-weight:600;font-size:16px;">🌐 Website</a>
  <a href="https://huggingface.co/xlangai/OpenCUA-7B" style="display:inline-block;padding:8px 24px;background:#2b2b2b;color:#ffffff;border-radius:36px;text-decoration:none;font-weight:600;font-size:16px;">🤖 OpenCUA-7B</a>
  <a href="https://huggingface.co/xlangai/OpenCUA-32B" style="display:inline-block;padding:8px 24px;background:#2b2b2b;color:#ffffff;border-radius:36px;text-decoration:none;font-weight:600;font-size:16px;">🚀 OpenCUA-32B</a>
  <a href="https://huggingface.co/spaces/xlangai/OpenCUA-demo" style="display:inline-block;padding:8px 24px;background:#2b2b2b;color:#ffffff;border-radius:36px;text-decoration:none;font-weight:600;font-size:16px;">🤗 Hugging Face Demo</a>


</div>

<div style="max-width:900px;margin:0 auto;">



# 1. Introduction
<div style="
  max-width: 880px;              /* 可按需调节整体宽度 */
  margin: 0 auto;               /* 居中容器 */
  text-align: justify;          /* 关键：两端对齐 */
  text-justify: inter-word;     /* 优化英文对齐效果 */
  line-height: 1.6;">
  
OpenCUA models (OpenCUA-7B and OpenCUA-32B) are end-to-end computer-use foundation models than can produce executable actions in the computer environments.  They are based on the weights of Qwen2.5-VL-7B-Instruction and Qwen2.5-VL-32B-Instruction. 
They demonstrate superior performance across CUA benchmarks. In particular, <b>OpenCUA-32B</b> achieves an average success rate of **34.8%** on [OSWorld-Verified](https://os-world.github.io/), 
establishing a new state-of-the-art (SOTA) among open-source models and surpassing OpenAI CUA (GPT-4o). Both models also have strong grounding performance, OpenCUA-32B achieves 59.6% on [OSWorld-G](https://osworld-grounding.github.io/) and 55.3% on [Screenspot-Pro](https://arxiv.org/abs/2504.07981).
</div>

### Key Features

- **Superior Computer-Use Capablity**: Able to execute multi-step computer-use actions with effective planning and reasoning
- **Multi-OS Support**: Trained on demonstrations across Ubuntu, Windows, and macOS
- **Visual Grounding**: Strong GUI element recognition and spatial reasoning capabilities
- **Multi-Image Context**: Processes up to 3 screenshot history for better context understanding
- **Reflective Reasoning**: Enhanced with reflective long Chain-of-Thought that identifies errors and provides corrective reasoning


## 2. Performance

### Online Agent Evaluation
OpenCUA models achieves strong performance on **[OSWorld-Verified](https://os-world.github.io/)**. 
OPENCUA-32B achieves the best performance among all open-source models with an average success rate of 34.8%, outperforming prior baselines by large margins. 
It also closes the gap to proprietary Claude models.
<div align="center">

| **Model**                        | **15 Steps** | **50 Steps** | **100 Steps** |
|-------------------------------|:--------:|:--------:|:---------:|
| **Proprietary**               |          |          |           |
| OpenAI CUA                    | 26.0     | 31.3     | 31.4      |
| Seed 1.5-VL                   | 27.9     | —        | 34.1      |
| Claude 3.7 Sonnet             | 27.1     | 35.8     | 35.9      |
| Claude 4 Sonnet               | 31.2     | 43.9     | 41.5      |
| **Open-Source**               |          |          |           |
| Qwen 2.5-VL-32B-Instruct       | 3.0      | —        | 3.9       |
| Qwen 2.5-VL-72B-Instruct       | 4.4      | —        | 5.0       |
| Kimi-VL-A3B                   | 9.7      | —        | 10.3      |
| UI-TARS-72B-DPO               | 24.0     | 25.8     | 27.1      |
| UI-TARS-1.5-7B                | 24.5     | 27.3     | 27.4      |
| OpenCUA-7B *(Ours)*           | 24.3     | 27.9     | 26.6      |
| **OpenCUA-32B *(Ours)***      | **29.7** | **34.1** | **34.8**  |
</div>

*OpenCUA scores are the mean of 3 independent runs.*

### GUI Grounding Performance
<div align="center">

| **Model** | **OSWorld-G** | **ScreenSpot-V2** | **ScreenSpot-Pro** |
|-------|-----------|---------------|----------------|
| Qwen2.5-VL-7B | 31.4 | 88.8 | 27.6 |  
| Qwen2.5-VL-32B | 46.5 | 87.0 | 39.4 |
| UI-TARS-72B | 57.1 | 90.3 | 38.1 |
| **OpenCUA-A3B** | 48.6 | 91.4 | 28.5 |
| **OpenCUA-7B** | 45.7 | 88.5 | 23.7 |
| **OpenCUA-2.5-7B** | 55.3 | 92.3 | 50.0 |
| **OpenCUA-2.5-32B** | **59.6** | **93.4** | **55.3** |
</div>

### AgentNetBench (Offline Evaluation)
<div align="center">

| **Model** | **Coordinate Actions** | **Content Actions** | **Function Actions** | **Average** |
|-------|-------------------|-----------------|------------------|---------|
| Qwen2.5-VL-7B | 50.7 | 40.8 | 3.1 | 48.0 |
| Qwen2.5-VL-32B | 66.6 | 47.2 | 41.5 | 64.8 |
| Qwen2.5-VL-72B | 67.2 | 52.6 | 50.5 | 67.0 |
| OpenAI CUA          | 71.7 | 57.3 | **80.0** | 73.1 |
| **OpenCUA-2.5-7B**  | 79.0 | 62.0 | 44.3 | 75.2 |
| **OpenCUA-2.5-32B** | **81.9** | 66.1 | 55.7 | **79.1** |
</div>

#  🚀 Quick Start
<div style="border-left: 6px solid #f28c28; background: #fff8e6; padding: 12px 16px; margin: 16px 0;">
  <strong>⚠️ Important for Qwen-based Models (OpenCUA-7B, OpenCUA-32B):</strong>
  
  To align with our training infrastructure, we have modified the model in two places:
  <ul style="margin-top: 8px;">
    <li>1. Multimodal Rotary Position Embedding (M-RoPE) has been replaced with 1D RoPE</strong>.</li>
    <li>2. Using the same Tokenizer and ChatTemplate as Kimi-VL.</li>
    <li>Do not use the default transformers and vllm classes to load the model. Tokenizer and Chat Template should be aligned if training the models.</li>
  </ul>
</div>


## Installation & Download

First, install the required transformers dependencies:

```bash
conda create -n opencua python=3.10
conda activate opencua
pip install -r requirement.txt
```

Download the model weight from huggingface:
```bash
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="xlangai/OpenCUA-7B",
    local_dir="OpenCUA-7B",                
    local_dir_use_symlinks=False  
)
```

## 🎯 GUI Grounding 

The following code demonstrates how to use OpenCUA models for GUI grounding tasks:

```python
import base64
import torch
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import json

def encode_image(image_path: str) -> str:
    """Encode image to base64 string for model input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def load_opencua_model(model_path: str):
    """Load OpenCUA model, tokenizer, and image processor."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype="auto", 
        device_map="auto", 
        trust_remote_code=True
    )
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    return model, tokenizer, image_processor

def create_grounding_messages(image_path: str, instruction: str):
    """Create chat messages for GUI grounding task."""
    system_prompt = (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{encode_image(image_path)}"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    return messages

def run_inference(model, tokenizer, image_processor, messages, image_path):
    """Run inference on the model."""
    # Prepare text input
    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    input_ids = torch.tensor([input_ids]).to(model.device)
    
    # Prepare image input  
    image = Image.open(image_path).convert('RGB')
    image_info = image_processor.preprocess(images=[image])
    pixel_values = torch.tensor(image_info['pixel_values']).to(
        dtype=torch.bfloat16, device=model.device
    )
    grid_thws = torch.tensor(image_info['image_grid_thw'])
    
    # Generate response
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            max_new_tokens=512,
            temperature=0
        )
    
    # Decode output
    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

# Example usage
model_path = "OpenCUA/OpenCUA-7B"  # or other model variants
image_path = "screenshot.png"
instruction = "Click on the submit button"

# Load model
model, tokenizer, image_processor = load_opencua_model(model_path)

# Create messages and run inference
messages = create_grounding_messages(image_path, instruction)
result = run_inference(model, tokenizer, image_processor, messages, image_path)

print("Model output:", result)
```

<div style="border-left: 6px solid #9ca3af; background: #f5f5f5; padding: 12px 16px; margin: 16px 0;">
  <em>Expected result: ```python\npyautogui.click(x=1443, y=343)\n```
</div>

You can also run the five grounding examples in [OpenCUA/model/inference/huggingface_inference.py](./inference/huggingface_inference.py):
``` 
cd ./model/inference/
python huggingface_inference.py
```

## 🖥️ Computer Use Agent
**[OpenCUAAgent](https://github.com/xlang-ai/OSWorld/blob/main/mm_agents/opencua_agent.py)** is developed in the [OSWorld](https://github.com/xlang-ai/OSWorld) environment based on OpenCUA models. It iteratively perceives the environment via screenshots, produces reflective long CoT as inner monologue, and predicts the next action to be executed. OpenCUAAgent uses 3 images in total and L2 CoT format in default.

Command for running OpenCUA-7B and OpenCUA-32B in OSWorld:
```
    python run_multienv_opencua.py \
        --headless \
        --observation_type screenshot \
        --model OpenCUA-32B \
        --result_dir ./results --test_all_meta_path evaluation_examples/test_all_no_gdrive.json \
        --max_steps 100 \
        --num_envs 30  \
        --coordinate_type qwen25
```
<div style="border-left: 6px solid #9ca3af; background: #f5f5f5; padding: 12px 16px; margin: 16px 0;">
  <em>Currently we only supports huggingface inference. We are implementing the vLLM supports of OpenCUA models. Please stay tuned.
</div>

## Important Notes on Coordinate Systems
<div style="border-left: 6px solid #9ca3af; background: #f5f5f5; padding: 12px 16px; margin: 16px 0;">
  <ul style="margin: 0;">
    <li><strong><code>OpenCUA/OpenCUA-A3B</code></strong> – Relative coordinates <em>(not supported in this code)</em></li>
    <li><strong><code>OpenCUA/OpenCUA-Qwen2-7B</code></strong> – Relative coordinates</li>
    <li><strong><code>OpenCUA/OpenCUA-7B</code></strong> – Absolute coordinates</li>
    <li><strong><code>OpenCUA/OpenCUA-32B</code></strong> – Absolute coordinates</li>
  </ul>
</div>

**OpenCUA models use different coordinate systems depending on the base model:**

- **OpenCUA-Qwen2-7B**: Outputs **relative coordinates** (0.0 to 1.0 range)
  ```python
  # Example output: pyautogui.click(x=0.5, y=0.3)
  # x=0.5 means 50% from left edge, y=0.3 means 30% from top edge
  
  # Convert to absolute coordinates:
  def qwen2_relative_to_absolute(rel_x, rel_y, original_width, original_height):
      abs_x = int(rel_x * original_width)
      abs_y = int(rel_y * original_height)
      return abs_x, abs_y
  ```

- **OpenCUA-7B and OpenCUA-32B** (Qwen2.5-based): Output **absolute coordinates** after smart resize
  ```python
  # Example output: pyautogui.click(x=960, y=324)  
  # These are coordinates on the smart-resized image, not the original image
  
  # Convert to original image coordinates:
  # Please refer to the smart_resize function in: https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L55
  def qwen25_smart_resize_to_absolute(model_x, model_y, original_width, original_height):
      # First, calculate the smart-resized dimensions
      resized_height, resized_width = smart_resize(original_height, original_width, factor = 28, min_pixels = 3136, max_pixels = 12845056)
      
      # Convert model output to relative coordinates on original image
      rel_x = model_x / resized_width
      rel_y = model_y / resized_height
      
      # Then convert to absolute coordinates on original image
      abs_x = int(rel_x * original_width)
      abs_y = int(rel_y * original_height)
      return abs_x, abs_y
  ```

<div style="border-left: 6px solid #9ca3af; background: #f5f5f5; padding: 12px 16px; margin: 16px 0;">
  <strong>Understanding Smart Resize for Qwen2.5-based Models:</strong>
  <p style="margin: 8px 0 0;">
    The Qwen2.5-VL models use a “smart resize” preprocessing that maintains aspect ratio while fitting within pixel constraints.
    For coordinate conversion, you need the smart resize function from the
    <a href="https://github.com/QwenLM/Qwen2.5-VL/blob/d2240f11656bfe404b9ba56db4e51cd09f522ff1/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L60">
      official Qwen2.5-VL implementation</a>.
  </p>
</div>

# TODO
## vLLM Support
We are actively working with the vLLM team to add support for OpenCUA models.

**Workaround:** For now, please use the standard transformers library as shown in the examples above. We will update this section once vLLM support becomes available.

## Training Code
OpenCUA models are developed based on the training infrastructure of Kimi Team. We are developting the training pipeline based on the open-source infrastructure as well.

## License

This project is licensed under the MIT License - see the LICENSE file in the root folder for details.

## Research Use and Disclaimer

OpenCUA models are intended for **research and educational purposes only**. 

### Prohibited Uses
- The model may **not** be used for any purpose or activity that violates applicable laws or regulations in any jurisdiction
- Use for illegal, unethical, or harmful activities is strictly prohibited

### Disclaimer
- The authors, contributors, and copyright holders are **not responsible** for any illegal, unethical, or harmful use of the Software, nor for any direct or indirect damages resulting from such use
- Use of the "OpenCUA" name, logo, or trademarks does **not** imply any endorsement or affiliation unless separate written permission is obtained
- Users are solely responsible for ensuring their use complies with applicable laws and regulations

## Citation

If you use OpenCUA models in your research, please cite our work:

```bibtex
@article{OpenCUA2025, 
  title={OpenCUA: Open Foundations for Computer-Use Agents}, 
  author={Wang, Xinyuan and Wang, Bowen and Lu, Dunjie and Yang, Junlin and Xie, Tianbao and Wang, Junli and Deng, Jiaqi and Guo, Xiaole and Xu, Yiheng and Wu, Chen Henry and Shen, Zhennan and Li, Zhuokai and Li, Ryan and Li, Xiaochuan and Chen, Junda and Zheng, Boyuan and Li, Peihang and Lei, Fangyu and Cao, Ruisheng and Fu, Yeqiao and Shin, Dongchan and Shin, Martin and Hu, Jiarui and Wang, Yuyan and Chen, Jixuan and Ye, Yuxiao and Zhang, Danyang and Wang, Yipu and Wang, Heng and Yang, Diyi and Zhong, Victor and Charles, Y. and Yang, Zhilin and Yu, Tao}, 
  year={2025}, 
  url={https://opencua.xlang.ai/} 
}
```

</div>
