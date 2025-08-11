import json
import base64
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
)

MODEL_DIR = "OpenCUA-7B"

TEST_CASE_DIR = "./grounding_examples"
test_cases = [
    f'{TEST_CASE_DIR}/test0.json',
    f'{TEST_CASE_DIR}/test1.json',
    f'{TEST_CASE_DIR}/test2.json',
    f'{TEST_CASE_DIR}/test3.json',   
    f'{TEST_CASE_DIR}/test4.json',
    
]

def encode_image(path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_test_messages(case_file):
    with open(case_file) as f:
        info = json.load(f)
        img_name = info['image'].split('/')[-1]
        img_path = f'{TEST_CASE_DIR}/{img_name}'
        user_prompt = info['instruction']
    SYSTEM_PROMPT = (
        "You are a GUI agent. You are given a task and a screenshot of the screen. "
        "You need to perform a series of pyautogui actions to complete the task."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{encode_image(img_path)}"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    return messages, img_path

# load models
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True)
image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

# run cases.
for tc in test_cases:
    messages, img_path = get_test_messages(tc)
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    
    image = Image.open(img_path).convert('RGB')
    info = image_processor.preprocess(images=[image])
    pixel_values = torch.tensor(info['pixel_values']).to(dtype=torch.bfloat16, device=model.device)
    grid_thws = torch.tensor(info['image_grid_thw'])
    input_ids = torch.tensor([input_ids]).to(model.device)
    
    generated_ids = model.generate(
        input_ids, 
        pixel_values=pixel_values, 
        grid_thws=grid_thws,
        max_new_tokens=512,
        temperature=0
        )

    prompt_len = input_ids.shape[1]
    generated_ids = generated_ids[:, prompt_len:]
    output_text = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("="*100)
    print(output_text[0])
    print("="*100)