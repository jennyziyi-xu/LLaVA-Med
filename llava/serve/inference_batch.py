import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from io import BytesIO
from transformers import TextStreamer

import json
import io
import sys
import ast


# --- Launch Model --- 
model_path='/n/scratch/users/j/jex451/llava-med-v1.5-mistral-7b'
model_base='/n/scratch/users/j/jex451/llava-med-v1.5-mistral-7b'
device='cuda:1'
max_new_tokens=512
load_8bit=False
load_4bit=False

disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)


# --- Define hyperparams ---

# TODO: change these. 
temperature=0
number_samples = 1

input_csv = "/home/jex451/data/medversa/test_mimic_processed.csv" 
pre_path = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files/"
inp_prompt = "You are an AI assistant specialized in biomedical topics. Describe the given chest X-ray images in detail."


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def inference(images_path):
    conv_mode = "mistral_instruct"
    conv = conv_templates[conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    number_images = len(images_path)
    images = [load_image(image) for image in images_path]

    image_tensor = process_images(images, image_processor, model.config)

    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    if images is not None:
        # first message
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp_prompt
        else:
            inp = DEFAULT_IMAGE_TOKEN * number_images + '\n' + inp_prompt
        conv.append_message(conv.roles[0], inp)
        images = None
    else:
        print("images is None. Error.")
        exit()
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print("prompt",prompt)


    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # direct stdout
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0.001 else False,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            use_cache=True,
            # stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.eos_token_id)
    
    # reset stdout
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return output


if __name__ == "__main__":
    # images_path = [pre_path + "p{}/p{}/s{}/{}.jpg".format('10', '10274145', '59166131', '29ab48f7-15a14464-5b7c1cc3-3ba3aa97-64ebc637'),
    # pre_path + "p{}/p{}/s{}/{}.jpg".format('10', '10274145', '59166131', '2cc38dd6-d1f5970f-055155bc-e9e8fccd-8ec98168')]
    images_path = [pre_path + "p{}/p{}/s{}/{}.jpg".format('10', '10274145', '59166131', '29ab48f7-15a14464-5b7c1cc3-3ba3aa97-64ebc637')]
    output = inference(images_path)
    print(output)

