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

import json, csv, io, sys, os
import ast
import pandas as pd

sys.path.append(os.path.join('/home/jex451', 'UQ/'))
from model_utils.base import *
from scripts.uncertainty_score import * 


# --- Launch Model --- 
model_path='/n/scratch/users/j/jex451/llava-med-v1.5-mistral-7b'
model_base='/n/scratch/users/j/jex451/llava-med-v1.5-mistral-7b'
device='cuda'
max_new_tokens=512
load_8bit=False
load_4bit=False

disable_torch_init()
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device=device)


# --- Define hyperparams ---

# TODO: change these. 
temperature=0.5
number_samples = 1000

input_csv = "/home/jex451/data/medversa/test_mimic_processed.csv" 
pre_path = "/n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/raw_jpg/files/"
inp_prompt = "You are an AI assistant specialized in biomedical topics. Describe the given chest X-ray images in detail."
result_csv_path = "/home/jex451/UQ/outputs/llava_med/inferences/inference_1.csv"
logits_file_path = "/home/jex451/UQ/outputs/llava_med/logits/logits_1.csv"
result_uq_path = "/home/jex451/UQ/outputs/llava_med/uq_scores/scores_1.csv"


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
        
        write_output_tokens(logits_file_path, output_ids[0])
    
    # reset stdout
    output = sys.stdout.getvalue()
    sys.stdout = old_stdout
    return output


if __name__ == "__main__":

    assert not os.path.exists(result_csv_path)
    if result_uq_path: assert not os.path.exists(result_uq_path)

    # Read from a csv file
    input_csv = pd.read_csv(input_csv)
    f = open(result_csv_path, 'w')
    writer_f = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    f.write("study_id,subject_id,target\n")
    f.flush()

    if result_uq_path:
        f_uq = open(result_uq_path, 'w')
        writer_uq = csv.writer(f_uq, quoting=csv.QUOTE_MINIMAL)
        f_uq.write("data_id,study_id,subject_id,max_prob,max_prob_sentence,max_prob_token,avg_prob,avg_prob_sentence,max_entropy,max_entropy_sentence,max_entropy_token,avg_entropy,avg_entropy_sentence,token_record\n")

    # loop through each row
    for index, row in input_csv.iterrows():
        if (index < number_samples):
            if (index % 20 == 0):
                print("Index is ", index)
            study_id = row['study_id']
            subject_id = row['subject_id']
            indication = row['indication']
            list_jpgs = ast.literal_eval(row['list_jpgs'])
            images_path = []

            for dicom_id in list_jpgs:
                image_path = pre_path + "p{}/p{}/s{}/{}.jpg".format(str(subject_id)[:2], subject_id, study_id, dicom_id)
                images_path.append(image_path)
            
            # Do inference
            
            output_text = inference(images_path)
            row_output = [study_id, subject_id, output_text]
            writer_f.writerow(row_output)
            f.flush()

            if result_uq_path:
                # Calculate uq scores. 
                uq = get_scores(logits_file_path, dot_token=28723, tokenizer=tokenizer)
                row = [index, study_id, subject_id]
                row.extend(uq)
                writer_uq.writerow(row)
                f_uq.flush()
    f.close()
    if result_uq_path:
        f_uq.close()

    

