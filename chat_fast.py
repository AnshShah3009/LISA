import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA chat optimized for CPU and GPU")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)

def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set the appropriate dtype based on the device
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Create LISA model
    tokenizer = AutoTokenizer.from_pretrained(args.version, use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    model = LISAForCausalLM.from_pretrained(
        args.version, 
        low_cpu_mem_usage=True, 
        vision_tower=args.vision_tower, 
        seg_token_idx=args.seg_token_idx,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=device, dtype=dtype)

    model.eval()

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    while True:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt (describe what you want to segment in the image): ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print(f"File not found in {image_path}")
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]
        
        start_time = time.time()
        
        image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(device=device, dtype=dtype)

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(device=device, dtype=dtype)

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)

        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=tokenizer,
            )

        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("Model response:", text_output)

        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask.cpu().numpy()[0]
            pred_mask = pred_mask > 0

            # Save the binary mask
            mask_save_path = f"{args.vis_save_path}/{os.path.splitext(os.path.basename(image_path))[0]}_mask_{i}.png"
            cv2.imwrite(mask_save_path, pred_mask.astype(np.uint8) * 255)
            print(f"Binary mask saved to: {mask_save_path}")

            # Save the masked image
            masked_save_path = f"{args.vis_save_path}/{os.path.splitext(os.path.basename(image_path))[0]}_masked_img_{i}.jpg"
            masked_img = image_np.copy()
            masked_img[pred_mask] = (image_np * 0.5 + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5)[pred_mask]
            masked_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(masked_save_path, masked_img)
            print(f"Masked image saved to: {masked_save_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
