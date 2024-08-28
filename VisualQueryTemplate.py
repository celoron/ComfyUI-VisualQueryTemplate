import re
from PIL import Image
import numpy as np
import time
from transformers import pipeline
import torch

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class VisualQueryTemplateNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (["Salesforce/blip-vqa-base", "Salesforce/blip-vqa-capfilt-large", "dandelin/vilt-b32-finetuned-vqa", "microsoft/git-large-vqav2"], ),
                "question": ("STRING", {"default": "{eye color} eyes, {hair style} {hair color} hair, {ethnicity} {gender}, {age number} years old, {facialhair}", "multiline": True, "dynamicPrompts": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)

    FUNCTION = "vqa_image"
    CATEGORY = "image"

    def vqa_image(self, images, model, question):
        start_time = time.time()

        device = 0 if torch.cuda.is_available() else -1
        vqa = pipeline(model=model, device=device)

        answers = []

        for image in images:
            pil_image = tensor2pil(image).convert("RGB")

            final_answer = question

            matches = re.findall(r'\{([^}]*)\}', question)

            for match in matches:

                match_answers = vqa(question=match, image=pil_image)

                print(match, match_answers)

                match_answer = match_answers[0]["answer"]

                final_answer = final_answer.replace("{"+match+"}", match_answer)

            
            answers.append(final_answer)

        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

        return (answers,)
