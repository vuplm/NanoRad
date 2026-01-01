from __future__ import annotations

import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM, SwinModel, AutoImageProcessor
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image
import numpy as np


class HybridVisionAbstractor(nn.Module):
    """
    The 'Spatially-Aware Semantic Compressor'.
    Combines CNN (for local texture/shape) with Attention (for semantic selection).
    """
    def __init__(self, visual_dim, llm_dim, num_queries=64, num_heads=8):
        super().__init__()
        self.num_queries = num_queries
        
        # 1. Spatial Block (CNN)
        self.conv_block = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU(),
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU()
        )
        
        # 2. Semantic Block (Dynamic Queries)
        self.vis_proj = nn.Linear(visual_dim, llm_dim)
        self.latents = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=llm_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=0.1
        )
        
        # 3. Processing Block (FFN)
        self.ln_q = nn.LayerNorm(llm_dim)
        self.ln_v = nn.LayerNorm(llm_dim)
        self.ln_out = nn.LayerNorm(llm_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4),
            nn.GELU(),
            nn.Linear(llm_dim * 4, llm_dim)
        )

    def inject_concept_init(self, concept_embeds):
        with torch.no_grad():
            num_concepts = concept_embeds.shape[0]
            self.latents[:, :num_concepts, :] = concept_embeds.unsqueeze(0)
        print(f">> INJECTED {num_concepts} DISEASE CONCEPTS INTO QUERIES 0-{num_concepts-1} <<")

    def forward(self, visual_features):
        # visual_features: (Batch, Seq_Len=49, Vis_Dim=1024)
        B, L, C = visual_features.shape
        H = W = int(L ** 0.5) 
        
        # Step A: Spatial Enhancement
        x_img = visual_features.permute(0, 2, 1).view(B, C, H, W)
        x_refined = self.conv_block(x_img)
        x_flat = x_refined.flatten(2).permute(0, 2, 1)
        
        # Step B: Dimension Projection
        x_vis = self.vis_proj(x_flat)
        
        # Step C: Semantic Selection
        queries = self.latents.repeat(B, 1, 1)
        q = self.ln_q(queries)
        k = v = self.ln_v(x_vis)
        
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        
        x = queries + attn_out
        x = self.ln_out(x)
        x = x + self.mlp(x)
        return x

class R2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.args = args

        # --- 1. Vision Encoder ---
        print(f'Loading vision encoder: {args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        self.visual_hidden_size = self.visual_encoder.config.hidden_size
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r, lora_alpha=args.vis_alpha, target_modules=["query", "value"],
                lora_dropout=args.lora_dropout, bias="none", modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)

        # --- 2. LLM ---
        print(f'Loading LLM model: {args.qwen_model}')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, use_fast=False, trust_remote_code=True)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )

        if args.llm_use_lora:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r,
                lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            
        self.embed_tokens = self.llama_model.get_input_embeddings()
        
        # --- 3. Connector (Hybrid) ---
        self.llm_hidden_size = self.llama_model.config.hidden_size
        self.disease_concepts = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
            "Lung Opacity", "No Finding", "Pleural Effusion", 
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        self.llama_proj = HybridVisionAbstractor(
            visual_dim=self.visual_hidden_size,
            llm_dim=self.llm_hidden_size,
            num_queries=64
        )
        
        # Inject Embeddings
        self._inject_diseases_into_connector()

        self.layer_norm = nn.LayerNorm(self.llm_hidden_size)
        self.end_sym = "<|im_end|>"
        self.prompt = "<|im_start|>system\nYou are a professional radiologist. Please generate a comprehensive and detailed diagnosis report for this chest xray image.<|im_end|>\n<|im_start|>user\nThe chest X-ray image shows:"

    def _inject_diseases_into_connector(self):
        print(f"Generating embeddings for {len(self.disease_concepts)} MIMIC-CXR concepts...")
        # Note: device mapping happens during forward/init, ensuring token access
        concept_vectors = []
        for concept in self.disease_concepts:
            ids = self.llama_tokenizer(concept, return_tensors="pt", add_special_tokens=False).input_ids
            # We move ids to cpu first, computation happens where embeddings are
            with torch.no_grad():
                # Temporarily move ids to model device if needed, but embeddings are usually on CPU during init
                embeds = self.embed_tokens(ids) 
            concept_vec = embeds.mean(dim=1).squeeze(0)
            concept_vectors.append(concept_vec)
        concept_stack = torch.stack(concept_vectors)
        self.llama_proj.inject_concept_init(concept_stack)

    def encode_img(self, images):
        # images: (Batch, C, H, W)
        images = images.to(self.device)
        visual_outputs = self.visual_encoder(images)
        image_embeds = visual_outputs.last_hidden_state
        
        # Connector
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.shape[:2], dtype=torch.long).to(self.device)
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        full_prompt = f"{self.prompt} <image><|im_end|>\n<|im_start|>assistant\n"
        batch_size = img_embeds.shape[0]
        p_before, p_after = full_prompt.split('<image>')
        
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)

        with torch.no_grad():
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = torch.ones(batch_size, wrapped_img_embeds.shape[1], device=img_embeds.device, dtype=atts_img.dtype)
        return wrapped_img_embeds, wrapped_atts_img

    def decode(self, output_token):
        if len(output_token) > 0 and output_token[0] == self.llama_tokenizer.pad_token_id:
            output_token = output_token[1:]
        if len(output_token) > 0 and output_token[0] == self.llama_tokenizer.bos_token_id:
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split(self.end_sym)[0].strip()
        for tag in ['<|im_start|>', '<|im_end|>', '<|endoftext|>', '<unk>']:
            output_text = output_text.replace(tag, '')
        return output_text

    def _parse_image(self, img_array):
        pixel_values = self.vit_feature_extractor(img_array, return_tensors="pt").pixel_values
        return pixel_values[0]

    @torch.no_grad()
    def inference(self, image_paths, beam_size=3, max_new_tokens=120, repetition_penalty=2.0, length_penalty=2.0):
        self.eval()
        images = []
        device = next(self.parameters()).device
        
        # Load and preprocess images
        for image_path in image_paths:
            if not os.path.exists(image_path):
                # Fallback for dummy image if path missing (optional)
                images.append(torch.zeros((3, 224, 224)))
                continue
                
            with Image.open(image_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        
        if len(images) == 0: return [""]
        
        # Stack images (Batching handled by outer loop usually, here we handle list)
        images = torch.stack(images).to(device)
        
        img_embeds, atts_img = self.encode_img(images)
        img_embeds = self.layer_norm(img_embeds)
        
        # Wrap prompt
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        outputs = self.llama_model.generate(
            inputs_embeds=img_embeds,
            attention_mask=atts_img,
            num_beams=beam_size,
            min_new_tokens=10,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id
        )
        
        return [self.decode(i) for i in outputs]


from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vuplm2004/lightmedvlm-mimic-phase2-1epochs-full",
    local_dir="lightmedvlm"
)


import argparse
from lightning.fabric.utilities.data import AttributeDict
torch.serialization.add_safe_globals([AttributeDict])

args = argparse.Namespace(
    vision_model="microsoft/swin-base-patch4-window7-224",
    qwen_model="Qwen/Qwen3-0.6B",
    vis_use_lora=False,
    vis_r=16,
    vis_alpha=32,
    lora_dropout=0.1,
    llm_use_lora=True,
    llm_r=8,
    llm_alpha=16,
    task="report"
)

ckpt_file = "/kaggle/working/lightmedvlm/checkpoints/checkpoint_epoch0_step10841_rougle_l0.3221699044935706_bleu0.135881_cider0.153468.pth"
model = R2GenGPT.load_from_checkpoint(ckpt_file, strict=False, args=args)
model = model.to("cuda")
model.eval()


test_img_iu = "/kaggle/input/mimic-700-images/iu_images/iu_images"
test_annot_iu = "/kaggle/input/mimic-700-images/test/test/report/iuxray_test.json"
out_iu = "iu_prediction_hybrid.json"


with open(test_annot_iu, "r") as f:
    test_data = json.load(f)
print("Số samples:", len(test_data))

results = []

for item in tqdm(test_data):
    img_paths = [os.path.join(test_img_iu, p) for p in item["image_path"]]
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = model.inference(img_paths)[0]

        
    results.append({
        "id": item["id"],
        "gt_report": item["report"],
        "pred_report": pred,
    })

with open(out_iu, "w") as f:
    json.dump(results, f, indent=4)


import re
