import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.distributed as dist
from transformers import BertTokenizer, AutoImageProcessor
from PIL import Image
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim, inter_dim, out_dim):
        super(MLP, self).__init__()
        self.hidden_1 = nn.Linear(in_dim, inter_dim)
        self.act = nn.GELU()
        self.hidden_2 = nn.Linear(inter_dim, out_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.act(self.hidden_1(x))
        x = self.dropout(x)
        return self.hidden_2(x)


class LightMedVLM(pl.LightningModule):
    def __init__(
        self,
        vision_model: str = "microsoft/swin-base-patch4-window7-224",
        llm_model: str = "Qwen/Qwen3-0.6B",

        # For training setup
        vis_use_lora: bool = False,
        vis_r: int = 8,
        vis_alpha: int = 16,
        freeze_vm: bool = False,
        llm_use_lora: bool = False,
        llm_r: int = 8,
        llm_alpha: int = 16,
        lora_dropout: float = 0.1,
        low_resource: bool = False,
        max_length: int = 256
    ):
        super().__init__()
        self.vision_model = vision_model
        self.llm_model = llm_model
        self.vis_use_lora = vis_use_lora
        self.vis_r = vis_r
        self.vis_alpha = vis_alpha
        self.freeze_vm = freeze_vm
        self.llm_use_lora = llm_use_lora
        self.llm_r = llm_r
        self.llm_alpha = llm_alpha
        self.lora_dropout = lora_dropout
        self.low_resource = low_resource
        self.max_length = max_length
        
        # Vision encoder setup
        print(f'Loading vision encoder: {self.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(self.vision_model)
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(self.vision_model)
        if self.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=self.vis_r,
                lora_alpha=self.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=self.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif self.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder: {self.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder: {self.vision_model} -- Done')

        # LLM model setup
        print('Loading LLM model')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model, 
            use_fast=False,
            trust_remote_code=True
        )
        print(f"BOS token ID: {self.tokenizer.bos_token_id}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")
        if self.low_resource:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.llm_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        if self.llm_use_lora:
            self.embed_tokens = self.model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=self.llm_r, 
                lora_alpha=self.llm_alpha, 
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            print('Loading LLM LoRA Done')         
        else:
            self.embed_tokens = self.model.get_input_embeddings()
            for name, param in self.model.named_parameters():
                param.requires_grad = False
            print('Loading LLM Done')

        # Projector setup
        self.proj = MLP(
            in_dim=self.visual_encoder.num_features,
            inter_dim=2048,
            out_dim=self.model.config.hidden_size
        )
        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)

        # System prompt setup
        self.end_sym = "<|im_end|>"
        self.report_prompt = "<|im_start|>system\nYou are a professional radiologist. Please generate a comprehensive and detailed diagnosis report for this chest xray image.<|im_end|>\n<|im_start|>user\nThe chest X-ray image shows:"
        self.prompt = "<|im_start|>system\nYou are a professional radiologist. Please generate a comprehensive and detailed diagnosis report for this chest xray image.<|im_end|>\n<|im_start|>user\nThe chest X-ray image shows:"
        
    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            # Swin transformer
            visual_outputs = self.visual_encoder(image)
            image_embed = visual_outputs['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)

        inputs = self.proj(image_embeds)
        atts = torch.ones(inputs.size()[:-1], dtype=torch.long).to(image.device)
        return inputs, atts

    def prompt_wrap_report(self, img_embeds, atts_img):
        """
        Wrap image embeddings with Qwen-style prompt.
        Format: {prompt_before} <image> {prompt_after}
        """
        # Full prompt with image placeholder
        full_prompt = f"{self.report_prompt} <image><|im_end|>\n<|im_start|>assistant\n"
        
        batch_size = img_embeds.shape[0]
        
        # Split prompt at image placeholder
        p_before, p_after = full_prompt.split('<image>')
        
        # Tokenize prompt parts (no special tokens added)
        p_before_tokens = self.tokenizer(
            p_before,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)
        
        p_after_tokens = self.tokenizer(
            p_after,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)
        
        # Get embeddings using frozen embed_tokens
        with torch.no_grad():
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        
        # Concatenate: [prompt_before] + [image] + [prompt_after]
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        
        # Create attention mask for entire sequence
        wrapped_atts_img = torch.ones(
            batch_size, 
            wrapped_img_embeds.shape[1], 
            device=img_embeds.device, 
            dtype=atts_img.dtype
        )
        
        return wrapped_img_embeds, wrapped_atts_img

    def prompt_wrap(self, img_embeds, atts_img):
        # Full prompt with image placeholder
        full_prompt = f"{self.prompt} <image><|im_end|>\n<|im_start|>assistant\n"
        
        batch_size = img_embeds.shape[0]
        
        # Split prompt at image placeholder
        p_before, p_after = full_prompt.split('<image>')
        
        # Tokenize prompt parts (no special tokens added)
        p_before_tokens = self.tokenizer(
            p_before,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)
        
        p_after_tokens = self.tokenizer(
            p_after,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)
        
        # Get embeddings using frozen embed_tokens
        with torch.no_grad():
            p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        
        # Concatenate: [prompt_before] + [image] + [prompt_after]
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        
        # Create attention mask for entire sequence
        wrapped_atts_img = torch.ones(
            batch_size, 
            wrapped_img_embeds.shape[1], 
            device=img_embeds.device, 
            dtype=atts_img.dtype
        )
        
        return wrapped_img_embeds, wrapped_atts_img
    def forward(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        # Wrap image with prompt
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text"]]

        # Tokenize target text
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        # Create labels: mask prompt+image tokens with -100, keep text tokens
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        # Create empty targets for prompt+image tokens
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(image[0].device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # Get text embeddings
        with torch.no_grad():
            to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        
        # Concatenate all embeddings: [prompt+image] + [text]
        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        # Forward through LLM
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        all_loss = outputs.loss

        return {"loss": all_loss}

    def decode(self, output_token):
        """Decode output tokens to text."""
        # Remove special tokens at the beginning
        if len(output_token) > 0 and output_token[0] == self.tokenizer.pad_token_id:
            output_token = output_token[1:]
        if len(output_token) > 0 and output_token[0] == self.tokenizer.bos_token_id:
            output_token = output_token[1:]
        
        # Decode to text
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        
        # Split at end symbol and clean up
        output_text = output_text.split(self.end_sym)[0].strip()
        
        # Remove Qwen special tokens
        output_text = output_text.replace('<|im_start|>', '')
        output_text = output_text.replace('<|im_end|>', '')
        output_text = output_text.replace('<|endoftext|>', '')
        output_text = output_text.replace('<unk>', '')
        
        return output_text

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 
        
    @torch.no_grad()
    def inference_report(self, image_paths, 
                        beam_size=3, 
                        do_sample=False, 
                        min_new_tokens=10, 
                        max_new_tokens=120, 
                        repetition_penalty=2.0, 
                        length_penalty=2.0, 
                        temperature=0):
        """Generate text from images."""
        self.eval()

        images = []
        device = next(self.parameters()).device
        for image_path in image_paths:
            with Image.open(image_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                image = image.unsqueeze(0).to(device)
                images.append(image)


        dtype = self.model.dtype
        
        img_embeds, atts_img = self.encode_img(images)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds = img_embeds.to(dtype)
        img_embeds, atts_img = self.prompt_wrap_report(img_embeds, atts_img)

        outputs = self.model.generate(
            inputs_embeds=img_embeds,
            attention_mask=atts_img,
            num_beams=beam_size,
            do_sample=do_sample,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        out = [self.decode(i) for i in outputs]
        return out

    @torch.no_grad()
    def inference(self, image_paths, 
                        beam_size=3, 
                        do_sample=False, 
                        min_new_tokens=10, 
                        max_new_tokens=120, 
                        repetition_penalty=2.0, 
                        length_penalty=2.0, 
                        temperature=0):
        """Generate text from images."""
        self.eval()

        images = []
        device = next(self.parameters()).device
        for image_path in image_paths:
            with Image.open(image_path) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                image = image.unsqueeze(0).to(device)
                images.append(image)


        dtype = self.model.dtype
        
        img_embeds, atts_img = self.encode_img(images)
        img_embeds = self.layer_norm(img_embeds)

        img_embeds = img_embeds.to(dtype)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        outputs = self.model.generate(
            inputs_embeds=img_embeds,
            attention_mask=atts_img,
            num_beams=beam_size,
            do_sample=do_sample,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        out = [self.decode(i) for i in outputs]
        return out