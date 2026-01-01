from lightning.pytorch.callbacks import ModelCheckpoint
from huggingface_hub import login
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import seed_everything
from pprint import pprint
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor

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
import argparse

# =============================================================================
# SOTA COMPONENT: Hybrid Connector (C-Abstractor + Dynamic Queries)
# =============================================================================


class HybridVisionAbstractor(nn.Module):
    """
    The 'Spatially-Aware Semantic Compressor'.
    Combines CNN (for local texture/shape) with Attention (for semantic selection).
    Optimized for Qwen-0.6B to provide high-signal, low-noise tokens.
    """

    def __init__(self, visual_dim, llm_dim, num_queries=64, num_heads=8):
        super().__init__()
        self.num_queries = num_queries

        # 1. Spatial Block (CNN)
        # Refine features using convolutions to capture medical textures (edges, opacities)
        self.conv_block = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU(),  # Modern activation (better than ReLU for vision)
            nn.Conv2d(visual_dim, visual_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(visual_dim),
            nn.SiLU()
        )

        # 2. Semantic Block (Dynamic Queries)
        # Project to LLM space
        self.vis_proj = nn.Linear(visual_dim, llm_dim)

        # Learnable Queries (The "Questions" the model asks the image)
        self.latents = nn.Parameter(torch.randn(1, num_queries, llm_dim))

        # Cross Attention: Queries attending to Conv-Refined Image Features
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
        """
        NOVELTY FUNCTION: Overwrites the first N random queries with 
        actual disease embeddings (Atelectasis, Cardiomegaly, etc.)
        """
        with torch.no_grad():
            num_concepts = concept_embeds.shape[0]
            # Copy disease vectors into the first slots of the learnable queries
            # Shape: (1, Num_Concepts, Dim)
            self.latents[:, :num_concepts, :] = concept_embeds.unsqueeze(0)

        print(
            f">> INJECTED {num_concepts} DISEASE CONCEPTS INTO QUERIES 0-{num_concepts-1} <<")

    def forward(self, visual_features):
        # visual_features: (Batch, Seq_Len=49, Vis_Dim=1024) from Swin
        B, L, C = visual_features.shape
        H = W = int(L ** 0.5)  # Assuming square input (7x7 for Swin Base)

        # --- Step A: Spatial Enhancement (CNN) ---
        # Reshape to Image format: (B, C, H, W)
        x_img = visual_features.permute(0, 2, 1).view(B, C, H, W)

        # Apply Convolutions
        x_refined = self.conv_block(x_img)

        # Flatten back to Sequence: (B, 49, C)
        x_flat = x_refined.flatten(2).permute(0, 2, 1)

        # --- Step B: Dimension Projection ---
        # Project to LLM size: (B, 49, LLM_Dim)
        x_vis = self.vis_proj(x_flat)

        # --- Step C: Semantic Selection (Attention) ---
        # Prepare Queries
        queries = self.latents.repeat(B, 1, 1)  # (B, 64, LLM_Dim)

        q = self.ln_q(queries)
        k = v = self.ln_v(x_vis)

        # The Attention Mechanism picks the relevant info from the Conv features
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)

        # Residual + Norm
        x = queries + attn_out
        x = self.ln_out(x)

        # FFN
        x = x + self.mlp(x)

        # Final Output: (Batch, 64, LLM_Dim)
        return x


class LightMedVLMForTraining(pl.LightningModule):
    """
    LightMedVLMForTraining model with Qwen 0.6B and Hybrid Abstractor.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        # ---------------- Vision Encoder (Swin) ----------------
        print(f'Loading vision encoder: {args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)

        # Determine Visual Hidden Size (Swin Base is 1024)
        self.visual_hidden_size = self.visual_encoder.config.hidden_size

        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(
                self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(
                f'Loading Frozen vision encoder: {args.vision_model} -- Done')

        # ---------------- LLM (Qwen) ----------------
        print('Loading Qwen model')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(
            args.qwen_model,
            use_fast=False,
            trust_remote_code=True
        )

        # Safety check for special tokens
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token_id = self.llama_tokenizer.eos_token_id

        print(f"BOS token ID: {self.llama_tokenizer.bos_token_id}")
        print(f"EOS token ID: {self.llama_tokenizer.eos_token_id}")
        print(f"PAD token ID: {self.llama_tokenizer.pad_token_id}")

        if args.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.qwen_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                args.qwen_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading Qwen LoRA Done')
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading Qwen Done')
        # --------------------------------------------------------
        # DEFINE MIMIC-CXR CONCEPTS
        # --------------------------------------------------------
        self.disease_concepts = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        # ---------------- Connector (Hybrid Abstractor) ----------------
        self.llm_hidden_size = self.llama_model.config.hidden_size

        print(
            f"Initializing HybridVisionAbstractor: VisDim={self.visual_hidden_size} -> LLMDim={self.llm_hidden_size}")
        # Replacing MLP with the Hybrid Connector
        self.llama_proj = HybridVisionAbstractor(
            visual_dim=self.visual_hidden_size,
            llm_dim=self.llm_hidden_size,
            num_queries=64  # Dynamic compression to 64 tokens
        )
        # --------------------------------------------------------
        # SOTA TRICK: Extract Embeddings & Inject
        # --------------------------------------------------------
        self._inject_diseases_into_connector()

        self.layer_norm = nn.LayerNorm(self.llm_hidden_size)
        self.end_sym = "<|im_end|>"

        self._setup_prompts()

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(
                f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def _inject_diseases_into_connector(self):
        """
        Uses the frozen LLM embedding layer to turn disease names 
        into vectors, then pushes them into the Connector.
        """
        print(
            f"Generating embeddings for {len(self.disease_concepts)} MIMIC-CXR concepts...")
        device = self.llama_model.device

        concept_vectors = []
        for concept in self.disease_concepts:
            # 1. Tokenize the concept (e.g., "Pleural Effusion" -> [234, 551])
            ids = self.llama_tokenizer(
                concept, return_tensors="pt", add_special_tokens=False).input_ids
            ids = ids.to(self.embed_tokens.weight.device)

            # 2. Get Embeddings
            with torch.no_grad():
                embeds = self.embed_tokens(ids)  # Shape: (1, Seq_Len, Dim)

            # 3. Mean Pooling (if concept is multiple tokens, average them to get 1 vector)
            concept_vec = embeds.mean(dim=1).squeeze(0)  # Shape: (Dim,)
            concept_vectors.append(concept_vec)

        # Stack all 14 vectors
        concept_stack = torch.stack(concept_vectors)  # Shape: (14, Dim)

        # Inject into the Hybrid Abstractor
        self.llama_proj.inject_concept_init(concept_stack)

    def _setup_prompts(self):
        if self.args.task == 'label':
            self.prompt = "<|im_start|>system\nYou are a medical assistant...<|im_end|>\n<|im_start|>user\nThe chest xray image shows:"
        elif self.args.task == 'triple':
            self.prompt = "<|im_start|>system\nYou are a medical assistant...<|im_end|>\n<|im_start|>user\nThe chest xray image shows:"
        elif self.args.task == 'report':
            self.prompt = "<|im_start|>system\nYou are a professional radiologist. Please generate a comprehensive and detailed diagnosis report for this chest xray image.<|im_end|>\n<|im_start|>user\nThe chest X-ray image shows:"

    def score(self, ref, hypo):
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores

    def encode_img(self, images):
        """
        Robust encoding that handles List of Views (e.g. [Front, Lat]) 
        and Multi-View Tensors.
        """
        # 1. Handle List Input
        if isinstance(images, list):
            # If the list contains 4D tensors (Batch, C, H, W), it means we have [View1_Batch, View2_Batch]
            # We must stack along dim=1 to get (Batch, Views, C, H, W)
            if images[0].dim() == 4:
                images = torch.stack(images, dim=1)
            else:
                # Standard stacking for list of 3D samples
                images = torch.stack(images)

        # Move to correct device
        images = images.to(self.device)

        # 2. Handle Multi-View Input (5D -> 4D)
        # Input shape should now be (Batch, Num_Views, C, H, W)
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            # Collapse Batch and View dims to feed into Swin
            images = images.view(B * N, C, H, W)
            multi_view = True
        else:
            multi_view = False
            B = images.shape[0]  # Batch size if 4D

        # 3. Forward Vision Encoder (Swin)
        # Output: last_hidden_state is (Batch*N, 49, 1024)
        visual_outputs = self.visual_encoder(images)
        image_embeds = visual_outputs.last_hidden_state

        # 4. Handle Multi-View Merge (Aggregating Views)
        if multi_view:
            # Reshape back to (Batch, Num_Views, Seq, Dim)
            _, Seq, Dim = image_embeds.shape
            image_embeds = image_embeds.view(B, N, Seq, Dim)
            # Mean Pooling: Combine Frontal/Lateral features into one robust representation
            image_embeds = image_embeds.mean(dim=1)  # (Batch, 49, 1024)

        # 5. Forward Hybrid Abstractor
        # Input: (Batch, 49, 1024) -> Output: (Batch, 64, LLM_Dim)
        inputs_llama = self.llama_proj(image_embeds)

        # 6. Create Attention Mask
        atts_llama = torch.ones(
            inputs_llama.shape[:2],  # (Batch, 64)
            dtype=torch.long
        ).to(self.device)

        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img):
        full_prompt = f"{self.prompt} <image><|im_end|>\n<|im_start|>assistant\n"
        batch_size = img_embeds.shape[0]

        p_before, p_after = full_prompt.split('<image>')

        p_before_tokens = self.llama_tokenizer(
            p_before,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)

        p_after_tokens = self.llama_tokenizer(
            p_after,
            return_tensors="pt",
            add_special_tokens=False
        ).to(img_embeds.device)

        with torch.no_grad():
            p_before_embeds = self.embed_tokens(
                p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.embed_tokens(
                p_after_tokens.input_ids).expand(batch_size, -1, -1)

        # Concatenate: [prompt_before] + [image] + [prompt_after]
        wrapped_img_embeds = torch.cat(
            [p_before_embeds, img_embeds, p_after_embeds], dim=1)

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

        # Encode
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        # FIX: Get device from the tensor, not the list
        device = img_embeds.device

        # Wrap with text prompt
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text"]]

        # Tokenize target text
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)  # <--- USE 'device' HERE

        # Create labels
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       # <--- USE 'device' HERE
                       dtype=torch.long).to(device).fill_(-100)
        )
        # Now dimensions will match!
        targets = torch.cat([empty_targets, targets], dim=1)

        # ... rest of the function ...
        with torch.no_grad():
            to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
            use_cache=False
        )
        return {"loss": outputs.loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]

        save_obj = {
            "state_dict": state_dict,
            "hyper_parameters": self.hparams,
            "pytorch-lightning_version": pl.__version__,
            "epoch": current_epoch,
            "global_step": global_step,
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path,
                    'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_rougel{:3f}_bleu{:3f}_cider{:3f}.pth".format(
                current_epoch, global_step, eval_res['ROUGE_L'], eval_res['Bleu_4'], eval_res['CIDEr']
            ),
        )
        self.print("Saving checkpoint at step {} to {}.".format(
            global_step, save_to))
        torch.save(save_obj, save_to)

    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        text = samples["text"]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        # Ensure correct device usage if needed (though generate handles inputs well)
        device = img_embeds.device

        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        outputs = self.llama_model.generate(
            inputs_embeds=img_embeds,
            attention_mask=atts_img,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append(
            {"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref

    def decode(self, output_token):
        if len(output_token) > 0 and output_token[0] == self.llama_tokenizer.pad_token_id:
            output_token = output_token[1:]
        if len(output_token) > 0 and output_token[0] == self.llama_tokenizer.bos_token_id:
            output_token = output_token[1:]

        output_text = self.llama_tokenizer.decode(
            output_token, add_special_tokens=False)

        output_text = output_text.split(self.end_sym)[0].strip()
        output_text = output_text.replace('<|im_start|>', '')
        output_text = output_text.replace('<|im_end|>', '')
        output_text = output_text.replace('<|endoftext|>', '')
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        self.val_step_outputs = self.val_step_outputs
        ref, hypo, ids = [], [], []

        if isinstance(self.val_step_outputs[0], dict):
            for out in self.val_step_outputs:
                ref.extend(out['ref'])
                hypo.extend(out['hypo'])
                ids.extend(out['id'])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder,
                  f"result_{current_epoch}_{global_step}.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if val_score > self.val_score:
            self.save_checkpoint(eval_res)
            self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        # reuse validation logic for testing
        return self.validation_step(samples, batch_idx)

    def on_test_epoch_end(self):
        # reuse validation logic (or customize if needed)
        self.val_step_outputs = self.test_step_outputs
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
