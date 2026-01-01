import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from peft import get_peft_model, LoraConfig, TaskType
import torch.distributed as dist
from transformers import BertTokenizer, AutoImageProcessor
from PIL import Image
import numpy as np
import math

# ==============================================================================
# REPLACED MLP WITH HYBRID VISION ABSTRACTOR
# ==============================================================================


class HybridVisionAbstractor(nn.Module):
    def __init__(self, visual_dim, llm_dim, num_queries=64):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(visual_dim, visual_dim, 3, 1, 1), nn.BatchNorm2d(
                visual_dim), nn.SiLU(),
            nn.Conv2d(visual_dim, visual_dim, 3, 1,
                      1), nn.BatchNorm2d(visual_dim), nn.SiLU()
        )
        self.vis_proj = nn.Linear(visual_dim, llm_dim)
        self.latents = nn.Parameter(torch.randn(1, num_queries, llm_dim))
        self.cross_attn = nn.MultiheadAttention(
            llm_dim, 8, batch_first=True, dropout=0.1)
        self.ln_q, self.ln_v, self.ln_out = nn.LayerNorm(
            llm_dim), nn.LayerNorm(llm_dim), nn.LayerNorm(llm_dim)
        self.mlp = nn.Sequential(
            nn.Linear(llm_dim, llm_dim*4), nn.GELU(), nn.Linear(llm_dim*4, llm_dim))

    def inject_concept_init(self, concept_embeds):
        with torch.no_grad():
            # Inject concepts into the first N learnable queries
            self.latents[:, :concept_embeds.shape[0],
                         :] = concept_embeds.unsqueeze(0)
        print(f">> INJECTED {concept_embeds.shape[0]} DISEASE CONCEPTS <<")

    def forward(self, visual_features):
        # visual_features: (Batch, Seq_Len, Vis_Dim)
        B, L, C = visual_features.shape
        H = W = int(L**0.5)

        # Spatial Refinement (CNN)
        x_img = visual_features.permute(0, 2, 1).view(B, C, H, W)
        x_refined = self.conv_block(x_img).flatten(2).permute(0, 2, 1)

        # Projection
        x_vis = self.vis_proj(x_refined)

        # Cross Attention (Queries attending to Image)
        q = self.ln_q(self.latents.repeat(B, 1, 1))
        k = v = self.ln_v(x_vis)
        attn_out, _ = self.cross_attn(q, k, v)

        # Residual + MLP
        x = self.ln_out(q + attn_out)
        x = x + self.mlp(x)
        return x


class LightMedVLMForTraining(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        print(f'Loading vision encoder: {args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(
            args.vision_model)
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
        else:
            print(
                f'Loading Trainable vision encoder: {args.vision_model} -- Done')

        print('Loading LLM model')
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.llm_model,
            use_fast=False,
            trust_remote_code=True
        )

        print(f"BOS token ID: {self.tokenizer.bos_token_id}")
        print(f"EOS token ID: {self.tokenizer.eos_token_id}")
        print(f"PAD token ID: {self.tokenizer.pad_token_id}")

        # Load LLM model
        if args.low_resource:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype=torch.bfloat16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.llm_model,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        if args.llm_use_lora:
            self.embed_tokens = self.model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.llm_r,
                lora_alpha=args.llm_alpha,
                lora_dropout=args.lora_dropout,
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

        # --- REPLACED MLP WITH HYBRID VISION ABSTRACTOR ---
        # NOTE: Renamed self.proj to self.llm_proj to imply complex connector,
        # but logic matches the Hybrid structure.
        self.visual_hidden_size = self.visual_encoder.config.hidden_size
        self.llm_hidden_size = self.model.config.hidden_size

        self.llm_proj = HybridVisionAbstractor(
            visual_dim=self.visual_hidden_size,
            llm_dim=self.llm_hidden_size,
            num_queries=64
        )

        # Initialize concepts
        self.disease_concepts = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
                                 "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
                                 "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        self._inject_diseases_into_connector()

        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)

        self.end_sym = "<|im_end|>"

        self.system_prompt = "<|im_start|>system\nYou are a professional radiologist. Please answer the question based on the chest X-ray image and choose from the following two options: [yes, no].<|im_end|>\n"

        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

    def _inject_diseases_into_connector(self):
        """Helper to inject disease embeddings into the Hybrid Abstractor"""
        device = self.embed_tokens.weight.device
        concept_vectors = []
        for concept in self.disease_concepts:
            # Use self.tokenizer as defined in init
            ids = self.tokenizer(concept, return_tensors="pt",
                                 add_special_tokens=False).input_ids
            # Move ids to correct device (if model is already on GPU)
            if hasattr(self.embed_tokens.weight, 'device'):
                ids = ids.to(self.embed_tokens.weight.device)

            with torch.no_grad():
                embeds = self.embed_tokens(ids)
            concept_vec = embeds.mean(dim=1).squeeze(0)
            concept_vectors.append(concept_vec)

        if len(concept_vectors) > 0:
            concept_stack = torch.stack(concept_vectors)
            self.llm_proj.inject_concept_init(concept_stack)

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
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
        Updated encode_img to work with HybridVisionAbstractor.
        Input images: Tensor (B, 3, H, W) or List[Tensor]
        """
        # Ensure input is a tensor batch for efficiency
        if isinstance(images, list):
            # Stack if it's a list of tensors
            images = torch.stack(images)

        images = images.to(self.device)

        # 1. Vision Encoder
        visual_outputs = self.visual_encoder(images)
        image_embeds = visual_outputs.last_hidden_state  # (B, Seq, Dim)

        # 2. Hybrid Projector (replaces self.proj)
        inputs_llama = self.llm_proj(image_embeds)  # (B, 64, Dim)

        # 3. Create Attention Mask
        atts_llama = torch.ones(
            inputs_llama.shape[:2], dtype=torch.long).to(images.device)

        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, atts_img, questions):
        """
        Wrap image embeddings with Qwen-style prompt including the question.
        Format: {system_prompt} <user_start> {question} <image> <user_end> <assistant_start>
        """
        batch_size = img_embeds.shape[0]

        # Build prompts for each item in the batch
        wrapped_embeds_list = []
        wrapped_atts_list = []

        for i in range(batch_size):
            question = questions[i] if questions[i] is not None else "Describe the following image in detail."

            # Construct full prompt with question
            full_prompt = f"{self.system_prompt}<|im_start|>user\n{question} <image><|im_end|>\n<|im_start|>assistant\n"

            # Split at image placeholder
            p_before, p_after = full_prompt.split('<image>')

            # Tokenize prompt parts
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

            # Get embeddings
            with torch.no_grad():
                p_before_embeds = self.embed_tokens(p_before_tokens.input_ids)
                p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)

            # Concatenate: [prompt_before] + [image] + [prompt_after]
            # img_embeds[i:i+1] is (1, 64, Dim)
            wrapped_embeds = torch.cat([
                p_before_embeds,
                img_embeds[i:i+1],
                p_after_embeds
            ], dim=1)

            wrapped_embeds_list.append(wrapped_embeds)

            # Create attention mask
            wrapped_atts = torch.ones(
                wrapped_embeds.shape[1],
                device=img_embeds.device,
                dtype=atts_img.dtype
            )
            wrapped_atts_list.append(wrapped_atts)

        # Find max sequence length in the batch
        max_seq_len = max(embeds.shape[1] for embeds in wrapped_embeds_list)

        # Pad all sequences to the same length
        padded_embeds_list = []
        padded_atts_list = []

        for embeds, atts in zip(wrapped_embeds_list, wrapped_atts_list):
            seq_len = embeds.shape[1]
            if seq_len < max_seq_len:
                # Pad embeddings with zeros
                padding_size = max_seq_len - seq_len
                padding = torch.zeros(
                    embeds.shape[0],
                    padding_size,
                    embeds.shape[2],
                    dtype=embeds.dtype,
                    device=embeds.device
                )
                embeds = torch.cat([embeds, padding], dim=1)

                # Pad attention mask with zeros (masked positions)
                atts_padding = torch.zeros(
                    padding_size,
                    dtype=atts.dtype,
                    device=atts.device
                )
                atts = torch.cat([atts, atts_padding], dim=0)

            padded_embeds_list.append(embeds)
            padded_atts_list.append(atts)

        # Stack all items in the batch
        wrapped_img_embeds = torch.cat(padded_embeds_list, dim=0)
        wrapped_atts_img = torch.stack(padded_atts_list, dim=0)

        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        image = samples["image"]
        questions = samples.get("question", [None] * len(samples["text"]))

        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        # Wrap image with prompt (now includes question)
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img, questions)

        self.tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["text"]]

        # Tokenize target text
        to_regress_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)

        # Create labels: mask prompt+image tokens with -100, keep text tokens
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        # Create empty targets for prompt+image tokens
        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]],
                       dtype=torch.long).to(img_embeds.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        # Get text embeddings
        with torch.no_grad():
            to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        # Concatenate all embeddings: [prompt+image] + [text]
        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_img, to_regress_tokens.attention_mask], dim=1)

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
            "checkpoint_epoch{}_step{}_rougle_l{:3}_bleu{:3f}_cider{:3f}.pth".format(
                current_epoch, global_step, eval_res['ROUGE_L'], eval_res['Bleu_4'], eval_res['CIDEr']
            ),
        )
        self.print("Saving checkpoint at step {} to {}.".format(
            global_step, save_to))
        torch.save(save_obj, save_to)

    @torch.no_grad()
    def validation_forward(self, samples):
        """
        Validation forward:
        - Only does generation
        - Does NOT compute training loss
        """
        self.model.eval()

        image = samples["image"]
        questions = samples.get("question", [None] * len(image))
        refs = samples["text"]  # ground truth answers

        # ---- Encode image ----
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)

        # ---- Build the multimodal prompt ----
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img, questions)

        # Cast to LLM dtype (important for BF16 or FP16)
        dtype = self.model.dtype
        img_embeds = img_embeds.to(dtype)

        # ---- Generation ----
        outputs = self.model.generate(
            inputs_embeds=img_embeds,
            attention_mask=atts_img,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # ---- Decode outputs ----
        preds = [self.decode(o) for o in outputs]

        return preds, refs, questions

    def validation_step(self, batch, batch_idx):
        preds, refs, questions = self.validation_forward(batch)

        # Save for epoch-end scoring
        self.val_step_outputs.append({
            "preds": preds,
            "refs": refs,
            "questions": questions,
            "ids": batch["id"],
        })

        # Optionally log dummy val loss (for progress bar)
        dummy_loss = torch.tensor(0.0, device=self.device)
        self.log("val_loss", dummy_loss, prog_bar=True,
                 on_step=False, on_epoch=True)

        return preds

    def decode(self, output_token):
        """Decode output tokens to text."""
        # Remove special tokens at the beginning
        if len(output_token) > 0 and output_token[0] == self.tokenizer.pad_token_id:
            output_token = output_token[1:]
        if len(output_token) > 0 and output_token[0] == self.tokenizer.bos_token_id:
            output_token = output_token[1:]

        # Decode to text
        output_text = self.tokenizer.decode(
            output_token, add_special_tokens=False)

        # Split at end symbol and clean up
        output_text = output_text.split(self.end_sym)[0].strip()

        # Remove Qwen special tokens
        output_text = output_text.replace('<|im_start|>', '')
        output_text = output_text.replace('<|im_end|>', '')
        output_text = output_text.replace('<|endoftext|>', '')
        output_text = output_text.replace('<unk>', '')

        return output_text

    def on_validation_epoch_end(self):
        if len(self.val_step_outputs) == 0:
            return

        preds, refs, ids = [], [], []

        for out in self.val_step_outputs:
            preds.extend(out["preds"])
            refs.extend(out["refs"])
            ids.extend(out["ids"])

        # Build dicts for evalcap scoring
        ref_dict = {k: [v] for k, v in zip(ids, refs)}
        pred_dict = {k: [v] for k, v in zip(ids, preds)}

        metrics = self.score(ref_dict, pred_dict)
        self.log_dict(metrics, prog_bar=True, logger=True)

        # Save JSON
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)

        json.dump(pred_dict, open(os.path.join(
            result_folder, f"val_pred.json"), 'w'))
        json.dump(ref_dict, open(os.path.join(
            result_folder, f"val_ref.json"), 'w'))

        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        text = samples["text"]
        questions = samples.get("question", [None] * len(text))

        to_regress_tokens = self.tokenizer(
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
        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img, questions)

        batch_size = img_embeds.shape[0]
        inputs_embeds = img_embeds
        attention_mask = atts_img

        outputs = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append(
            {"hypo": hypo, "ref": ref, "id": samples["id"], "question": questions})
        return hypo, ref

    def on_test_epoch_end(self):
        self.test_step_outputs = self.test_step_outputs
        ref, hypo, ids = [], [], []

        if isinstance(self.test_step_outputs[0], dict):
            for out in self.test_step_outputs:
                ref.extend(out['ref'])
                hypo.extend(out['hypo'])
                ids.extend(out['id'])

        ref = {k: [v] for k, v in zip(ids, ref)}
        hypo = {k: [v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref, hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(
            result_folder, "test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

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

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    @torch.no_grad()
    def all_gather(self, data):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        dist.barrier()
        gather_data = [None for _ in range(torch.distributed.get_world_size())]
        dist.all_gather_object(gather_data, data)
        return gather_data

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(
            img, return_tensors="pt").pixel_values
        return pixel_values[0]

    @torch.no_grad()
    def inference(self, image_paths, question=None, beam_size=1, do_sample=False,
                  min_new_tokens=1, max_new_tokens=100, repetition_penalty=1.0,
                  length_penalty=1.0, temperature=1.0):
        """
        Generate answer from images and question.

        Args:
            image_paths: List of image paths
            question: Question text (optional, defaults to general description)
            beam_size, do_sample, etc.: Generation parameters
        """
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

        # Use the question in the prompt
        if question is None:
            question = "Describe the following image in detail."
        questions = [question]

        img_embeds, atts_img = self.prompt_wrap(
            img_embeds, atts_img, questions)

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

        hypo = [self.decode(i) for i in outputs]
        return hypo
