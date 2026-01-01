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


class LightMedVLMForTraining(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        print(f'Loading vision encoder: {args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                r=args.vis_r,
                lora_alpha=args.vis_alpha,
                target_modules=["query", "value"],
                lora_dropout=args.lora_dropout,
                bias="none",
                modules_to_save=["classifier"],
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder: {args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder: {args.vision_model} -- Done')

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
        
        self.proj = MLP(
            in_dim=self.visual_encoder.num_features,
            inter_dim=2048,
            out_dim=self.model.config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(self.model.config.hidden_size)
        
        self.end_sym = "<|im_end|>"

        self.prompt = "<|im_start|>system\nYou are a professional radiologist. Please generate a comprehensive and detailed diagnosis report for this chest xray image.<|im_end|>\n<|im_start|>user\nThe chest X-ray image shows:"
        
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        
        


        
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

    def prompt_wrap(self, img_embeds, atts_img):
        """
        Wrap image embeddings with Qwen-style prompt.
        Format: {prompt_before} <image> {prompt_after}
        """
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
            max_length=self.hparams.max_length,
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
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_rougle_l{:3}_bleu{:3f}_cider{:3f}.pth".format(
                current_epoch, global_step, eval_res['ROUGE_L'],eval_res['Bleu_4'], eval_res['CIDEr']
            ),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        text = samples["text"]

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
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

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
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
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
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}.json"), 'w'))
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
        self.tokenizer.padding_side = "right"
        text = samples["text"]

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
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

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
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
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
        json.dump(hypo, open(os.path.join(result_folder, "test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
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

    def Clip_loss(self, image_embeds, text_embeds): # image_embeds: [6, 4096], text_embeds: [6, 4096]
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=image_embeds.device).long()

        return (F.cross_entropy(logits_per_text, labels) +
                F.cross_entropy(logits_per_image, labels)) / 2

    def Temporal_weight(self, text_embeds, text_mask, mode='former'):
        bs, N = text_embeds.shape[0], text_embeds.shape[1]
        if mode=='former':
            for bs_idx in range(bs):
                one_sent = 0.0
                weights = 0.0
                num_tokens = sum(text_mask[bs_idx])
                for idx in range(num_tokens):
                    w = (num_tokens - idx + 1) ** (-0.5)
                    one_sent += w * text_embeds[bs_idx,idx,:]
                    weights += w
                one_sent = one_sent / weights
                if bs_idx == 0:
                    all_embeds = one_sent.unsqueeze(0)
                else:
                    all_embeds = torch.cat((all_embeds, one_sent.unsqueeze(0)), dim=0)

        elif mode=='latter':
            all_embeds = 0.0
            weights = 0.0
            for idx in range(N):
                w = (N - idx + 1) ** (-0.5)
                all_embeds += w * text_embeds[:,idx,:]
                weights += w
            all_embeds = all_embeds / weights
        
        elif mode=='mean':
            for bs_idx in range(bs):
                one_sent = 0.0
                cnt = 0.0
                num_tokens = sum(text_mask[bs_idx])
                for idx in range(num_tokens):
                    one_sent += text_embeds[bs_idx,idx,:]
                    cnt += 1
                one_sent = one_sent / cnt
                if bs_idx == 0:
                    all_embeds = one_sent.unsqueeze(0)
                else:
                    all_embeds = torch.cat((all_embeds, one_sent.unsqueeze(0)), dim=0)
            # all_embeds = torch.mean(text_embeds, dim=1)

        return all_embeds

    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 
        
    @torch.no_grad()
    def inference(self, image_paths, beam_size=1, do_sample=False, min_new_tokens=1, 
                 max_new_tokens=100, repetition_penalty=1.0, length_penalty=1.0, 
                 temperature=1.0):
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
        
        hypo = [self.decode(i) for i in outputs]
        return hypo

