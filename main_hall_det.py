import argparse
import csv
import math
import os
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms

from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import EntailmentDeberta, get_semantic_ids
from SeEntLib.demo import get_sentence_semantic_entropy_w_semantic_ids

from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)  


class AddPoissonNoise:
    def __init__(self, scale=50): 
        self.scale = scale

    def __call__(self, img):
        img_scaled = (img * self.scale).clamp(0, 255)  
        noise = torch.poisson(img_scaled) / self.scale  
        return noise.clamp(0, 1)  
    

def get_sequence_log_logits(token_ids, scores, special_token_ids):
    # token_ids: torch.Size([no_samples, num_gen_tokens]); 
    # scores: torch.Size([no_samples, num_gen_tokens, num_vocab])
    # Keep sequence lengths aligned and clamp ids to avoid CUDA gather OOB.
    seq_len = min(token_ids.size(1), scores.size(1))
    if seq_len == 0:
        # Keep downstream entropy computation stable when no tokens were generated.
        return [torch.tensor([0.0], dtype=torch.float32) for _ in range(token_ids.size(0))]

    token_ids = token_ids[:, :seq_len]
    scores = scores[:, :seq_len, :]

    log_logits = F.log_softmax(scores, dim=-1)
    special_token_ids = special_token_ids.to(token_ids.device)
    valid_vocab_mask = (token_ids >= 0) & (token_ids < log_logits.size(-1))
    safe_token_ids = token_ids.clamp(min=0, max=log_logits.size(-1) - 1)
    selected_log_logits = torch.gather(log_logits, 2, safe_token_ids.unsqueeze(-1)).squeeze(-1)
    mask = (~torch.isin(token_ids, special_token_ids)) & valid_vocab_mask
    filtered_log_logits = []
    for logits, row_mask in zip(selected_log_logits, mask):
        row_logits = logits[row_mask]
        if row_logits.numel() == 0:
            row_logits = torch.tensor([0.0], dtype=logits.dtype, device=logits.device)
        filtered_log_logits.append(row_logits.cpu())

    return filtered_log_logits


def safe_float(x, default=0.0):
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return float(default)
        x = x.detach().float().cpu()
        if x.numel() != 1:
            x = x.mean()
        x = x.item()

    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(default)

    return x if math.isfinite(x) else float(default)


def safe_prob_dist(prob_dist):
    if not isinstance(prob_dist, torch.Tensor):
        return torch.tensor([1.0], dtype=torch.float32)

    prob_dist = prob_dist.detach().float().flatten().cpu()
    if prob_dist.numel() == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    prob_dist = torch.nan_to_num(prob_dist, nan=0.0, posinf=0.0, neginf=0.0)
    total = prob_dist.sum()
    if total <= 0:
        return torch.full_like(prob_dist, fill_value=1.0 / prob_dist.numel())

    return prob_dist / total


def get_messages(prompt, image_input):
    messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a medical image analysis expert."}] 
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "image": image_input}
                    ]
                }
            ]
    return messages


def load_vlm_model_and_processor(model_id, hf_token, device0):
    common_kwargs = {
        "token": hf_token,
        "torch_dtype": torch.bfloat16,
        "device_map": device0,
    }

    if AutoModelForImageTextToText is not None:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **common_kwargs).eval()
            processor = AutoProcessor.from_pretrained(model_id, token=hf_token, backend="pil")
            return model, processor
        except ValueError as exc:
            # llava-med-v1.5-mistral-7b ships model_type=llava_mistral (transformers 4.36 era).
            # Newer auto mappings can miss this key, so fallback to causal-lm path.
            if "llava_mistral" not in str(exc):
                raise

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            **common_kwargs,
        ).eval()
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token, trust_remote_code=True, backend="pil")
        return model, processor
    except ValueError as exc:
        # llava-med-v1.5-mistral-7b ships model_type=llava_mistral (transformers 4.36 era).
        # Keep an actionable error if caller uses an incompatible environment.
        if "llava_mistral" not in str(exc):
            raise
        raise RuntimeError(
            "Model type llava_mistral is not supported in this transformers build. "
            "Use the pixi environment/task pinned for llava (vase-llava)."
        ) from exc


def main_pred_hallscore(modelid="google/medgemma-4b-it", csv_file='outputs/radvqa_medgemma_hallscore.csv'):
    device0 = torch.device("cuda:0")  # GPU for medgemma
    device1 = torch.device("cuda:1")  # GPU for entailment model

    # if no output directory, create one
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    # load model
    model_id = modelid # "microsoft/llava-med-v1.5-mistral-7b" "google/medgemma-4b-it" 
    hf_token = os.getenv("HF_TOKEN")
    model, processor = load_vlm_model_and_processor(model_id, hf_token, device0)

    # load dataset (open-ended VQA test samples)
    test_set = load_dataset("flaviagiammarino/vqa-rad", split="test").filter(lambda x: x["answer"].lower() != "yes" and x["answer"].lower() != "no")
    
    num_samples = 10
    entailment_model = EntailmentDeberta(device=device1)

    img_trans_ori = transforms.Compose([ # PIL.Image
        transforms.Resize((512, 512)),
    ])

    img_trans_noi = transforms.Compose([ # PIL.Image
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)), 
        transforms.RandomRotation(degrees=10), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
        transforms.ToTensor(), # tensor
        AddGaussianNoise(mean=0, std=0.07), # std=0.01~0.03, slight noise; std=0.05~0.1, strong noise
        AddPoissonNoise(scale=70), # scale=10-100
        transforms.ToPILImage(), # PIL.Image
    ])

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['img_idx', 'question', 'ref_answer', 'gen_answer', 'RadFlag', 'SE', 'VASE'])
    
    with torch.no_grad():
        for idx in tqdm(range(test_set.__len__()), desc='inference'):

            image = test_set[idx]['image'].convert('RGB') if test_set[idx]['image'].mode != 'RGB' else test_set[idx]['image'] 
            question, ref_answer = test_set[idx]['question'], test_set[idx]['answer'] 

            # generate answer when temperature == 0.1
            image_input = img_trans_ori(image) 
            prompt = 'Answer this question as concisely as possible based on the provide images: ' + question
            messages = get_messages(prompt, image_input)
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            input_len = inputs["input_ids"].shape[-1] # input_ids, attention_mask, token_type_ids, pixel_values
            with torch.inference_mode():
                outputs = model.generate(**inputs, 
                                         max_new_tokens=200, do_sample=True, temperature=0.1, top_p=0.9, num_beams=1, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id, 
                                         return_dict_in_generate=True, output_scores=True, output_attentions=False,
                                         )
            token_ids, scores = outputs.sequences[0][input_len:], outputs.scores
            gen_answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # SE & VASE & RadFlag estimation
            def run_one_pass(inputs):
                with torch.inference_mode():
                    outputs = model.generate(**inputs, 
                                            max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.9, num_beams=1, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id, 
                                            return_dict_in_generate=True, output_scores=True, output_attentions=False,
                                            )
                return outputs
            
            inputs['input_ids'] = inputs['input_ids'].repeat(num_samples, 1)
            inputs['attention_mask'] = inputs['attention_mask'].repeat(num_samples, 1)
            if 'token_type_ids' in inputs:
                inputs['token_type_ids'] = inputs['token_type_ids'].repeat(num_samples, 1)
            inputs['pixel_values'] = inputs['pixel_values'].repeat(num_samples, 1, 1, 1)
            outputs = run_one_pass(inputs) 
            token_ids, scores = outputs.sequences[:,input_len:], outputs.scores
            sam_answers = processor.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

            def make_noisy_inputs(image_input, prompt):
                noisy_inputs_list = []

                for _ in range(num_samples):
                    noisy_image_input = img_trans_noi(image_input)
                    noi_messages = get_messages(prompt, noisy_image_input)
                    noisy_inputs = processor.apply_chat_template(noi_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                    noisy_inputs_list.append(noisy_inputs)

                noisy_inputs = {key: torch.cat([ni[key] for ni in noisy_inputs_list], dim=0) for key in noisy_inputs_list[0]}
                return noisy_inputs

            noisy_inputs = make_noisy_inputs(image_input, prompt)
            noisy_outputs = run_one_pass(noisy_inputs)
            noisy_input_len = noisy_inputs["input_ids"].shape[-1]
            token_ids_noisy, scores_noisy = noisy_outputs.sequences[:, noisy_input_len:], noisy_outputs.scores
            sam_answers_noisy = processor.tokenizer.batch_decode(token_ids_noisy, skip_special_tokens=True)

            temp_gen_answer = question + ' ' + gen_answer
            temp_sam_answers = [f'{question} {r}' for r in sam_answers]
            temp_sam_answers_noisy = [f'{question} {r}' for r in sam_answers_noisy]
            semantic_ids = get_semantic_ids([temp_gen_answer]+temp_sam_answers+temp_sam_answers_noisy, model=entailment_model, strict_entailment=False)

            # RadFlag
            anchor_semantic_id = semantic_ids[0]
            num_support_samples = semantic_ids[1:num_samples+1].count(anchor_semantic_id)
            RadFlag = num_support_samples / num_samples
            RadFlag = safe_float(RadFlag)

            # Semantic Entropy
            token_ids = token_ids.to(device1)
            scores = torch.stack(scores, dim=1).to(device1)
            filtered_log_logits = get_sequence_log_logits(token_ids, scores, torch.tensor(processor.tokenizer.all_special_ids, device=token_ids.device))
            SeEnt, SeDist = get_sentence_semantic_entropy_w_semantic_ids(filtered_log_logits, semantic_ids[1:num_samples+1])
            SeEnt = safe_float(SeEnt)
            SeDist = safe_prob_dist(SeDist)
            
            # VASE
            alpha = 1.0
            token_ids_noisy = token_ids_noisy.to(device1)
            scores_noisy = torch.stack(scores_noisy, dim=1).to(device1)
            filtered_log_logits_noisy = get_sequence_log_logits(token_ids_noisy, scores_noisy, torch.tensor(processor.tokenizer.all_special_ids, device=token_ids_noisy.device))
            _, SeDist_noisy = get_sentence_semantic_entropy_w_semantic_ids(filtered_log_logits_noisy, semantic_ids[num_samples+1:2*num_samples+1])
            SeDist_noisy = safe_prob_dist(SeDist_noisy)

            align_SeDist = torch.zeros(max(semantic_ids)+1, device='cpu')
            align_SeDist_noisy = torch.zeros_like(align_SeDist)
            align_SeDist[torch.unique(torch.tensor(semantic_ids[1:num_samples+1]))] = SeDist
            align_SeDist_noisy[torch.unique(torch.tensor(semantic_ids[num_samples+1:2*num_samples+1]))] = SeDist_noisy
            ViSeDist= align_SeDist + alpha * (align_SeDist - align_SeDist_noisy)
            ViSeDist = torch.nan_to_num(ViSeDist, nan=0.0, posinf=0.0, neginf=0.0)
            ViSeDist = torch.softmax(ViSeDist, dim=-1) # remove negative values
            ViSeDist = torch.clamp(ViSeDist, min=1e-10)
            ViSeDist = ViSeDist / ViSeDist.sum()
            ViSeEnt = safe_float(-torch.sum(ViSeDist * torch.log(ViSeDist)))

            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([idx, question, ref_answer, gen_answer, RadFlag, SeEnt, ViSeEnt])
            
            del sam_answers_noisy
            del outputs, semantic_ids, temp_sam_answers_noisy, temp_sam_answers, temp_gen_answer 
            del token_ids, scores
            del token_ids_noisy, scores_noisy, filtered_log_logits_noisy, noisy_outputs, filtered_log_logits
            del SeDist, SeDist_noisy, ViSeDist, align_SeDist, align_SeDist_noisy

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=os.getenv("VASE_MODEL_ID", "google/medgemma-4b-it"),
        help="Model id to run for hallucinaton detection",
    )
    parser.add_argument(
        "--csv-file",
        default=os.getenv("VASE_OUTPUT_CSV", "outputs/radvqa_medgemma_hallscore.csv"),
        help="Output csv file path",
    )
    return parser.parse_args()


# python main_hall_det.py
if __name__ == '__main__':
    args = parse_args()
    main_pred_hallscore(args.model_id, args.csv_file)
