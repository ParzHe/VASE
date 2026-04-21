import argparse
import csv
import os

import torch
from tqdm import tqdm
from datasets import load_dataset
from torchvision import transforms

from SeEntLib.uncertainty.uncertainty_measures.semantic_entropy import (
    EntailmentDeberta, get_semantic_ids,
)
from SeEntLib.demo import get_sentence_semantic_entropy_w_semantic_ids
from main_hall_det import (
    AddGaussianNoise, AddPoissonNoise,
    build_inputs, get_sequence_log_logits,
    is_chexagent, load_vlm_model_and_processor,
    safe_float, safe_prob_dist,
)


LAMBDA = 1.0  # visual amplification coefficient (UniVRSE paper §V-C; equals VASE alpha)


def align_probs(dist, ids_subset, n_classes, device):
    """Place per-class probabilities into a vector of length n_classes."""
    aligned = torch.zeros(n_classes, device=device)
    unique_ids = torch.unique(torch.tensor(ids_subset))
    aligned[unique_ids] = dist.to(device)
    return aligned


def entropy_from_logits_contrast(p_main, p_contrast, lam=LAMBDA):
    """UniVRSE VSD: softmax((1+lam)*p_main - lam*p_contrast), then shannon entropy."""
    vsd = (1.0 + lam) * p_main - lam * p_contrast
    vsd = torch.nan_to_num(vsd, nan=0.0, posinf=0.0, neginf=0.0)
    vsd = torch.softmax(vsd, dim=-1)
    vsd = torch.clamp(vsd, min=1e-10)
    vsd = vsd / vsd.sum()
    return safe_float(-torch.sum(vsd * torch.log(vsd)))


def main_pred_hallscore(modelid="google/medgemma-4b-it",
                        csv_file="outputs/radvqa_univrse_hallscore.csv",
                        limit=None):
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    os.makedirs(os.path.dirname(csv_file) or ".", exist_ok=True)

    model_id = modelid
    hf_token = os.getenv("HF_TOKEN")
    model, processor = load_vlm_model_and_processor(model_id, hf_token, device0)
    model_dtype = torch.float16 if is_chexagent(model_id) else torch.bfloat16

    test_set = load_dataset("flaviagiammarino/vqa-rad", split="test").filter(
        lambda x: x["answer"].lower() != "yes" and x["answer"].lower() != "no"
    )

    num_samples = 10
    entailment_model = EntailmentDeberta(device=device1)

    img_trans_ori = transforms.Compose([
        transforms.Resize((512, 512)),
    ])

    # UniVRSE "weak transformations": crop 90-100%, rot +/-10, translate <=10%, brightness/contrast 0.8-1.2.
    img_trans_weak = transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    # UniVRSE "visual distortion": noise-only (Gaussian std=0.07 + Poisson scale=70).
    img_trans_noise_only = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std=0.07),
        AddPoissonNoise(scale=70),
        transforms.ToPILImage(),
    ])

    # VASE's combined transform (kept here so the VASE score in the new CSV
    # matches what main_hall_det.py produces on its own).
    img_trans_vase_noi = transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.9, 1.0)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        AddGaussianNoise(mean=0, std=0.07),
        AddPoissonNoise(scale=70),
        transforms.ToPILImage(),
    ])

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['img_idx', 'question', 'ref_answer', 'gen_answer',
                         'RadFlag', 'SE', 'VASE', 'UniVRSE'])

    total = test_set.__len__()
    if limit is not None:
        total = min(total, int(limit))

    def batch_run(inputs):
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=200, do_sample=True, temperature=1.0,
                top_p=0.9, num_beams=1, use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id,
                return_dict_in_generate=True, output_scores=True,
                output_attentions=False,
            )
        return out

    def tile_inputs(inputs, reps):
        """In-place replicate single-sample inputs to batch size `reps`."""
        inputs['input_ids'] = inputs['input_ids'].repeat(reps, 1)
        inputs['attention_mask'] = inputs['attention_mask'].repeat(reps, 1)
        if 'token_type_ids' in inputs:
            inputs['token_type_ids'] = inputs['token_type_ids'].repeat(reps, 1)
        pv = inputs['pixel_values']
        inputs['pixel_values'] = pv.repeat(reps, *([1] * (pv.ndim - 1)))
        return inputs

    def per_sample_inputs(prompt, image_transform, image_src, reps):
        """Build a batch where each row uses a fresh draw of image_transform(image_src)."""
        piece_list = []
        for _ in range(reps):
            piece = build_inputs(processor, model, prompt,
                                 image_transform(image_src), model_dtype, model_id)
            piece_list.append(piece)
        return {k: torch.cat([p[k] for p in piece_list], dim=0) for k in piece_list[0]}

    with torch.no_grad():
        for idx in tqdm(range(total), desc='inference'):
            image = (test_set[idx]['image'].convert('RGB')
                     if test_set[idx]['image'].mode != 'RGB'
                     else test_set[idx]['image'])
            question = test_set[idx]['question']
            ref_answer = test_set[idx]['answer']

            # 1) T=0.1 anchor answer on the plain resized image.
            image_input = img_trans_ori(image)
            prompt = ('Answer this question as concisely as possible based on the '
                      'provide images: ' + question)
            inputs_lowT = build_inputs(processor, model, prompt, image_input, model_dtype, model_id)
            input_len = 0 if is_chexagent(model_id) else inputs_lowT["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs_lowT, max_new_tokens=200, do_sample=True, temperature=0.1,
                    top_p=0.9, num_beams=1, use_cache=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True, output_scores=True,
                    output_attentions=False,
                )
            token_ids_anchor = outputs.sequences[0][input_len:]
            gen_answer = processor.tokenizer.decode(token_ids_anchor, skip_special_tokens=True)

            # 2) SE/RadFlag: M samples on plain resized image.
            inputs_ori = build_inputs(processor, model, prompt, image_input, model_dtype, model_id)
            inputs_ori = tile_inputs(inputs_ori, num_samples)
            outputs_ori = batch_run(inputs_ori)
            token_ids_ori = outputs_ori.sequences[:, input_len:]
            scores_ori = outputs_ori.scores
            sam_answers_ori = processor.tokenizer.batch_decode(token_ids_ori, skip_special_tokens=True)

            # 3) VASE branch: M samples on transform+noise image (matches main_hall_det.py).
            inputs_vase = per_sample_inputs(prompt, img_trans_vase_noi, image_input, num_samples)
            outputs_vase = batch_run(inputs_vase)
            vase_input_len = 0 if is_chexagent(model_id) else inputs_vase["input_ids"].shape[-1]
            token_ids_vase = outputs_vase.sequences[:, vase_input_len:]
            scores_vase = outputs_vase.scores
            sam_answers_vase = processor.tokenizer.batch_decode(token_ids_vase, skip_special_tokens=True)

            # 4) UniVRSE weak branch: M samples, each with a fresh weak transformation.
            inputs_weak = per_sample_inputs(prompt, img_trans_weak, image, num_samples)
            outputs_weak = batch_run(inputs_weak)
            weak_input_len = 0 if is_chexagent(model_id) else inputs_weak["input_ids"].shape[-1]
            token_ids_weak = outputs_weak.sequences[:, weak_input_len:]
            scores_weak = outputs_weak.scores
            sam_answers_weak = processor.tokenizer.batch_decode(token_ids_weak, skip_special_tokens=True)

            # 5) UniVRSE noise branch: single noise-distorted image, M high-temp samples.
            noisy_image = img_trans_noise_only(image)
            inputs_noise = build_inputs(processor, model, prompt, noisy_image, model_dtype, model_id)
            inputs_noise = tile_inputs(inputs_noise, num_samples)
            outputs_noise = batch_run(inputs_noise)
            noise_input_len = 0 if is_chexagent(model_id) else inputs_noise["input_ids"].shape[-1]
            token_ids_noise = outputs_noise.sequences[:, noise_input_len:]
            scores_noise = outputs_noise.scores
            sam_answers_noise = processor.tokenizer.batch_decode(token_ids_noise, skip_special_tokens=True)

            # Unified semantic clustering across: anchor + ori + VASE-noi + weak + noise.
            # Layout (lengths): [1, M, M, M, M].
            all_texts = (
                [f'{question} {gen_answer}']
                + [f'{question} {r}' for r in sam_answers_ori]
                + [f'{question} {r}' for r in sam_answers_vase]
                + [f'{question} {r}' for r in sam_answers_weak]
                + [f'{question} {r}' for r in sam_answers_noise]
            )
            semantic_ids = get_semantic_ids(all_texts, model=entailment_model,
                                            strict_entailment=False)

            sid_anchor = semantic_ids[0]
            sid_ori = semantic_ids[1:1 + num_samples]
            sid_vase = semantic_ids[1 + num_samples:1 + 2 * num_samples]
            sid_weak = semantic_ids[1 + 2 * num_samples:1 + 3 * num_samples]
            sid_noise = semantic_ids[1 + 3 * num_samples:1 + 4 * num_samples]

            RadFlag = safe_float(sid_ori.count(sid_anchor) / num_samples)

            token_ids_ori_d = token_ids_ori.to(device1)
            scores_ori_stacked = torch.stack(scores_ori, dim=1).to(device1)
            special_ids = torch.tensor(processor.tokenizer.all_special_ids, device=device1)
            llogits_ori = get_sequence_log_logits(token_ids_ori_d, scores_ori_stacked, special_ids)
            SeEnt, SeDist_ori = get_sentence_semantic_entropy_w_semantic_ids(llogits_ori, sid_ori)
            SeEnt = safe_float(SeEnt)
            SeDist_ori = safe_prob_dist(SeDist_ori)

            # VASE (kept 1:1 with main_hall_det.py: contrast ori vs VASE-noi, alpha=1.0).
            token_ids_vase_d = token_ids_vase.to(device1)
            scores_vase_stacked = torch.stack(scores_vase, dim=1).to(device1)
            llogits_vase = get_sequence_log_logits(token_ids_vase_d, scores_vase_stacked, special_ids)
            _, SeDist_vase = get_sentence_semantic_entropy_w_semantic_ids(llogits_vase, sid_vase)
            SeDist_vase = safe_prob_dist(SeDist_vase)

            n_classes = max(semantic_ids) + 1
            align_ori = align_probs(SeDist_ori, sid_ori, n_classes, device='cpu')
            align_vase = align_probs(SeDist_vase, sid_vase, n_classes, device='cpu')
            vase_raw = align_ori + 1.0 * (align_ori - align_vase)
            vase_raw = torch.nan_to_num(vase_raw, nan=0.0, posinf=0.0, neginf=0.0)
            vase_raw = torch.softmax(vase_raw, dim=-1)
            vase_raw = torch.clamp(vase_raw, min=1e-10)
            vase_raw = vase_raw / vase_raw.sum()
            VASE_score = safe_float(-torch.sum(vase_raw * torch.log(vase_raw)))

            # UniVRSE: contrast weak SPD vs noise SPD.
            token_ids_weak_d = token_ids_weak.to(device1)
            scores_weak_stacked = torch.stack(scores_weak, dim=1).to(device1)
            llogits_weak = get_sequence_log_logits(token_ids_weak_d, scores_weak_stacked, special_ids)
            _, SeDist_weak = get_sentence_semantic_entropy_w_semantic_ids(llogits_weak, sid_weak)
            SeDist_weak = safe_prob_dist(SeDist_weak)

            token_ids_noise_d = token_ids_noise.to(device1)
            scores_noise_stacked = torch.stack(scores_noise, dim=1).to(device1)
            llogits_noise = get_sequence_log_logits(token_ids_noise_d, scores_noise_stacked, special_ids)
            _, SeDist_noise = get_sentence_semantic_entropy_w_semantic_ids(llogits_noise, sid_noise)
            SeDist_noise = safe_prob_dist(SeDist_noise)

            align_weak = align_probs(SeDist_weak, sid_weak, n_classes, device='cpu')
            align_noise = align_probs(SeDist_noise, sid_noise, n_classes, device='cpu')
            UniVRSE_score = entropy_from_logits_contrast(align_weak, align_noise, lam=LAMBDA)

            with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([idx, question, ref_answer, gen_answer,
                                 RadFlag, SeEnt, VASE_score, UniVRSE_score])

            del outputs, outputs_ori, outputs_vase, outputs_weak, outputs_noise
            del token_ids_ori, token_ids_vase, token_ids_weak, token_ids_noise
            del scores_ori, scores_vase, scores_weak, scores_noise
            del token_ids_ori_d, token_ids_vase_d, token_ids_weak_d, token_ids_noise_d
            del scores_ori_stacked, scores_vase_stacked, scores_weak_stacked, scores_noise_stacked
            del llogits_ori, llogits_vase, llogits_weak, llogits_noise
            del SeDist_ori, SeDist_vase, SeDist_weak, SeDist_noise
            del align_ori, align_vase, align_weak, align_noise, vase_raw
            del semantic_ids, all_texts
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default=os.getenv("UNIVRSE_MODEL_ID", "google/medgemma-4b-it"),
        help="VLM id (e.g. google/medgemma-4b-it or chaoyinshe/llava-med-v1.5-mistral-7b-hff).",
    )
    parser.add_argument(
        "--csv-file",
        default=os.getenv("UNIVRSE_OUTPUT_CSV", "outputs/radvqa_medgemma_hallscore.csv"),
        help="Output CSV with SE/VASE/RadFlag/UniVRSE scores.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional debug cap on number of VQA-RAD samples.",
    )
    return parser.parse_args()


# python VASE/main_univrse_det.py
if __name__ == '__main__':
    args = parse_args()
    main_pred_hallscore(args.model_id, args.csv_file, limit=args.limit)