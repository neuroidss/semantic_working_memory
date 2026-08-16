# vla_jepa_wrapper.py
# Полный код файла. Стабильное латентное предсказание будущего с защитой от NaN.

import sys
import os
import torch
import numpy as np
import traceback
import re
from huggingface_hub import hf_hub_download

current_file_path = os.path.dirname(os.path.abspath(__file__))
vla_jepa_path = None

for _ in range(4):
    potential_path = os.path.join(check_dir := current_file_path, "VLA-JEPA")
    if os.path.exists(potential_path):
        vla_jepa_path = potential_path
        break
    sibling_path = os.path.join(os.path.dirname(check_dir), "VLA-JEPA")
    if os.path.exists(sibling_path):
        vla_jepa_path = sibling_path
        break
    current_file_path = os.path.dirname(check_dir)

if vla_jepa_path:
    if vla_jepa_path not in sys.path:
        sys.path.insert(0, vla_jepa_path)
    print(f"[BrainEngine] Dynamically located VLA-JEPA package at: {vla_jepa_path}")
else:
    print("[BrainEngine] Warning: 'VLA-JEPA' folder not found. Running in standalone fallback mode.")

def sanitize_starvla_code(local_vla_path):
    vlm_dir = os.path.join(local_vla_path, "starVLA", "model", "modules", "vlm")
    files_to_patch = ["QWen2_5.py", "QWen3.py"]
    
    for fname in files_to_patch:
        fpath = os.path.join(vlm_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    code = f.read()
                patched_code = code.replace('attn_implementation="flash_attention_2"', 'attn_implementation="sdpa"')
                if code != patched_code:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        f.write(patched_code)
                    print(f"[VLA-JEPA] Patched {fname}: forced native 'sdpa' instead of 'flash_attention_2'")
            except Exception as e:
                print(f"[VLA-JEPA] Warning: Failed to patch {fname} codebase: {e}")

if vla_jepa_path:
    sanitize_starvla_code(vla_jepa_path)

HAS_STAR_VLA = False
VLA_JEPA_Class = None

try:
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA as _vla
    VLA_JEPA_Class = _vla
    print("[VLA-JEPA] VLA_JEPA class successfully imported!")
    HAS_STAR_VLA = True
except Exception as e:
    print(f"[VLA-JEPA] Warning: 'starVLA' framework import failed! Error: {e}")
    traceback.print_exc()
    HAS_STAR_VLA = False

def sanitize_config_yaml(yaml_path):
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = re.sub(r'["\']/home/[^"\']+/Qwen3-VL-2B-Instruct["\']', '"Qwen/Qwen3-VL-2B-Instruct"', content)
        content = re.sub(r'["\']/home/[^"\']+/vjepa2-vitl-fpc64-256["\']', '"facebook/vjepa2-vitl-fpc64-256"', content)
        replacements = {
            "/home/dataset-local/models/Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
            "/home/dataset-local/models/vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
            "/home/dataset-assist-0/algorithm/ginwind/models/vjepa2-vitl-fpc64-256": "facebook/vjepa2-vitl-fpc64-256",
            "/home/dataset-assist-0/algorithm/ginwind/models/Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
            "./playground/Pretrained_models/Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct"
        }
        for src, dst in replacements.items():
            content = content.replace(src, dst)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[VLA-JEPA] config.yaml successfully sanitized for public HuggingFace repos!")
    except Exception as e:
        print(f"[VLA-JEPA] Warning: Failed to sanitize config.yaml: {e}")

def download_vla_jepa_assets(local_vla_path):
    repo_id = "ginwind/VLA-JEPA"
    run_dir = os.path.join(local_vla_path, "SimplerEnv")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    yaml_path = os.path.join(run_dir, "config.yaml")
    json_path = os.path.join(run_dir, "dataset_statistics.json")
    pt_path = os.path.join(ckpt_dir, "VLA-JEPA-SimplerEnv.pt")
    if not os.path.exists(yaml_path):
        hf_hub_download(repo_id=repo_id, filename="SimplerEnv/config.yaml", local_dir=local_vla_path)
    sanitize_config_yaml(yaml_path)
    if not os.path.exists(json_path):
        hf_hub_download(repo_id=repo_id, filename="SimplerEnv/dataset_statistics.json", local_dir=local_vla_path)
    if not os.path.exists(pt_path):
        hf_hub_download(repo_id=repo_id, filename="SimplerEnv/checkpoints/VLA-JEPA-SimplerEnv.pt", local_dir=local_vla_path)
    return pt_path


def safe_normalize(t, dim=-1, eps=1e-5):
    norm = torch.norm(t, p=2, dim=dim, keepdim=True)
    return t / torch.clamp(norm, min=eps)


class VLA_JEPA_Wrapper:
    def __init__(self, render, device="cuda", remote_conn=None):
        self.device = device
        self.dtype = torch.float16
        self.render = render
        self.jepa_dim = 768
        self.remote_conn = remote_conn
        
        if self.remote_conn is not None:
            print("[VLA_JEPA] Running in REMOTE CLIENT mode (0.1s hot load!)")
            self.language_action_memory = {}
            self.model = None
            return
        
        self.model = None
        if HAS_STAR_VLA and vla_jepa_path and VLA_JEPA_Class is not None:
            try:
                checkpoint_path = download_vla_jepa_assets(vla_jepa_path)
                self.model = VLA_JEPA_Class.from_pretrained(checkpoint_path)
                self.jepa_dim = self.model.config.framework.vj2_model.get("jepa_dim", 2048)
                print("[VLA-JEPA] Model successfully loaded and active!")
            except Exception as e:
                print(f"[VLA-JEPA] Error loading checkpoints: {e}. Running in standby mode.")
        
        self.language_action_memory = {}

    def encode_language_action(self, text_instruction):
        if text_instruction in self.language_action_memory:
            return self.language_action_memory[text_instruction]
        action_token = self.render.encode_prompt(text_instruction)
        self.language_action_memory[text_instruction] = action_token
        return action_token

    def encode_world_state(self, pil_image):
        if self.remote_conn is not None:
            from io import BytesIO
            img_buf = BytesIO()
            pil_image.save(img_buf, format='JPEG')
            self.remote_conn.send({'cmd': 'encode_world_state', 'image_bytes': img_buf.getvalue()})
            arr = self.remote_conn.recv()
            return torch.tensor(arr, dtype=self.dtype, device=self.device)

        if self.model is not None:
            try:
                return self.model.encode_vision(pil_image)
            except: pass
        return torch.zeros((1, 77, self.jepa_dim), dtype=self.dtype, device=self.device)

    def project_120d_to_intent(self, raw_120d_numpy, seq_len=77):
        vec = torch.tensor(np.nan_to_num(raw_120d_numpy, nan=0.0), dtype=self.dtype, device=self.device)
        intent = vec.repeat(self.jepa_dim // 120 + 1)[:self.jepa_dim]
        intent_normalized = safe_normalize(intent, dim=0)
        return intent_normalized.unsqueeze(0).unsqueeze(0).expand(1, seq_len, self.jepa_dim)

    def predict_future_state(self, current_state, action_token, eeg_intent, focus_level, incoming_threat_token=None, threat_level=0.0):
        if self.remote_conn is not None:
            self.remote_conn.send({
                'cmd': 'predict_future_state',
                'current_state': current_state.cpu().numpy(),
                'action_token': action_token.cpu().numpy(),
                'eeg_intent': eeg_intent.cpu().numpy(),
                'focus_level': focus_level,
                'incoming_threat_token': incoming_threat_token.cpu().numpy() if incoming_threat_token is not None else None,
                'threat_level': threat_level
            })
            arr = self.remote_conn.recv()
            return torch.tensor(arr, dtype=self.dtype, device=self.device)

        # Санация весов во избежание Nan-утечек
        current_state = torch.nan_to_num(current_state, nan=0.0)
        action_token = torch.nan_to_num(action_token, nan=0.0)
        eeg_intent = torch.nan_to_num(eeg_intent, nan=0.0)

        if current_state.shape[1] != action_token.shape[1]:
            current_state_aligned = torch.nn.functional.interpolate(
                current_state.transpose(1, 2).to(torch.float32),
                size=action_token.shape[1],
                mode='linear',
                align_corners=False
            ).transpose(1, 2).to(self.dtype)
        else:
            current_state_aligned = current_state

        if action_token.shape[-1] != current_state_aligned.shape[-1]:
            diff = current_state_aligned.shape[-1] - action_token.shape[-1]
            if diff > 0:
                action_token_aligned = torch.nn.functional.pad(action_token, (0, diff))
                eeg_intent_aligned = torch.nn.functional.pad(eeg_intent, (0, diff))
            else:
                action_token_aligned = action_token[..., :current_state_aligned.shape[-1]]
                eeg_intent_aligned = eeg_intent[..., :current_state_aligned.shape[-1]]
        else:
            action_token_aligned = action_token
            eeg_intent_aligned = eeg_intent
            
        intent_norm = safe_normalize(eeg_intent_aligned, dim=-1)
        action_norm = safe_normalize(action_token_aligned, dim=-1)
        
        res_dim = min(intent_norm.shape[-1], action_norm.shape[-1])
        resonance = torch.sum(intent_norm[..., :res_dim] * action_norm[..., :res_dim], dim=-1, keepdim=True)
        
        # Интеграция воздействий внутри латентного предсказателя
        if incoming_threat_token is not None and threat_level > 0.05:
            if incoming_threat_token.shape[1] != current_state_aligned.shape[1]:
                threat_token_aligned = torch.nn.functional.interpolate(
                    incoming_threat_token.transpose(1, 2).to(torch.float32),
                    size=current_state_aligned.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).to(self.dtype)
            else:
                threat_token_aligned = incoming_threat_token

            if threat_token_aligned.shape[-1] != current_state_aligned.shape[-1]:
                diff_t = current_state_aligned.shape[-1] - threat_token_aligned.shape[-1]
                if diff_t > 0:
                    threat_token_aligned = torch.nn.functional.pad(threat_token_aligned, (0, diff_t))
                else:
                    threat_token_aligned = threat_token_aligned[..., :current_state_aligned.shape[-1]]

            # Суммируем волю кастера и силу попадания оппонента в латентном пространстве
            transition = (action_token_aligned * resonance * focus_level) + (threat_token_aligned * (0.45 * threat_level))
        else:
            transition = action_token_aligned * resonance * focus_level
        
        if self.model is not None:
            try:
                return self.model.predict_next_chunk(current_state_aligned, transition)
            except: pass
                
        return torch.nan_to_num(current_state_aligned + transition, nan=0.0)

    def translate_to_sd(self, jepa_state, current_pil_image, base_sd_embeds=None, res_factor=0.0):
        if base_sd_embeds is not None:
            res_dim = min(jepa_state.shape[-1], base_sd_embeds.shape[-1])
            jepa_state_resized = jepa_state[..., :res_dim]
            
            if jepa_state_resized.shape[1] != base_sd_embeds.shape[1]:
                jepa_state_resized = torch.nn.functional.interpolate(
                    jepa_state_resized.transpose(1, 2).to(torch.float32),
                    size=base_sd_embeds.shape[1],
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2).to(base_sd_embeds.dtype)
                
            return base_sd_embeds + (jepa_state_resized * (0.35 * res_factor))
        return jepa_state
