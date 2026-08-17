#!/usr/bin/env python3
"""
🧠 NeuroCanvas: Auto-Priority Real-Time Working Memory MusicGen Engine
Auto NI -20 Priority • Multi-Core CPU Affinity • Fused CUDA Matrix Pipeline
"""

import os
import sys
import time
import math
import queue
import threading
import numpy as np
import torch
import torch.nn.functional as F
import sounddevice as sd

from audiocraft.models import MusicGen
from audiocraft.models.lm import LMModel
from audiocraft.modules.transformer import StreamingMultiheadAttention, _get_attention_time_dimension
from audiocraft.modules.conditioners import T5Conditioner
from audiocraft.utils import utils

try:
    import pylsl
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print("❌ [LSL ERROR] pylsl is NOT installed in this environment!", flush=True)

# --- UNBUFFERED STDOUT ---
sys.stdout.reconfigure(line_buffering=True)
os.environ['LIBLSL_LOG_LEVEL'] = "-2"

# ==============================================================================
# 0. СИСТЕМНЫЙ АВТО-ПРИОРИТЕТ (NI -20) И МНОГОЯДЕРНОСТЬ CPU
# ==============================================================================
def set_process_realtime_priority():
    pid = os.getpid()
    # 1. Выставляем максимальный приоритет Linux NI -20
    try:
        os.setpriority(os.PRIO_PROCESS, 0, -20)
        print(f"⚡ [System] Auto-Priority applied: PID {pid} -> NI -20 (Real-Time PRI 0)", flush=True)
    except PermissionError:
        try:
            os.setpriority(os.PRIO_PROCESS, 0, -10)
            print(f"⚡ [System] Auto-Priority applied: PID {pid} -> NI -10", flush=True)
        except Exception as e:
            print(f"⚠️ [System] Note: Run with sudo or setcap to allow NI -20 without htop ({e})", flush=True)
            
    # 2. Разрешаем PyTorch задействовать несколько ядер CPU
    try:
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(min(4, num_cores))
        torch.set_num_interop_threads(2)
    except Exception:
        pass

# ==============================================================================
# GPU CONFIG & MAXIMUM AMPERE OPTIMIZATIONS (RTX 3060)
# ==============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

SAMPLE_RATE = 32000
FRAME_RATE = 50  # 50 токенов в секунду
NUM_CHANNELS = 16
CH_PER_PATCH = 16
FS = 250.0
BUF_SIZE = 256
NUM_DENSE_FREQS = 32
LATENT_DIM = 768

# Electrode Geometry (16-ch circular patch)
COORDS_16_X = np.array([10.14, 7.43, 2.75, 2.72, -2.72, -2.75, -7.42, -10.14, -10.14, -7.43, -2.75, -2.72, 2.72, 2.75, 7.43, 10.14], dtype=np.float32)
COORDS_16_Y = np.array([-2.72, -7.43, -4.77, -10.15, -10.14, -4.77, -7.42, -2.73, 2.72, 7.43, 4.76, 10.14, 10.15, 4.77, 7.42, 2.71], dtype=np.float32)
RAW_I_IDX, RAW_J_IDX = np.triu_indices(NUM_CHANNELS, k=1)
RAW_DX = (COORDS_16_X[RAW_J_IDX] - COORDS_16_X[RAW_I_IDX]).astype(np.float32)
RAW_DY = (COORDS_16_Y[RAW_J_IDX] - COORDS_16_Y[RAW_I_IDX]).astype(np.float32)
PAIR_ANGLES = np.arctan2(RAW_DY, RAW_DX)
SORTED_ORDER = np.argsort(PAIR_ANGLES)
I_IDX_GPU = torch.from_numpy(RAW_I_IDX[SORTED_ORDER]).to(DEVICE)
J_IDX_GPU = torch.from_numpy(RAW_J_IDX[SORTED_ORDER]).to(DEVICE)
NUM_PAIRS = len(I_IDX_GPU)

# Concurrency & Queues
stop_event = threading.Event()
audio_queue = queue.Queue(maxsize=40)
GLOBAL_ENGINE = None
CACHED_BASE_T5 = None

# ТЮНИНГ ДЛЯ РЕАЛТАЙМА 1.0X+ (140 BPM)
session_params = {
    "target_block": 2.4,    # 2.4с чанк для максимального КПД CUDA
    "target_context": 0.4,  # 0.4с (~1 бит на 140 BPM) — ультрабыстрый префикс
    "speed_safety_factor": 1.0,
    "target_buffer_count": 2,
    "prompt": "hypnotic psychedelic trance 140 bpm, punchy rolling bassline, driving kick drum, acid synth arpeggios, crisp hi-hats, studio mastering"
}

telemetry = {
    "rtf": 0.0,
    "speed": 1.0,
    "buffer_len": 0,
    "chunk_count": 0,
    "theta_freq": 6.0,
    "theta_sync": 0.0,
    "continuity": 0.0,
    "stream_name": "Scanning for LSL EEG...",
    "is_real": False
}

# ==============================================================================
# GPU-КРОССФЕЙДЕР И ПИК-ЛИМИТЕР
# ==============================================================================
class AudioCrossfaderGPU:
    def __init__(self, fade_len=512):
        self.fade_len = fade_len
        self.fade_in = torch.linspace(0.0, 1.0, fade_len, device=DEVICE, dtype=torch.float32)
        self.fade_out = torch.linspace(1.0, 0.0, fade_len, device=DEVICE, dtype=torch.float32)
        self.overlap = None

    def process(self, chunk: torch.Tensor) -> torch.Tensor:
        peak = torch.max(torch.abs(chunk))
        if peak > 0.92:
            chunk = chunk / (peak + 1e-5) * 0.92
            
        T = chunk.shape[-1]
        if T <= self.fade_len:
            return chunk
            
        if self.overlap is not None:
            chunk[..., :self.fade_len] = chunk[..., :self.fade_len] * self.fade_in + self.overlap * self.fade_out
            
        self.overlap = chunk[..., -self.fade_len:].clone()
        return chunk[..., :-self.fade_len]

# ==============================================================================
# 1. СТАТИЧЕСКИЙ KV-CACHE (0 ДИНАМИЧЕСКИХ АЛЛОКАЦИЙ В СЕКУНДУ)
# ==============================================================================
_orig_complete_kv = StreamingMultiheadAttention._complete_kv

def fast_complete_kv(self, k, v):
    if self.cross_attention or not self._is_streaming:
        return _orig_complete_kv(self, k, v)
        
    time_dim = _get_attention_time_dimension(self.memory_efficient)
    
    if 'static_k' not in self._streaming_state:
        shape_k = list(k.shape)
        shape_k[time_dim] = 1024  # Запас буфера на 1024 токенов
        self._streaming_state['static_k'] = torch.zeros(shape_k, device=k.device, dtype=k.dtype)
        if v is not k:
            self._streaming_state['static_v'] = torch.zeros(shape_k, device=v.device, dtype=v.dtype)
        self._streaming_state['curr_len'] = 0
        
    curr_len = self._streaming_state['curr_len']
    slen = k.shape[time_dim]
    static_k = self._streaming_state['static_k']
    
    if curr_len + slen > static_k.shape[time_dim]:
        return _orig_complete_kv(self, k, v)
        
    if time_dim == 2:
        static_k[:, :, curr_len:curr_len+slen, :] = k
    else:
        static_k[:, curr_len:curr_len+slen, :, :] = k
        
    if v is not k:
        static_v = self._streaming_state['static_v']
        if time_dim == 2:
            static_v[:, :, curr_len:curr_len+slen, :] = v
        else:
            static_v[:, curr_len:curr_len+slen, :, :] = v
        nv = static_v[:, :, :curr_len+slen, :] if time_dim == 2 else static_v[:, :curr_len+slen, :, :]
    else:
        nv = static_k[:, :, :curr_len+slen, :] if time_dim == 2 else static_k[:, :curr_len+slen, :, :]
        
    nk = static_k[:, :, :curr_len+slen, :] if time_dim == 2 else static_k[:, :curr_len+slen, :, :]
    self._streaming_state['curr_len'] = curr_len + slen
    return nk, nv

StreamingMultiheadAttention._complete_kv = fast_complete_kv

# ==============================================================================
# 2. СПЛАВЛЕННЫЙ SINGLE-BATCH ДИСПЕТЧЕР
# ==============================================================================
def fast_lm_generate(self,
                     prompt=None,
                     conditions=[],
                     num_samples=None,
                     max_gen_len=256,
                     use_sampling=True,
                     temp=1.0,
                     top_k=250,
                     top_p=0.0,
                     cfg_coef=1.0,
                     cfg_coef_beta=None,
                     two_step_cfg=None,
                     remove_prompts=False,
                     check=False,
                     callback=None) -> torch.Tensor:
    
    assert not self.training
    first_param = next(iter(self.parameters()))
    device = first_param.device

    if conditions:
        tokenized = self.condition_provider.tokenize(conditions)
        cfg_conditions = self.condition_provider(tokenized)
        cross_attn_src = cfg_conditions['description'][0]  # [1, L, D]
    else:
        cross_attn_src = None

    if prompt is None:
        num_samples = num_samples or 1
        prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

    B, K, T = prompt.shape
    start_offset = T
    pattern = self.pattern_provider.get_pattern(max_gen_len)
    unknown_token = -1

    gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
    gen_codes[..., :start_offset] = prompt
    gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
    start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)

    if not hasattr(self, '_fused_head_w'):
        self._fused_head_w = torch.stack([l.weight for l in self.linears], dim=0)  # [4, 2048, 1024]
        if self.linears[0].bias is not None:
            self._fused_head_b = torch.stack([l.bias for l in self.linears], dim=0)
        else:
            self._fused_head_b = None

    with self.streaming():
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]
        
        for offset in range(start_offset_sequence, gen_sequence_len):
            curr_sequence = gen_sequence[..., prev_offset:offset]
            
            input_ = (self.emb[0](curr_sequence[:, 0]) + 
                      self.emb[1](curr_sequence[:, 1]) + 
                      self.emb[2](curr_sequence[:, 2]) + 
                      self.emb[3](curr_sequence[:, 3]))
            
            out = self.transformer(input_, cross_attention_src=cross_attn_src)
            if self.out_norm:
                out = self.out_norm(out)
                
            out_last = out[:, -1, :].unsqueeze(0).expand(4, B, -1)  # [4, B, 1024]
            if self._fused_head_b is not None:
                logits = torch.baddbmm(self._fused_head_b.unsqueeze(1), out_last, self._fused_head_w.transpose(1, 2))
            else:
                logits = torch.bmm(out_last, self._fused_head_w.transpose(1, 2))
            
            logits = logits.permute(1, 0, 2)  # [B, 4, 2048]
            
            if use_sampling and temp > 0.0:
                top_logits, top_indices = torch.topk(logits / temp, k=50, dim=-1)
                top_probs = torch.softmax(top_logits, dim=-1)
                sample_idx = torch.multinomial(top_probs.view(-1, 50), 1)
                next_token = torch.gather(top_indices.view(-1, 50), -1, sample_idx).view(B, K, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id
            
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == unknown_token,
                next_token, gen_sequence[..., offset:offset+1]
            )
            prev_offset = offset
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)

    out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)
    out_start_offset = start_offset if remove_prompts else 0
    return out_codes[..., out_start_offset:max_gen_len]

LMModel.generate = fast_lm_generate

# ==============================================================================
# 3. PRE-ALLOCATED GPU MATRIX PHASE ENGINE
# ==============================================================================
class PureGPUMatrixPhaseEngine:
    def __init__(self, embed_matrix: torch.Tensor, num_freqs=NUM_DENSE_FREQS):
        self.num_freqs = num_freqs
        
        print("[NeuroGen] Pre-allocating CUDA Matrix Buffers at Startup...", flush=True)
        _, _, V_all = torch.pca_lowrank(embed_matrix, q=NUM_PAIRS)
        self.proj_120_to_768 = V_all[:, :NUM_PAIRS].T.contiguous().to(DEVICE)
        
        freqs = np.fft.fftfreq(BUF_SIZE, d=1.0 / FS).astype(np.float32)
        self.freqs_gpu = torch.from_numpy(freqs).to(DEVICE)
        
        notch = np.ones_like(freqs, dtype=np.float32)
        notch[(np.abs(freqs) >= 48.0) & (np.abs(freqs) <= 52.0)] = 0.0
        notch[(np.abs(freqs) >= 98.0) & (np.abs(freqs) <= 102.0)] = 0.0
        self.notch_gpu = torch.from_numpy(notch).to(DEVICE).view(1, BUF_SIZE)
        
        self.f_theta = (torch.exp(-0.5 * ((self.freqs_gpu - 6.0) / 1.5)**2) * 2.0).view(1, BUF_SIZE)
        self.f_theta[0, self.freqs_gpu < 0] = 0.0
        
        gamma_centers = torch.linspace(30.0, 85.0, num_freqs, device=DEVICE).view(num_freqs, 1, 1)
        freqs_3d = self.freqs_gpu.view(1, 1, BUF_SIZE)
        gamma_filters = torch.exp(-0.5 * ((freqs_3d - gamma_centers) / 4.5)**2) * 2.0
        gamma_filters[:, :, self.freqs_gpu < 0] = 0.0
        self.gamma_filters_batched = gamma_filters.contiguous()
        
        self.slot_angles = (-math.pi + (2.0 * math.pi / self.num_freqs) * (torch.arange(self.num_freqs, device=DEVICE, dtype=torch.float32) + 0.5)).view(self.num_freqs, 1)
        
        self.smoothed_latent_path = torch.zeros((num_freqs, LATENT_DIM), device=DEVICE, dtype=torch.float32)
        self.prev_future_phasor = torch.zeros(NUM_PAIRS, device=DEVICE, dtype=torch.cfloat)
        self.buf_gpu = torch.zeros((NUM_CHANNELS, BUF_SIZE), device=DEVICE, dtype=torch.float32)
        self.t_vec = torch.linspace(0, 1, BUF_SIZE, device=DEVICE)
        self.ch_phase = torch.linspace(0, 2 * math.pi, NUM_CHANNELS, device=DEVICE).view(NUM_CHANNELS, 1)
        self.t_sim = 0.0

    @torch.inference_mode()
    def process_frame(self, buffer_np, is_real=True):
        if is_real:
            self.buf_gpu.copy_(torch.from_numpy(buffer_np))
            self.buf_gpu.sub_(torch.mean(self.buf_gpu, dim=1, keepdim=True))
            fft_clean = torch.fft.fft(self.buf_gpu, dim=1).mul_(self.notch_gpu)
        else:
            self.t_sim += 0.06
            sig_theta = torch.sin(2 * math.pi * 6.0 * self.t_vec + self.ch_phase + self.t_sim * 4.0)
            sig_gamma = torch.sin(2 * math.pi * 45.0 * self.t_vec + self.ch_phase * 2.0 + self.t_sim * 12.0) * 0.4
            self.buf_gpu.copy_(sig_theta).add_(sig_gamma)
            fft_clean = torch.fft.fft(self.buf_gpu, dim=1)

        # 1. Theta Phase
        Z_theta_all = torch.fft.ifft(fft_clean * self.f_theta, dim=1)
        P_theta_all = Z_theta_all / (torch.abs(Z_theta_all) + 1e-12)
        mean_theta_phasor = torch.mean(P_theta_all, dim=0)
        phi_theta_global = torch.angle(mean_theta_phasor)
        theta_sync_R = float(torch.abs(mean_theta_phasor[-1]).item())

        dphi = (torch.diff(phi_theta_global) + math.pi) % (2.0 * math.pi) - math.pi
        inst_f_theta = float(torch.clamp(torch.mean(dphi[-32:]) * (FS / (2.0 * math.pi)), 3.5, 9.0).item())

        # 2. Batched Gamma Filtering [32, 16, 256]
        fft_expanded = fft_clean.unsqueeze(0)
        Z_gamma_all = torch.fft.ifft(fft_expanded * self.gamma_filters_batched, dim=-1)
        P_gamma_all = Z_gamma_all / (torch.abs(Z_gamma_all) + 1e-12)

        # 3. Phase Gating
        p_diff = phi_theta_global.unsqueeze(0) - self.slot_angles
        w = torch.exp(3.2 * torch.cos(p_diff))
        w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)

        cg = P_gamma_all[:, I_IDX_GPU, :] * torch.conj(P_gamma_all[:, J_IDX_GPU, :])
        psi_field = torch.sum(cg * w.unsqueeze(1), dim=-1)  # [32, 120]

        # 4. Anchor-Referenced iPLV
        past_anchor = psi_field[0:1, :]
        vine_iplv = torch.imag(psi_field * torch.conj(past_anchor))

        # 5. SVD Projection in VRAM
        raw_latent = torch.matmul(vine_iplv, self.proj_120_to_768)
        raw_latent = raw_latent / (torch.norm(raw_latent, dim=-1, keepdim=True) + 1e-6)
        self.smoothed_latent_path.mul_(0.85).add_(raw_latent, alpha=0.15)

        # 6. Continuity
        future_phasor = psi_field[-1]
        inter_chain = torch.real(torch.sum(past_anchor.squeeze(0) * torch.conj(self.prev_future_phasor)))
        chain_norm = (torch.norm(past_anchor) * torch.norm(self.prev_future_phasor) + 1e-6)
        chain_coherence = float(torch.clamp(inter_chain / chain_norm, -1.0, 1.0).item())
        self.prev_future_phasor.copy_(future_phasor)

        return {
            'chain_continuity': chain_coherence if is_real else 0.85,
            'theta_sync_R': theta_sync_R,
            'inst_f_theta': inst_f_theta
        }

# ==============================================================================
# 4. КЭШИРОВАННАЯ ИНЪЕКЦИЯ T5 (0 МС НА ТЕКСТ)
# ==============================================================================
_orig_t5_forward = T5Conditioner.forward

def neuro_t5_forward(self, inputs):
    global CACHED_BASE_T5, GLOBAL_ENGINE
    mask = inputs['attention_mask']
    
    if CACHED_BASE_T5 is None:
        with torch.set_grad_enabled(self.finetune), self.autocast:
            CACHED_BASE_T5 = self.t5(**inputs).last_hidden_state.detach().clone()
            
    embeds = CACHED_BASE_T5.clone()
    
    if GLOBAL_ENGINE is not None:
        drift = GLOBAL_ENGINE.smoothed_latent_path.unsqueeze(0).to(embeds.device).type(embeds.dtype)
        L = embeds.shape[1]
        
        if drift.shape[1] != L and drift.shape[1] > 1:
            drift = F.interpolate(drift.permute(0, 2, 1), size=L, mode='linear', align_corners=True).permute(0, 2, 1)
        elif drift.shape[1] == 1:
            drift = drift.expand(-1, L, -1)
            
        embeds = embeds + (drift * 0.35)

    embeds = self.output_proj(embeds.to(self.output_proj.weight))
    embeds = (embeds * mask.unsqueeze(-1))
    return embeds, mask

T5Conditioner.forward = neuro_t5_forward

# ==============================================================================
# 5. АДАПТИВНАЯ СИСТЕМА
# ==============================================================================
def adapt_system(gen_time, audio_duration, buffer_size, current_speed, current_block_duration):
    gen_time = max(gen_time, 0.000001)
    real_rtf = audio_duration / gen_time
    target_block = session_params['target_block']
    
    if current_block_duration < target_block: 
        current_block_duration *= 1.05
    else: 
        current_block_duration *= 0.95
    current_block_duration = max(0.5, min(8.0, current_block_duration))
    
    base_speed = real_rtf * session_params['speed_safety_factor']
    target_buf = session_params['target_buffer_count']
    
    if buffer_size == 0: 
        base_speed *= 0.70 
    elif buffer_size < target_buf: 
        base_speed *= 0.97
    elif buffer_size > target_buf + 1: 
        base_speed *= 1.03
        
    alpha = 0.10
    current_speed = (current_speed * (1 - alpha)) + (base_speed * alpha)
    current_speed = max(0.4, min(1.2, current_speed))
    return current_speed, current_block_duration, real_rtf

def resample_chunk_gpu(wav, speed):
    if abs(speed - 1.0) < 0.01: 
        return wav
    new_len = int(wav.shape[-1] / speed)
    if new_len < 1: 
        return wav
    return F.interpolate(wav.float(), size=new_len, mode='linear', align_corners=False)

# ==============================================================================
# 6. ТУРБО-ГЕНЕРАТОР (1.0x+ СТРИМИНГ)
# ==============================================================================
def generator_worker(model):
    global telemetry
    current_block_duration = session_params['target_block']
    current_speed = 1.0
    prompt = session_params['prompt']
    crossfader = AudioCrossfaderGPU(fade_len=512)
    
    print(f"⚡ [NeuroGen] 1.0x+ Real-Time Psytrance Generator Active: '{prompt}'", flush=True)
    with torch.inference_mode():
        model.set_generation_params(
            duration=current_block_duration, 
            cfg_coef=1.0, 
            use_sampling=True, 
            top_k=250, 
            temperature=1.0
        )
        
        init_audio, current_tokens = model.generate([prompt], progress=False, return_tokens=True)
        clean_init_gpu = crossfader.process(init_audio)
        audio_queue.put(clean_init_gpu[0, 0].cpu().float().numpy())
        
        count = 0
        while not stop_event.is_set():
            t0 = time.time()
            
            ctx_token_len = int(session_params['target_context'] * FRAME_RATE)
            prompt_tokens = current_tokens[..., -ctx_token_len:]
            
            gen_token_len = int(current_block_duration * FRAME_RATE)
            total_tokens_target = prompt_tokens.shape[-1] + gen_token_len
            total_duration_sec = total_tokens_target / FRAME_RATE
            
            model.set_generation_params(
                duration=total_duration_sec, 
                cfg_coef=1.0, 
                use_sampling=True, 
                top_k=250, 
                temperature=1.0
            )
            
            attributes = [model._prepare_tokens_and_attributes([prompt], None)[0][0]]
            with model.autocast:
                out_tokens = model.lm.generate(
                    prompt_tokens, 
                    attributes, 
                    max_gen_len=total_tokens_target,
                    **model.generation_params
                )
            
            current_tokens = out_tokens
            
            new_tokens = out_tokens[..., prompt_tokens.shape[-1]:]
            tokens_to_decode = out_tokens[..., -(new_tokens.shape[-1] + 2):]
            
            with torch.no_grad():
                decoded_audio = model.compression_model.decode(tokens_to_decode, None)
                
            new_chunk = decoded_audio[..., 1280:]
            
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.time() - t0
            audio_len = new_chunk.shape[-1] / SAMPLE_RATE
            
            current_speed, current_block_duration, real_rtf = adapt_system(
                dt, audio_len, audio_queue.qsize(), current_speed, current_block_duration
            )
            
            processed_gpu = resample_chunk_gpu(new_chunk, current_speed)
            clean_gpu = crossfader.process(processed_gpu)
            audio_queue.put(clean_gpu[0, 0].cpu().float().numpy())
            
            count += 1
            telemetry["rtf"] = real_rtf
            telemetry["speed"] = current_speed
            telemetry["buffer_len"] = audio_queue.qsize()
            telemetry["chunk_count"] = count
            
            s_name = telemetry["stream_name"]
            th_f = telemetry["theta_freq"]
            th_sync = telemetry["theta_sync"]
            is_r = "LIVE EEG" if telemetry["is_real"] else "SIM"
            
            print(f"[NeuroGen] Chunk {count:03d} | RTF: {real_rtf:.2f}x | PlaySpeed: {current_speed:.2f}x | Buf: {audio_queue.qsize()} | [{is_r}] {s_name} (θ: {th_f:.1f}Hz, R: {th_sync*100:.0f}%)", flush=True)

def player_worker():
    print("🔈 [NeuroGen] Audio Player buffer starting...", flush=True)
    while audio_queue.qsize() < session_params['target_buffer_count'] and not stop_event.is_set(): 
        time.sleep(0.05)
        
    print("🔈 [NeuroGen] Audio Output Active (Hi-Fi 32kHz Clean Stream).", flush=True)
    with sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', latency='high') as stream:
        while not stop_event.is_set():
            try:
                data_np = audio_queue.get(timeout=1.0)
                data_np = np.ascontiguousarray(data_np)
                stream.write(data_np.tobytes())
            except queue.Empty:
                pass
            except Exception:
                time.sleep(0.05)

# ==============================================================================
# 7. ASYNC LSL INGESTION & 5 HZ DSP WORKER
# ==============================================================================
class AsyncLSLManager:
    def __init__(self):
        self.inlet = None
        self.raw_buffer = np.zeros((NUM_CHANNELS, BUF_SIZE), dtype=np.float32)
        self.is_connected = False
        self.stream_name = "Scanning for LSL..."
        self.lock = threading.Lock()
        if HAS_LSL:
            threading.Thread(target=self._scan_loop, daemon=True).start()

    def _scan_loop(self):
        while not stop_event.is_set():
            if not self.is_connected:
                try:
                    streams = resolve_streams(wait_time=1.0)
                    for s in streams:
                        s_name = s.name()
                        s_type = s.type()
                        if 'FreeEEG' in s_name or 'EEG' in s_name or s_type.upper() == 'EEG':
                            inlet = StreamInlet(s, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                            with self.lock:
                                self.inlet = inlet
                                self.is_connected = True
                                self.stream_name = s_name
                            print(f"\n⚡ [LSL] CONNECTED to {s_name}!\n", flush=True)
                            break
                except Exception:
                    pass
            time.sleep(2.0)

    def pull(self):
        if not HAS_LSL or not self.is_connected or self.inlet is None:
            return self.raw_buffer, False, self.stream_name
        with self.lock:
            try:
                chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                if chunk:
                    arr = np.array(chunk, dtype=np.float32).T[:NUM_CHANNELS, :]
                    n = arr.shape[1]
                    if n >= BUF_SIZE: 
                        self.raw_buffer = arr[:, -BUF_SIZE:]
                    else:
                        self.raw_buffer = np.roll(self.raw_buffer, -n, axis=1)
                        self.raw_buffer[:, -n:] = arr
                return self.raw_buffer, True, self.stream_name
            except Exception:
                self.inlet = None
                self.is_connected = False
                return self.raw_buffer, False, "Disconnected"

def dsp_worker(lsl_mgr):
    global telemetry
    
    while not stop_event.is_set():
        raw_buf, is_real, stream_name = lsl_mgr.pull()
        
        # 100% Batched GPU DSP расчет (5 Hz)
        res = GLOBAL_ENGINE.process_frame(raw_buf, is_real=is_real)
        
        telemetry["stream_name"] = stream_name
        telemetry["is_real"] = is_real
        telemetry["theta_sync"] = res['theta_sync_R']
        telemetry["theta_freq"] = res['inst_f_theta']
        telemetry["continuity"] = res['chain_continuity']
        
        time.sleep(0.20)  # 5 Hz

# ==============================================================================
# 8. MAIN ENTRY POINT
# ==============================================================================
def main():
    global GLOBAL_ENGINE
    
    # 0. Автоматически повышаем приоритет процесса до NI -20 (PRI 0)
    set_process_realtime_priority()
    
    print("=" * 70)
    print("🧠 NEUROCANVAS: 1.0x+ REAL-TIME PSYTRANCE ENGINE (AUTO-PRIORITY NI -20)")
    print("=" * 70)

    # 1. Загрузка MusicGen в FP16
    print(f"[NeuroGen] Loading MusicGen Small on {DEVICE} in FP16...", flush=True)
    m = MusicGen.get_pretrained('facebook/musicgen-small', device=DEVICE)
    m.lm.eval()
    m.compression_model.eval()
    if DEVICE.type == 'cuda':
        m.lm.to(torch.float16)
        m.compression_model.to(torch.float16)
        
    print("[NeuroGen] Model Loaded (Fused BMM Heads & Static KV-Cache).", flush=True)

    # 2. Инициализация SVD Phase Engine в VRAM
    t5_model = m.lm.condition_provider.conditioners['description'].t5
    raw_embeds = t5_model.shared.weight.detach().to(torch.float32)
    embed_matrix = raw_embeds / (torch.norm(raw_embeds, dim=-1, keepdim=True) + 1e-7)
    
    GLOBAL_ENGINE = PureGPUMatrixPhaseEngine(embed_matrix=embed_matrix, num_freqs=NUM_DENSE_FREQS)
    print("[NeuroGen] Pure GPU Matrix Phase Engine Ready.", flush=True)

    # 3. Запуск фоновых потоков
    lsl_mgr = AsyncLSLManager()
    t_dsp = threading.Thread(target=dsp_worker, args=(lsl_mgr,), daemon=True)
    t_gen = threading.Thread(target=generator_worker, args=(m,), daemon=True)
    t_play = threading.Thread(target=player_worker, daemon=True)

    t_dsp.start()
    t_gen.start()
    t_play.start()

    print("\n[NeuroGen] Real-Time 140 BPM Psytrance Online! Press Ctrl+C to exit.\n", flush=True)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[NeuroGen] Stopping engine gracefully...", flush=True)
        stop_event.set()
        t_gen.join(timeout=2)
        t_play.join(timeout=2)
        print("[NeuroGen] Shutdown complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()
