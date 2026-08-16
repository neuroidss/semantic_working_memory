#!/usr/bin/env python3
"""
NeuroCanvas: High-Detail Pure Latent Walk Engine (v21)
Full LCM Multi-Step Refinement (Strength ~ 0.62-0.72) for deep artistic detail.
Zero artificial noise. Fast OpenCV C++ I/O.
"""

import os
os.environ['LIBLSL_LOG_LEVEL'] = "-2"
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import sys
import time
import math
import threading
from typing import Tuple, Optional, List
from multiprocessing.connection import Client

import numpy as np
import cv2
import pygame
import torch
import torch.nn.functional as F

try:
    import pylsl
    from pylsl import StreamInlet, resolve_streams
    HAS_LSL = True
except ImportError:
    HAS_LSL = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available(): torch.backends.cudnn.benchmark = True

SCREEN_W, SCREEN_H = 1440, 840
NUM_CHANNELS = 16
CH_PER_PATCH = 16
FS = 250.0
BUF_SIZE = 256
NUM_DENSE_FREQS = 32
LATENT_DIM = 768

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

# ==============================================================================
# ЧИСТАЯ ХИРУРГИЯ ЦВЕТА (БЕЗ ШУМА, ТОЛЬКО БАЛАНС)
# ==============================================================================
def apply_clean_surgery_bgr(bgr: np.ndarray, color_nerf: float = 0.8) -> np.ndarray:
    res = bgr.astype(np.float32)
    # Подавление паразитного зеленого канала
    mu = np.mean(res, axis=(0, 1))
    target_g = (mu[0] + mu[2]) * 0.5
    if mu[1] > target_g:
        res[:, :, 1] -= (mu[1] - target_g) * color_nerf
        
    # Мягкий якорь контраста
    mu_t, std_t = cv2.meanStdDev(res)
    target_std = np.maximum(44.0, std_t)
    target_mu = np.clip(mu_t, 90.0, 155.0)
    
    res = (res - mu_t.reshape(1, 1, 3)) * (target_std / (std_t + 1e-5)).reshape(1, 1, 3) + target_mu.reshape(1, 1, 3)
    return np.clip(res, 0, 255).astype(np.uint8)

class FullVocabularyAtlas:
    def __init__(self, remote_conn):
        if remote_conn is not None:
            try:
                remote_conn.send({'cmd': 'get_vocab_atlas'})
                atlas_data = remote_conn.recv()
                self.vocab_size = atlas_data['vocab_size']
                self.embed_matrix = torch.from_numpy(atlas_data['embed_matrix']).to(DEVICE)
                self.coords_2d_gpu = torch.from_numpy(atlas_data['coords_2d']).to(DEVICE)
                self.V_pca_gpu = torch.from_numpy(atlas_data['V_pca']).to(DEVICE)
                self.token_words = atlas_data['token_words']
                return
            except Exception:
                pass
        self.vocab_size = 49408
        raw_embeds = torch.randn(self.vocab_size, LATENT_DIM, device=DEVICE)
        self.embed_matrix = raw_embeds / (torch.norm(raw_embeds, dim=-1, keepdim=True) + 1e-7)
        self.token_words = [f"dim_{i}" for i in range(self.vocab_size)]
        _, _, V_pca = torch.pca_lowrank(self.embed_matrix, q=2)
        coords_2d = torch.matmul(self.embed_matrix, V_pca[:, :2])
        self.coords_2d_gpu = coords_2d / (torch.max(torch.norm(coords_2d, dim=-1)) + 1e-6)
        self.V_pca_gpu = V_pca[:, :2]

class FullVocabPhaseEngine:
    def __init__(self, atlas: FullVocabularyAtlas, num_freqs=NUM_DENSE_FREQS, out_w=1080, out_h=330):
        self.atlas = atlas
        self.num_freqs = num_freqs
        self.out_w = out_w
        self.out_h = out_h
        freqs = np.fft.fftfreq(BUF_SIZE, d=1.0/FS).astype(np.float32)
        self.freqs_gpu = torch.from_numpy(freqs).to(DEVICE)
        notch = np.ones_like(freqs, dtype=np.float32)
        notch[(np.abs(freqs) >= 48.0) & (np.abs(freqs) <= 52.0)] = 0.0
        notch[(np.abs(freqs) >= 98.0) & (np.abs(freqs) <= 102.0)] = 0.0
        self.notch_gpu = torch.from_numpy(notch).to(DEVICE).view(1, BUF_SIZE)
        self.f_theta = (torch.exp(-0.5 * ((self.freqs_gpu - 6.0) / 1.5)**2) * 2.0).view(1, BUF_SIZE)
        self.f_theta[0, self.freqs_gpu < 0] = 0.0
        self.gamma_centers = np.linspace(30.0, 85.0, num_freqs, dtype=np.float32)
        self.gamma_filters = []
        for fc in self.gamma_centers:
            f_b = (torch.exp(-0.5 * ((self.freqs_gpu - fc) / 4.5)**2) * 2.0).view(1, BUF_SIZE)
            f_b[0, self.freqs_gpu < 0] = 0.0
            self.gamma_filters.append(f_b)
            
        # SVD Матрица осей словаря [120, 768]
        _, _, V_all = torch.pca_lowrank(self.atlas.embed_matrix, q=NUM_PAIRS)
        self.proj_120_to_768 = V_all[:, :NUM_PAIRS].T.to(DEVICE)

        self.smoothed_latent_path = torch.zeros((num_freqs, LATENT_DIM), device=DEVICE, dtype=torch.float32)
        self.prev_future_phasor = torch.zeros(NUM_PAIRS, device=DEVICE, dtype=torch.cfloat)

    @torch.inference_mode()
    def process_frame(self, buffer_np):
        centered_np = np.nan_to_num(buffer_np - np.mean(buffer_np, axis=1, keepdims=True))
        buf_gpu = torch.from_numpy(centered_np).to(DEVICE, non_blocking=True)
        fft_clean_gpu = torch.fft.fft(buf_gpu, dim=1) * self.notch_gpu

        Z_theta_all = torch.fft.ifft(fft_clean_gpu * self.f_theta, dim=1)
        P_theta_all = Z_theta_all / (torch.abs(Z_theta_all) + 1e-12)
        mean_theta_phasor = torch.mean(P_theta_all, dim=0)
        phi_theta_global = torch.angle(mean_theta_phasor)
        theta_sync_R = float(torch.abs(mean_theta_phasor[-1]).item())

        dphi = (torch.diff(phi_theta_global) + math.pi) % (2.0 * math.pi) - math.pi
        inst_f_theta = float(torch.clamp(torch.mean(dphi[-32:]) * (FS / (2.0 * math.pi)), 3.5, 9.0).item())

        slot_angles = -math.pi + (2.0 * math.pi / self.num_freqs) * (torch.arange(self.num_freqs, device=DEVICE, dtype=torch.float32) + 0.5)
        psi_dense_list = []
        for k in range(self.num_freqs):
            Z_k = torch.fft.ifft(fft_clean_gpu * self.gamma_filters[k], dim=1)
            P_k = Z_k / (torch.abs(Z_k) + 1e-12)
            p_diff = phi_theta_global - slot_angles[k]
            w_k = torch.exp(3.2 * torch.cos(p_diff))
            w_k = w_k / (torch.sum(w_k) + 1e-6)
            cg_k = P_k[I_IDX_GPU, :] * torch.conj(P_k[J_IDX_GPU, :])
            psi_k = torch.sum(cg_k * w_k.unsqueeze(0), dim=1)
            psi_dense_list.append(psi_k)

        psi_field_tensor = torch.stack(psi_dense_list, dim=0)
        past_anchor = psi_field_tensor[0]
        vine_rel_cross = psi_field_tensor * torch.conj(past_anchor.unsqueeze(0))
        vine_iplv = torch.imag(vine_rel_cross)
        
        # Проекция на 768-D SVD базис
        raw_latent_path = torch.matmul(vine_iplv, self.proj_120_to_768)
        raw_latent_path = raw_latent_path / (torch.norm(raw_latent_path, dim=-1, keepdim=True) + 1e-6)
        self.smoothed_latent_path = self.smoothed_latent_path * 0.85 + raw_latent_path * 0.15

        vine_2d_pts = torch.matmul(self.smoothed_latent_path, self.atlas.V_pca_gpu) * 2.2

        future_phasor = psi_field_tensor[-1]
        inter_chain = torch.real(torch.sum(past_anchor * torch.conj(self.prev_future_phasor)))
        chain_norm = (torch.norm(past_anchor) * torch.norm(self.prev_future_phasor) + 1e-6)
        chain_coherence = float(torch.clamp(inter_chain / chain_norm, -1.0, 1.0).item())
        self.prev_future_phasor = future_phasor.clone()

        # Фазовое поле (GPU)
        angles = torch.angle(psi_field_tensor) 
        mags = torch.abs(psi_field_tensor)     
        H = (angles + math.pi) / (2.0 * math.pi) 
        V = torch.clamp(mags * 2.8, 0.12, 1.0)    
        r = torch.clamp(torch.abs(H * 6.0 - 3.0) - 1.0, 0.0, 1.0)
        g = torch.clamp(2.0 - torch.abs(H * 6.0 - 2.0), 0.0, 1.0)
        b = torch.clamp(2.0 - torch.abs(H * 6.0 - 4.0), 0.0, 1.0)
        
        rgb_tensor = torch.stack([r, g, b], dim=-1) * V.unsqueeze(-1) * 255.0 
        img_bhwc = rgb_tensor.permute(2, 0, 1).unsqueeze(0) 
        smooth_field_gpu = F.interpolate(img_bhwc, size=(self.out_h, self.out_w), mode='bicubic', align_corners=False)
        smooth_field_uint8 = smooth_field_gpu.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)

        return {
            'chain_continuity': chain_coherence,
            'theta_sync_R': theta_sync_R,
            'inst_f_theta': inst_f_theta,
            'latent_path': self.smoothed_latent_path,
            'vine_2d_pts': vine_2d_pts.cpu().numpy(),
            'rgb_field_buffer': smooth_field_uint8.cpu().numpy() 
        }

class AsyncLSLManager:
    def __init__(self):
        self.inlet = None
        self.raw_buffer = np.zeros((CH_PER_PATCH, BUF_SIZE), dtype=np.float32)
        self.is_connected = False
        self.stream_name = "Searching LSL..."
        self.lock = threading.Lock()
        self.running = True
        if HAS_LSL:
            self.thread = threading.Thread(target=self._scan_loop, daemon=True)
            self.thread.start()

    def _scan_loop(self):
        while self.running:
            if self.inlet is None:
                try:
                    streams = resolve_streams(wait_time=0.4)
                    eeg_streams = [s for s in streams if s.type() == 'EEG' or 'FreeEEG' in s.name()]
                    if eeg_streams:
                        target = eeg_streams[0]
                        inlet = StreamInlet(target, max_buflen=1, max_chunklen=BUF_SIZE, recover=True)
                        with self.lock:
                            self.inlet = inlet
                            self.is_connected = True
                            self.stream_name = f"{target.name()}"
                except Exception: pass
            time.sleep(1.5)

    def pull_data(self):
        if not HAS_LSL or self.inlet is None: return self.raw_buffer, False, "Simulation Mode"
        with self.lock:
            try:
                chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=BUF_SIZE)
                if chunk:
                    n = len(chunk)
                    new_arr = np.array(chunk, dtype=np.float32).T
                    n_ch = min(CH_PER_PATCH, new_arr.shape[0])
                    if n >= BUF_SIZE: self.raw_buffer[:n_ch, :] = new_arr[:n_ch, -BUF_SIZE:]
                    else:
                        self.raw_buffer = np.roll(self.raw_buffer, -n, axis=1)
                        self.raw_buffer[:n_ch, -n:] = new_arr[:n_ch, :]
                return self.raw_buffer.copy(), True, self.stream_name
            except Exception:
                self.inlet = None
                self.is_connected = False
                return self.raw_buffer, False, "Disconnected"

# ==============================================================================
# HIGH-DETAIL DIFFUSION CLIENT (STRENGTH ~ 0.62-0.72 ДЛЯ ГЛУБОКОЙ ПРОРИСОВКИ)
# ==============================================================================
class ClosedLoopDiffusionClient:
    def __init__(self, remote_conn, render_w=384, render_h=288):
        self.remote_conn = remote_conn
        
        dummy_bgr = np.random.randint(110, 150, (render_h, render_w, 3), dtype=np.uint8)
        self.current_bgr = dummy_bgr
        self.latest_rgb_bytes = cv2.cvtColor(dummy_bgr, cv2.COLOR_BGR2RGB).tobytes()
        
        self.target_latent_path = None
        
        # 🔥 ПОЛНАЯ СИЛА ДИФФУЗИИ ДЛЯ МАКСИМАЛЬНОЙ ДЕТАЛИЗАЦИИ 🔥
        self.render_strength = 0.64
        self.fps = 0.0
        self.lock = threading.Lock()
        self.running = True
        
        self.neutral_embeds = None
        if self.remote_conn is not None:
            try:
                self.remote_conn.send({'cmd': 'encode_prompt', 'text': ""})
                neutral_np = self.remote_conn.recv()
                self.neutral_embeds = torch.tensor(neutral_np, device=DEVICE, dtype=torch.float32)
            except Exception as e:
                print(f"[DirectClient] Error: {e}")
        
        self.thread = threading.Thread(target=self._render_loop, daemon=True)
        self.thread.start()

    def update_intent_continuous(self, latent_path: torch.Tensor, continuity: float):
        with self.lock:
            self.target_latent_path = latent_path
            # Диапазон 0.60 - 0.72 для глубокой художественной прорисовки
            self.render_strength = min(0.72, 0.60 + 0.12 * continuity)

    def _render_loop(self):
        frame_times = []
        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, 75,
            cv2.IMWRITE_JPEG_OPTIMIZE, 0
        ]
        
        while self.running:
            if self.remote_conn is None or self.target_latent_path is None or self.neutral_embeds is None:
                time.sleep(0.005); continue

            t0 = time.time()
            with self.lock:
                brain_tensor = self.target_latent_path.clone()
                cur_bgr = self.current_bgr.copy()
                st = self.render_strength

            # 1. 32 частоты тета-гамма цикла -> 77 слотов внимания
            brain_seq = brain_tensor.unsqueeze(0).permute(0, 2, 1) # [1, 768, 32]
            brain_drift_77 = F.interpolate(brain_seq, size=77, mode='linear', align_corners=True).permute(0, 2, 1)
            brain_drift = F.normalize(brain_drift_77, dim=-1) * 2.5
            prompt_embeds = (self.neutral_embeds + brain_drift).cpu().numpy()

            # 2. Быстрое сжатие C++ OpenCV
            success, enc_buf = cv2.imencode('.jpg', cur_bgr, encode_params)
            if not success: continue

            try:
                self.remote_conn.send({
                    'cmd': 'generate',
                    'prompt_embeds': prompt_embeds,
                    'image_bytes': enc_buf.tobytes(),
                    'strength': st
                })
                
                img_bytes = self.remote_conn.recv()
                
                if isinstance(img_bytes, bytes):
                    raw_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                    
                    if raw_bgr is not None:
                        # Чистая хирургия цвета (без искусственного шума)
                        healed_bgr = apply_clean_surgery_bgr(raw_bgr, color_nerf=0.8)
                        rgb_bytes = cv2.cvtColor(healed_bgr, cv2.COLOR_BGR2RGB).tobytes()
                        
                        with self.lock:
                            self.current_bgr = healed_bgr
                            self.latest_rgb_bytes = rgb_bytes

                dt = time.time() - t0
                frame_times.append(dt)
                if len(frame_times) > 8: frame_times.pop(0)
                self.fps = 1.0 / (np.mean(frame_times) + 1e-6)
            except Exception:
                time.sleep(0.02)

    def get_latest_frame(self) -> Tuple[bytes, float]:
        with self.lock: return self.latest_rgb_bytes, self.fps

def run_app():
    pygame.init()
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.SCALED
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags, vsync=0)
    pygame.display.set_caption("NeuroCanvas: Full Strength Detail Manifold Walk")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("consolas", 12)
    bold_font = pygame.font.SysFont("consolas", 14, bold=True)
    header_font = pygame.font.SysFont("consolas", 17, bold=True)
    large_font = pygame.font.SysFont("consolas", 19, bold=True)

    remote_conn = None
    try:
        remote_conn = Client(('localhost', 6000), authkey=b'brain')
        print("⚡ [DirectClient] Linked to Port 6000!")
    except Exception as e: 
        print(f"Cannot connect to brain_server on port 6000: {e}")

    atlas = FullVocabularyAtlas(remote_conn=remote_conn)
    lsl_mgr = AsyncLSLManager()
    
    field_w, field_h = 1080, 330
    engine = FullVocabPhaseEngine(atlas=atlas, num_freqs=NUM_DENSE_FREQS, out_w=field_w, out_h=field_h)
    diffuser = ClosedLoopDiffusionClient(remote_conn=remote_conn, render_w=384, render_h=288)

    sample_indices = np.random.choice(atlas.vocab_size, 800, replace=False)
    star_coords_2d = atlas.coords_2d_gpu[sample_indices].cpu().numpy()

    t_sim = 0.0
    running = True
    
    while running:
        clock.tick(200)
        t_sim += 0.05

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        raw_buf, is_real, stream_name = lsl_mgr.pull_data()
        
        if is_real:
            res = engine.process_frame(raw_buf)
            chain_continuity = res['chain_continuity']
            theta_sync_R = res['theta_sync_R']
            inst_f_theta = res['inst_f_theta']
            latent_path = res['latent_path']
            vine_2d_pts = res['vine_2d_pts']
            rgb_field_buffer = res['rgb_field_buffer']
        else:
            sim_latent = torch.randn(NUM_DENSE_FREQS, LATENT_DIM, device=DEVICE)
            sim_latent = sim_latent / torch.norm(sim_latent, dim=-1, keepdim=True)
            latent_path = sim_latent
            v_2d = torch.matmul(sim_latent, atlas.V_pca_gpu) * 2.2
            vine_2d_pts = v_2d.cpu().numpy()
            chain_continuity = 0.85
            theta_sync_R = 0.95
            inst_f_theta = 6.0
            rgb_field_buffer = np.full((field_h, field_w, 3), 20, dtype=np.uint8)

        diffuser.update_intent_continuous(latent_path, chain_continuity)

        latest_bytes, gen_fps = diffuser.get_latest_frame()
        w_surf = pygame.image.frombuffer(latest_bytes, (384, 288), 'RGB')

        screen.fill((7, 10, 15))

        status_color = (0, 255, 120) if is_real else (255, 180, 0)
        screen.blit(header_font.render(f"🧠 NEURO-CANVAS: HIGH-DETAIL LATENT STREAM ({gen_fps:.1f} FPS)", True, (0, 255, 200)), (30, 15))
        screen.blit(font.render(f"EEG: {stream_name} | θ Carrier: {inst_f_theta:.1f} Hz (Sync R={theta_sync_R*100:.0f}%) | Strength: {diffuser.render_strength:.2f}", True, status_color), (30, 42))

        # 1. Генерация высокого разрешения
        dx, dy = 30, 75
        screen.blit(w_surf, (dx, dy))
        pygame.draw.rect(screen, (0, 255, 200), (dx, dy, 384, 288), 2)

        # 2. Звездный радар
        panel_x = 445
        panel_w = 965
        pygame.draw.rect(screen, (12, 17, 26), (panel_x, dy, panel_w, 288))
        pygame.draw.rect(screen, (0, 255, 255), (panel_x, dy, panel_w, 288), 1)

        c_x = int(panel_x + 180)
        c_y = int(dy + 144)
        comp_r = 115
        pygame.draw.circle(screen, (25, 35, 50), (c_x, c_y), comp_r, 1)
        pygame.draw.circle(screen, (35, 50, 70), (c_x, c_y), comp_r // 2, 1)

        for s_i in range(len(star_coords_2d)):
            sx = int(c_x + star_coords_2d[s_i, 0] * (comp_r - 10))
            sy = int(c_y + star_coords_2d[s_i, 1] * (comp_r - 10))
            screen.set_at((sx, sy), (40, 60, 80))

        distances = np.linalg.norm(vine_2d_pts, axis=1)
        max_reach = float(np.max(distances)) + 1e-6
        scale_fit = (comp_r - 10) / max(max_reach, comp_r - 10)

        screen_pts = []
        for idx in range(len(vine_2d_pts)):
            px = int(c_x + vine_2d_pts[idx, 0] * scale_fit)
            py = int(c_y + vine_2d_pts[idx, 1] * scale_fit)
            screen_pts.append((px, py))

        for idx in range(len(screen_pts) - 1):
            p1 = screen_pts[idx]
            p2 = screen_pts[idx + 1]
            t_ratio = idx / float(len(screen_pts) - 1)
            col_r = int(np.clip(255 * (t_ratio ** 1.2), 0, 255))
            col_b = int(np.clip(255 * (1.0 - t_ratio * 0.5), 0, 255))
            thickness = max(1, int(6.0 * (1.0 - t_ratio * 0.65)))
            pygame.draw.line(screen, (col_r, 100, col_b), p1, p2, thickness)

        tip_pt = screen_pts[-1]
        pygame.draw.circle(screen, (255, 100, 255), tip_pt, 7)
        pygame.draw.circle(screen, (255, 255, 255), tip_pt, 3)

        info_x = panel_x + 350
        screen.blit(large_font.render("HIGH-FIDELITY LATENT MANIFOLD WALK:", True, (0, 255, 255)), (info_x, dy + 25))
        screen.blit(font.render(f"Markov Continuity: {chain_continuity*100:5.1f}%", True, (0, 255, 120)), (info_x, dy + 50))
        screen.blit(bold_font.render("Full Diffusion Depth: Deep Detail & Contrast", True, (255, 200, 100)), (info_x, dy + 85))
        screen.blit(font.render("Clean SVD Geometry. Ready for RTX 40-series deployment.", True, (180, 180, 180)), (info_x, dy + 110))

        # 3. Фазовое поле
        bottom_y = 390
        panel_bot_h = 420
        pygame.draw.rect(screen, (12, 17, 26), (30, bottom_y, 1380, panel_bot_h))
        pygame.draw.rect(screen, (0, 255, 200), (30, bottom_y, 1380, panel_bot_h), 1)

        screen.blit(bold_font.render("SPATIALLY TRANSPOSED PHASE FIELD (GPU ACCELERATED | X: 0° -> 360° | Y: 30 -> 85 Hz):", True, (0, 255, 200)), (45, bottom_y + 12))

        field_x, field_y = 260, bottom_y + 40
        field_surf = pygame.image.frombuffer(rgb_field_buffer.tobytes(), (field_w, field_h), 'RGB')
        screen.blit(field_surf, (field_x, field_y))
        pygame.draw.rect(screen, (0, 200, 255), (field_x, field_y, field_w, field_h), 1)

        screen.blit(bold_font.render("85 Hz (Future)", True, (255, 100, 255)), (110, field_y + 5))
        screen.blit(bold_font.render("58 Hz (Mid)", True, (0, 255, 200)), (135, field_y + field_h // 2 - 8))
        screen.blit(bold_font.render("30 Hz (Past)", True, (0, 200, 255)), (125, field_y + field_h - 22))

        pygame.display.flip()

    lsl_mgr.running = False
    diffuser.running = False
    pygame.quit()

if __name__ == '__main__':
    run_app()
