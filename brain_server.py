# brain_server.py
# ПОЛНОСТЬЮ ВОССТАНОВЛЕННАЯ СТАБИЛЬНАЯ ВЕРСИЯ

#!/usr/bin/env python3
import os, sys, time, pickle
from multiprocessing.connection import Listener
from io import BytesIO
from PIL import Image
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from render_logic import NeuroRender
from vla_jepa_wrapper import VLA_JEPA_Wrapper

def main():
    print("🧠 STARTING NEURAL BRAIN SERVER (CUDA BACKEND) 🧠")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading aligned CLIP vision-text model for physical world analysis...")
    from transformers import CLIPModel, CLIPProcessor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    print("Loading heavy generative models (this takes a while, but only ONCE)...")
    render = NeuroRender(mode="lcm")
    vla_model = VLA_JEPA_Wrapper(render=render)
    
    vocab_atlas_cache = None
    
    address = ('localhost', 6000)
    listener = Listener(address, authkey=b'brain')
    print("🧠 Neural Brain Server is active and listening on localhost:6000")
    
    while True:
        try:
            conn = listener.accept()
            print("[SERVER] Client connected!")
            while True:
                try:
                    msg = conn.recv()
                    cmd = msg.get('cmd')
                    
                    if cmd == 'encode_prompt':
                        text = msg['text']
                        embeds = render.encode_prompt(text)
                        conn.send(embeds.cpu().numpy())
                        
                    elif cmd == 'encode_world_state':
                        img_bytes = msg['image_bytes']
                        img = Image.open(BytesIO(img_bytes))
                        state = vla_model.encode_world_state(img)
                        conn.send(state.cpu().numpy())
                        
                    elif cmd == 'get_vocab_atlas':
                        if vocab_atlas_cache is None:
                            print("[SERVER] Building Vocab Atlas from SD Text Encoder...")
                            tokenizer = render.pipe.tokenizer
                            text_model = render.pipe.text_encoder
                            
                            raw_embeds = text_model.get_input_embeddings().weight.detach().to(torch.float32)
                            vocab_size = raw_embeds.shape[0]

                            embed_matrix = raw_embeds / (torch.norm(raw_embeds, dim=-1, keepdim=True) + 1e-7)

                            token_words = []
                            for idx in range(vocab_size):
                                w = tokenizer.decode([idx]).strip().replace('</w>', '')
                                token_words.append(w if w else f"tok_{idx}")

                            _, _, V_pca = torch.pca_lowrank(embed_matrix, q=2)
                            coords_2d = torch.matmul(embed_matrix, V_pca[:, :2])
                            max_r = torch.max(torch.norm(coords_2d, dim=-1))
                            coords_2d = coords_2d / (max_r + 1e-6)

                            vocab_atlas_cache = {
                                'embed_matrix': embed_matrix.cpu().numpy(),
                                'coords_2d': coords_2d.cpu().numpy(),
                                'V_pca': V_pca[:, :2].cpu().numpy(),
                                'token_words': token_words,
                                'vocab_size': vocab_size
                            }
                            print(f"[SERVER] Vocab Atlas built and cached ({vocab_size} tokens).")
                        conn.send(vocab_atlas_cache)
                        
                    elif cmd == 'get_visual_similarities':
                        img_bytes = msg['image_bytes']
                        prompts = msg['prompts']
                        
                        img = Image.open(BytesIO(img_bytes))
                        inputs = clip_processor(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
                        
                        with torch.no_grad():
                            img_feats = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
                            text_feats = clip_model.get_text_features(input_ids=inputs['input_ids'])
                            
                            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                            
                            sims = (img_feats @ text_feats.T).squeeze(0).cpu().numpy()
                            
                        conn.send(sims)
                        
                    elif cmd == 'predict_future_state':
                        state_np = msg['current_state']
                        action_np = msg['action_token']
                        intent_np = msg['eeg_intent']
                        focus = msg['focus_level']
                        threat_np = msg.get('incoming_threat_token')
                        threat_lvl = msg.get('threat_level', 0.0)
                        
                        current_state = torch.tensor(state_np, dtype=vla_model.dtype, device=vla_model.device)
                        action_token = torch.tensor(action_np, dtype=vla_model.dtype, device=vla_model.device)
                        eeg_intent = torch.tensor(intent_np, dtype=vla_model.dtype, device=vla_model.device)
                        incoming_threat = torch.tensor(threat_np, dtype=vla_model.dtype, device=vla_model.device) if threat_np is not None else None
                        
                        future = vla_model.predict_future_state(
                            current_state, action_token, eeg_intent, focus, 
                            incoming_threat_token=incoming_threat, threat_level=threat_lvl
                        )
                        conn.send(future.cpu().numpy())
                        
                    elif cmd == 'generate':
                        embeds_np = msg.get('prompt_embeds')
                        prompt_text = msg.get('prompt')
                        img_bytes = msg.get('image_bytes')
                        strength = msg.get('strength', 0.5)
                        
                        embeds = torch.tensor(embeds_np, dtype=render.dtype, device=render.device) if embeds_np is not None else None
                        img = Image.open(BytesIO(img_bytes)) if img_bytes is not None else None
                        
                        out_img = render.generate(prompt=prompt_text, prompt_embeds=embeds, image=img, strength=strength)
                        
                        buf = BytesIO()
                        out_img.save(buf, format='JPEG')
                        conn.send(buf.getvalue())
                        
                except EOFError:
                    print("[SERVER] Client disconnected.")
                    break
                except Exception as e:
                    print(f"[SERVER] Error handling request: {e}")
                    try:
                        conn.send({'error': str(e)})
                    except:
                        break
        except Exception as e:
            print(f"[SERVER] Connection error: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
