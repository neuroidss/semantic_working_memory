# render_logic.py
import torch, cv2, numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, AutoPipelineForImage2Image, LCMScheduler, AutoencoderTiny

GW, GH = 512, 384

class NeuroRender:
    def __init__(self, mode="lcm", compile_unet=False, remote_conn=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16
        self.mode = mode.lower()
        self.remote_conn = remote_conn
        
        if self.remote_conn is not None:
            print("[NeuroRender] Active in REMOTE CLIENT mode (0.1s hot load!)")
            self.latent_dim = 768
            return
        
        if self.mode == "turbo":
            self.pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sd-turbo", torch_dtype=self.dtype, variant="fp16").to(self.device)
        else:
            self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", torch_dtype=self.dtype).to(self.device)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

        self.pipe.safety_checker = None
        self.pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=self.dtype).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        # Экономия памяти на RTX 3060
        try:
            self.pipe.enable_attention_slicing()
        except Exception as e:
            print(f"[NeuroRender] Warning enabling attention slicing: {e}")
            
        self.latent_dim = self.pipe.text_encoder.config.hidden_size
        
        if compile_unet and self.device == 'cuda':
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=False)
            except Exception as e:
                print(f"[!] Ошибка компиляции UNet: {e}")
        self._warmup()

    def _warmup(self):
        dummy_image = Image.fromarray(np.zeros((GH, GW, 3), dtype=np.uint8))
        with torch.no_grad():
            for _ in range(2):
                self.generate(prompt="warmup", image=dummy_image, strength=1.0)

    def encode_prompt(self, prompt_text):
        if self.remote_conn is not None:
            self.remote_conn.send({'cmd': 'encode_prompt', 'text': prompt_text})
            arr = self.remote_conn.recv()
            return torch.tensor(arr, dtype=self.dtype, device=self.device)
            
        with torch.no_grad():
            return self.pipe.text_encoder(self.pipe.tokenizer(
                prompt_text, return_tensors="pt", padding="max_length", 
                max_length=self.pipe.tokenizer.model_max_length, truncation=True
            ).input_ids.to(self.device))[0]

    def generate(self, prompt=None, prompt_embeds=None, image=None, strength=0.5):
        if self.remote_conn is not None:
            from io import BytesIO
            img_buf = None
            if image is not None:
                img_buf = BytesIO()
                image.save(img_buf, format='JPEG')
                img_buf = img_buf.getvalue()
                
            embeds_np = prompt_embeds.cpu().numpy() if prompt_embeds is not None else None
            
            self.remote_conn.send({
                'cmd': 'generate',
                'prompt_embeds': embeds_np,
                'image_bytes': img_buf,
                'strength': strength
            })
            img_bytes = self.remote_conn.recv()
            return Image.open(BytesIO(img_bytes))

        kwargs = {"image": image}
        if self.mode == "turbo":
            kwargs.update({"strength": max(0.5, strength), "num_inference_steps": 2, "guidance_scale": 0.0})
        else:
            num_steps = 3
            min_safe_strength = (1.0 / num_steps) + 0.02
            kwargs.update({"strength": max(strength, min_safe_strength), "num_inference_steps": num_steps, "guidance_scale": 1.0})
            
        if prompt_embeds is not None:
            kwargs["prompt_embeds"] = prompt_embeds
        elif prompt is not None:
            kwargs["prompt"] = prompt
            
        return self.pipe(**kwargs).images[0]
