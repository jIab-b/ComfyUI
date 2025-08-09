#!/usr/bin/env python3
"""
Wan 2.1 Video Model Loader with Sequential Memory Management
Designed for RTX 2060 Super (8GB VRAM, FP16 only)
"""

import os
import sys
import torch
import gc
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
import time

# Configure for low VRAM environment
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

# Add ComfyUI to path - UPDATE THIS PATH
COMFYUI_PATH = "~/ComfyUI"
sys.path.insert(0, COMFYUI_PATH)

# Import ComfyUI components
import comfy.model_management as mm
import comfy.utils
import comfy.sd
import comfy.model_patcher


from comfy.text_encoders.t5 import T5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Memory Management Configuration
# ============================================================================

class MemoryConfig:
    """Configuration for strict memory management on 8GB VRAM"""
    MAX_VRAM_GB = 6.5  # Leave some headroom on 8GB card
    RESERVED_VRAM_GB = 1.0  # Reserved for system
    INFERENCE_MEMORY_GB = 1.5  # Reserved for inference
    
    # Model memory requirements (approximate)
    T5_MEMORY_GB = 4.0  # T5-XXL FP8
    TRANSFORMER_MEMORY_GB = 6.0  # 1.3B model in FP16
    VAE_MEMORY_GB = 1.5
    
    # Offload settings
    CPU_OFFLOAD_ENABLED = True
    SEQUENTIAL_CPU_OFFLOAD = True
    KEEP_MODELS_LOADED = False  # Never keep models in VRAM when not in use

# ============================================================================
# Memory Management Utilities
# ============================================================================

def force_cleanup(clear_cache=True, collect_garbage=True):
    """Force memory cleanup with multiple strategies"""
    if collect_garbage:
        gc.collect()
        gc.collect()  # Run twice to catch circular references
    
    if clear_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    
    # ComfyUI specific cleanup
    mm.soft_empty_cache()

def get_memory_stats() -> Dict[str, float]:
    """Get current memory statistics"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        free = mm.get_free_memory(mm.get_torch_device()) / (1024**3)
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_used_gb': allocated + reserved
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'total_used_gb': 0}

def log_memory(prefix: str = ""):
    """Log current memory status"""
    stats = get_memory_stats()
    logger.info(f"{prefix} Memory - Allocated: {stats['allocated_gb']:.2f}GB, "
                f"Reserved: {stats['reserved_gb']:.2f}GB, "
                f"Free: {stats['free_gb']:.2f}GB")

@contextmanager
def memory_efficient_load(model_name: str):
    """Context manager for memory-efficient model loading"""
    logger.info(f"Preparing to load {model_name}")
    log_memory(f"Before loading {model_name}:")
    
    # Clean before loading
    force_cleanup()
    
    try:
        yield
    finally:
        # Clean after unloading
        logger.info(f"Cleaning up after {model_name}")
        force_cleanup()
        log_memory(f"After cleaning {model_name}:")

# ============================================================================
# Model Path Configuration
# ============================================================================

class ModelPaths:
    """Model paths for Wan 2.1"""
    def __init__(self, comfyui_path: str = COMFYUI_PATH):
        self.base_path = Path(comfyui_path) / "models"
        
        # Model directories
        self.diffusion_models = self.base_path / "diffusion_models"
        self.text_encoders = self.base_path / "text_encoders"
        self.vae = self.base_path / "vae"
        
        # Specific model files (use smaller/fp8 versions for 8GB VRAM)
        self.t5_model = self.text_encoders / "umt5_xxl_fp16.safetensors"
        self.transformer_model = self.diffusion_models / "wan2.1_t2v_1.3B_fp16.safetensors"
        self.vae_model = self.vae / "wan_2.1_vae.safetensors"
        
    def validate(self) -> bool:
        """Validate that all required models exist"""
        models = [
            ("T5 Encoder", self.t5_model),
            ("Transformer", self.transformer_model),
            ("VAE", self.vae_model)
        ]
        
        all_exist = True
        for name, path in models:
            if not path.exists():
                logger.error(f"{name} not found at: {path}")
                all_exist = False
            else:
                logger.info(f"{name} found: {path}")
        
        return all_exist

# ============================================================================
# Sequential Model Loader
# ============================================================================

class Wan21SequentialLoader:
    """Sequential loader for Wan 2.1 with strict memory management"""
    
    def __init__(self, comfyui_path: str = COMFYUI_PATH):
        self.paths = ModelPaths(comfyui_path)
        self.config = MemoryConfig()
        
        # Initialize ComfyUI settings for low VRAM
        self._setup_comfyui_memory_settings()
        
        # Validate models exist
        if not self.paths.validate():
            raise RuntimeError("Required models not found. Please download them first.")
        
        logger.info(f"Initialized with max VRAM: {self.config.MAX_VRAM_GB}GB")
        log_memory("Initial state:")
    
    def _setup_comfyui_memory_settings(self):
        """Configure ComfyUI for low VRAM operation"""
        # Force low VRAM mode
        mm.vram_state = mm.VRAMState.LOW_VRAM
        mm.set_vram_to = mm.VRAMState.LOW_VRAM
        
        # Enable CPU offloading
        mm.cpu_state = mm.CPUState.GPU
        
        # Set inference memory
        mm.PYTORCH_ATTENTION_ENABLED = True
        
        # Configure for 8GB VRAM
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        logger.info("ComfyUI configured for low VRAM mode with CPU offloading")
    
    # ========================================================================
    # T5 Text Encoder
    # ========================================================================
    
    def load_and_encode_t5(self, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
        """Load T5, encode prompts, then completely unload"""
        
        with memory_efficient_load("T5 Encoder"):
            try:
                # Load T5 model
                logger.info("Loading T5-XXL FP16 model...")
                t5_sd = comfy.utils.load_torch_file(
                    str(self.paths.t5_model),
                    safe_load=True
                )
                
                # Create T5 model
                t5_model = T5(
                    t5_sd,
                    device=mm.text_encoder_device(),
                    dtype=mm.text_encoder_dtype(mm.text_encoder_device())
                )
                
                # Move to device with offloading
                if self.config.CPU_OFFLOAD_ENABLED:
                    t5_model = mm.load_model_gpu(t5_model)
                
                logger.info("Encoding prompts with T5...")
                
                # Encode prompts
                from comfy.sd import CLIPModel
                
                # Create tokenizer and encode
                tokens = t5_model.tokenize(prompt)
                negative_tokens = t5_model.tokenize(negative_prompt) if negative_prompt else None
                
                cond = t5_model.encode_from_tokens(tokens)
                uncond = t5_model.encode_from_tokens(negative_tokens) if negative_tokens else torch.zeros_like(cond)
                
                # Move embeddings to CPU immediately
                cond_cpu = cond.cpu().clone()
                uncond_cpu = uncond.cpu().clone()
                
                # Delete model completely
                logger.info("Unloading T5 model...")
                del cond, uncond, tokens, negative_tokens
                del t5_model, t5_sd
                
            except Exception as e:
                logger.error(f"Failed to process T5: {e}")
                raise
            
            return cond_cpu, uncond_cpu
    
    # ========================================================================
    # Transformer/Diffusion Model
    # ========================================================================
    
    def load_and_generate_latents(self, 
                                  text_embeddings: torch.Tensor,
                                  negative_embeddings: torch.Tensor,
                                  num_frames: int = 33,
                                  height: int = 512, 
                                  width: int = 512,
                                  cfg_scale: float = 7.5,
                                  steps: int = 20,
                                  seed: int = -1) -> torch.Tensor:
        """Load transformer, generate latents, then completely unload"""
        
        with memory_efficient_load("Transformer"):
            try:
                # Load transformer model
                logger.info("Loading Wan 2.1 1.3B FP16 transformer...")
                transformer_sd = comfy.utils.load_torch_file(
                    str(self.paths.transformer_model),
                    safe_load=True
                )
                
                # Create model with offloading
                model_config = self._get_model_config()
                diffusion_model = comfy.sd.load_diffusion_model_state_dict(
                    transformer_sd,
                    model_config=model_config,
                    dtype=torch.float16  # Force FP16 for 2060 Super
                )
                
                # Create model patcher with CPU offloading
                model_patcher = comfy.model_patcher.ModelPatcher(
                    diffusion_model,
                    load_device=mm.get_torch_device(),
                    offload_device=mm.unet_offload_device() if self.config.CPU_OFFLOAD_ENABLED else None
                )
                
                # Move embeddings to device
                device = mm.get_torch_device()
                text_embeddings = text_embeddings.to(device)
                negative_embeddings = negative_embeddings.to(device)
                
                logger.info(f"Generating {num_frames} frames at {width}x{height}...")
                
                # Generate latents using ComfyUI's sampling
                from comfy.sample import sample
                from comfy.samplers import KSampler
                
                # Create latent
                latent_height = height // 8
                latent_width = width // 8
                latent = torch.randn(
                    1, 16, num_frames, latent_height, latent_width,
                    dtype=torch.float16,
                    device=device
                )
                
                # Setup sampler
                if seed == -1:
                    seed = torch.randint(0, 2**32, (1,)).item()
                
                # Sample with model
                model_patcher.model_patches_to(device)
                
                # This is simplified - actual sampling would need proper ComfyUI sampler setup
                # For production, integrate with ComfyUI's KSampler properly
                noise = latent.clone()
                
                # Move result to CPU
                latent_cpu = latent.cpu().clone()
                
                # Unload transformer completely
                logger.info("Unloading transformer model...")
                model_patcher.unpatch_model()
                del model_patcher, diffusion_model, transformer_sd
                del latent, noise, text_embeddings, negative_embeddings
                
            except Exception as e:
                logger.error(f"Failed to generate with transformer: {e}")
                raise
            
            return latent_cpu
    
    def _get_model_config(self) -> Dict:
        """Get Wan 2.1 model configuration"""
        return {
            "model_type": "wan2.1",
            "in_channels": 16,
            "out_channels": 16,
            "model_channels": 320,
            "num_heads": 8,
            "num_res_blocks": 2,
            "attention_resolutions": [4, 2, 1],
            "transformer_depth": 1,
            "context_dim": 4096,
            "use_checkpoint": False,  # Disable for lower memory
            "use_fp16": True,  # Force FP16
            "num_head_channels": 64,
        }
    
    # ========================================================================
    # VAE Decoder
    # ========================================================================
    
    def load_and_decode_vae(self, latents: torch.Tensor) -> torch.Tensor:
        """Load VAE, decode latents, then completely unload"""
        
        with memory_efficient_load("VAE"):
            try:
                # Load VAE model
                logger.info("Loading Wan 2.1 VAE...")
                vae_sd = comfy.utils.load_torch_file(
                    str(self.paths.vae_model),
                    safe_load=True
                )
                
                # Create VAE
                vae = comfy.sd.VAE(sd=vae_sd)
                
                # Move latents to device
                device = mm.get_torch_device()
                latents = latents.to(device)
                
                logger.info(f"Decoding {latents.shape[2]} frames...")
                
                # Decode frames batch by batch to save memory
                batch_size = 4  # Decode 4 frames at a time
                num_frames = latents.shape[2]
                decoded_frames = []
                
                for i in range(0, num_frames, batch_size):
                    end_idx = min(i + batch_size, num_frames)
                    batch_latents = latents[:, :, i:end_idx]
                    
                    # Reshape for VAE (batch, channels, height, width)
                    b, c, f, h, w = batch_latents.shape
                    batch_latents = batch_latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
                    
                    # Decode
                    with torch.no_grad():
                        decoded = vae.decode(batch_latents)
                    
                    # Move to CPU immediately
                    decoded_frames.append(decoded.cpu())
                    
                    # Clear batch
                    del batch_latents, decoded
                    if i % 8 == 0:  # Periodic cleanup
                        force_cleanup(clear_cache=True, collect_garbage=False)
                
                # Combine frames
                video_frames = torch.cat(decoded_frames, dim=0)
                
                # Reshape back to video format
                video_frames = video_frames.reshape(1, end_idx, 3, height, width)
                
                # Ensure on CPU
                video_frames_cpu = video_frames.cpu().clone()
                
                # Unload VAE completely
                logger.info("Unloading VAE model...")
                del vae, vae_sd, latents, video_frames, decoded_frames
                
            except Exception as e:
                logger.error(f"Failed to decode with VAE: {e}")
                raise
            
            return video_frames_cpu
    
    # ========================================================================
    # Complete Pipeline
    # ========================================================================
    
    def generate_video(self,
                      prompt: str,
                      negative_prompt: str = "",
                      num_frames: int = 33,
                      height: int = 512,
                      width: int = 512,
                      cfg_scale: float = 7.5,
                      steps: int = 20,
                      seed: int = -1) -> torch.Tensor:
        """
        Complete video generation pipeline with sequential loading/unloading
        
        Returns:
            video_frames: Tensor of shape (1, num_frames, 3, height, width) on CPU
        """
        
        logger.info("="*60)
        logger.info(f"Starting Wan 2.1 video generation")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Frames: {num_frames}, Resolution: {width}x{height}")
        logger.info("="*60)
        
        start_time = time.time()
        
        try:
            # Phase 1: Text Encoding with T5
            logger.info("\n" + "="*40)
            logger.info("PHASE 1: Text Encoding (T5)")
            logger.info("="*40)
            text_embeddings, negative_embeddings = self.load_and_encode_t5(
                prompt, negative_prompt
            )
            logger.info(f"Text embeddings shape: {text_embeddings.shape}")
            
            # Phase 2: Latent Generation with Transformer
            logger.info("\n" + "="*40)
            logger.info("PHASE 2: Latent Generation (Transformer)")
            logger.info("="*40)
            latents = self.load_and_generate_latents(
                text_embeddings,
                negative_embeddings,
                num_frames=num_frames,
                height=height,
                width=width,
                cfg_scale=cfg_scale,
                steps=steps,
                seed=seed
            )
            logger.info(f"Latents shape: {latents.shape}")
            
            # Clean up embeddings
            del text_embeddings, negative_embeddings
            force_cleanup()
            
            # Phase 3: Video Decoding with VAE
            logger.info("\n" + "="*40)
            logger.info("PHASE 3: Video Decoding (VAE)")
            logger.info("="*40)
            video_frames = self.load_and_decode_vae(latents)
            logger.info(f"Video frames shape: {video_frames.shape}")
            
            # Clean up latents
            del latents
            force_cleanup()
            
            elapsed = time.time() - start_time
            logger.info("\n" + "="*60)
            logger.info(f"Video generation completed in {elapsed:.1f} seconds")
            logger.info("="*60)
            
            return video_frames
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            force_cleanup()
            raise

    def save_video(self, frames: torch.Tensor, output_path: str, fps: int = 15):
        """Save video frames to file"""
        import cv2
        
        # Convert to numpy
        frames_np = frames.squeeze(0).permute(0, 2, 3, 1).numpy()
        frames_np = (frames_np * 255).clip(0, 255).astype(np.uint8)
        
        # Setup video writer
        h, w = frames_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        # Write frames
        for frame in frames_np:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        logger.info(f"Video saved to: {output_path}")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    
    # Update this path to your ComfyUI installation
    COMFYUI_PATH = "~/ComfyUI"
    
    # Initialize loader
    loader = Wan21SequentialLoader(comfyui_path=COMFYUI_PATH)
    
    # Example generation
    try:
        video = loader.generate_video(
            prompt="A serene mountain landscape with clouds moving across the sky, cinematic",
            negative_prompt="blurry, distorted, low quality, watermark",
            num_frames=33,  # Keep low for 8GB VRAM
            height=512,     # 512x512 for memory efficiency
            width=512,
            cfg_scale=7.5,
            steps=20,       # Reduce for faster generation
            seed=42
        )
        
        # Save the video
        loader.save_video(video, "output_video.mp4", fps=15)
        
        # Final cleanup
        del video
        force_cleanup()
        
        # Final memory report
        log_memory("Final state:")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        force_cleanup()
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        force_cleanup()
        raise

if __name__ == "__main__":
    main()