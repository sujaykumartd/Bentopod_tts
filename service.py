from __future__ import annotations

import os
import typing as t
from pathlib import Path
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import asyncio
import time

import bentoml


__all__ = ["Pod_TTS"]


device = "cuda:0" if torch.cuda.is_available() else "cpu"


MODEL_ID = "parler-tts/parler-tts-mini-v1"

@bentoml.service(
    resources={
        "gpu": 1,
        "memory": "8Gi",
    },
    traffic={"timeout": 300},
)
class Pod_TTS:
    def __init__(self) -> None:
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.description = "Jon's voice is monotone and confident. He is a professional male English speaker with a clear, consistent voice delivers content at a moderate pace. The recording has studio-quality audio with no background noise and maintains the same tone and style throughout."
        self.pause_samples = int(0.5 * self.model.config.sampling_rate)  # 0.5 second pause
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def _split_into_sentences(self, text):
        # Split text on sentence boundaries (., !, ?)
        import re
        sentences = re.split('(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def _process_chunk(self, chunk, input_ids):
        # Process each chunk with the same voice characteristics
        chunk_input_ids = self.tokenizer(chunk, return_tensors="pt").input_ids.to(self.device)
        generation = self.model.generate(input_ids=input_ids, prompt_input_ids=chunk_input_ids)
        return generation.cpu().numpy().squeeze()

    @bentoml.api()
    async def generate_audio(self, 
            context: bentoml.Context,
            prompt: str = "Hey, how are you doing today?",
            output_file: str = "parler_tts_out.wav"):
        # Split prompt into sentences for more natural chunks
        st = time.time()
        sentences = self._split_into_sentences(prompt)
        
        # Get input IDs for description - same voice profile used for all chunks
        input_ids = self.tokenizer(self.description, return_tensors="pt",padding=True,return_attention_mask=True).input_ids.to(self.device)
        
        # Create pause samples
        pause = np.zeros(self.pause_samples)
        
        # Process all chunks in parallel while maintaining voice consistency
        tasks = [self._process_chunk(sentence, input_ids) for sentence in sentences]
        audio_chunks = await asyncio.gather(*tasks)
        
        # Interleave audio chunks with pauses
        full_audio = []
        for chunk in audio_chunks:
            full_audio.append(chunk)
            full_audio.append(pause)
            
        # Concatenate all audio chunks and save
        audio_arr = np.concatenate(full_audio)
        sf.write(output_file, audio_arr, self.model.config.sampling_rate)
        end = time.time()
        time_taken = end - st
        print(f"Time taken: {time_taken} seconds")
        return {"output_file": output_file, "time_taken": time_taken}
