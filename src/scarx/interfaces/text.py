"""Byte-level plumbing, NOT a trained language model.

UTF-8 byte tokenization is fixed and lossless; generated invalid UTF-8 is
replaced for display and reported. Long-language training is gate-controlled.
"""
import numpy as np
from ..neural.core import Config, SpikingCore
from ..memory.store import EpisodeStore


class TextAgent:
    def __init__(self, model, store):
        if model.config.inputs != 256 or model.config.outputs != 256:
            raise ValueError("Text interface requires 256 byte inputs/outputs")
        self.model,self.store = model,store

    def generate(self, prompt, max_tokens=64):
        raw = prompt.encode("utf-8")
        if len(raw)>8192 or not 1 <= max_tokens <= 1024:
            raise ValueError("Prompt <=8192 bytes; generation 1..1024 bytes")
        self.model.reset_state()
        x = np.zeros(256,np.float32)
        p = self.model.probabilities()
        for token in raw or b" ":
            x.fill(0); x[token]=1
            p = self.model.step(x,track=False)
        out = bytearray()
        for _ in range(max_tokens):
            token = int(np.argmax(p))
            out.append(token)
            x.fill(0); x[token]=1
            p = self.model.step(x,track=False)
        try:
            text, valid = bytes(out).decode("utf-8"),True
        except UnicodeDecodeError:
            text, valid = bytes(out).decode("utf-8",errors="replace"),False
        return {"text":text,"valid_utf8":valid,"bytes_hex":out.hex(),"tokens":len(out),
                "status":"untrained_research_model","retrieval_used":False}

    def observe(self,*args,**kwargs):
        return self.store.observe(*args,**kwargs)

    def save(self,path):
        self.model.save(path)

    def load(self,path):
        self.model = SpikingCore.load(path)
        if self.model.config.inputs != 256 or self.model.config.outputs != 256:
            raise ValueError("Not a byte model")

