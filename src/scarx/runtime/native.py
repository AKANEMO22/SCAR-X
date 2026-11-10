"""C++ kernels with Python checkpoint/orchestration; NOT a standalone runtime."""
import ctypes as ct
from dataclasses import asdict
from pathlib import Path
import os
import numpy as np
from ..neural.core import SpikingCore

FLOAT = ct.POINTER(ct.c_float)
INT = ct.POINTER(ct.c_int32)
NAMES = ("w_in","w_rec","w_out","bias","stable_in","stable_rec","stable_out","stable_bias",
         "voltage","adaptation","spikes","pre","post","features","elig_in","elig_rec")


class NativeCore(SpikingCore):
    def __init__(self, config, library):
        super().__init__(config)
        self.dll = ct.CDLL(str(Path(library).resolve()))
        self.dll.scar_step.argtypes = [ct.c_int]*5+[FLOAT,ct.POINTER(FLOAT),INT,INT,FLOAT,ct.c_int,ct.c_int,FLOAT]
        self.dll.scar_step.restype = None
        self.dll.scar_learn.argtypes = [ct.c_int]*5+[FLOAT,ct.POINTER(FLOAT),INT,ct.c_int,ct.c_int,ct.POINTER(ct.c_double),ct.c_int,ct.c_double]
        self.dll.scar_learn.restype = ct.c_float
        self.params = np.array([getattr(config,n) for n in ("leak","threshold","adaptation_decay","adaptation_strength","trace_decay","feature_decay","readout_lr","hidden_lr","anchor_strength","update_clip","weight_clip")],dtype=np.float32)

    def _args(self):
        c = self.config
        pointers = (FLOAT*len(NAMES))(*(getattr(self,n).ctypes.data_as(FLOAT) for n in NAMES))
        return [c.inputs,c.hidden,c.outputs,c.fan_in,c.plastic_rows,self.params.ctypes.data_as(FLOAT),pointers]

    def step(self,x,track=True):
        x = np.ascontiguousarray(x,dtype=np.float32)
        if x.shape != (self.config.inputs,) or not np.all(np.isfinite(x)):
            raise ValueError("Invalid native input")
        out = np.empty(self.config.outputs,np.float32)
        self.dll.scar_step(*self._args(),self.indices.ctypes.data_as(INT),self.rows.ctypes.data_as(INT),x.ctypes.data_as(FLOAT),int(track),int(self.config.neuron=="rate"),out.ctypes.data_as(FLOAT))
        return out

    def learn(self,target,mode="full",reward=None):
        modes = {"frozen":0,"readout":1,"stdp":2,"full":3}
        if mode not in modes or not 0 <= target < self.config.outputs or (reward is not None and not np.isfinite(reward)):
            raise ValueError("Invalid native learning arguments")
        baseline = ct.c_double(self.reward_mean)
        result = self.dll.scar_learn(*self._args(),self.rows.ctypes.data_as(INT),target,modes[mode],ct.byref(baseline),int(reward is not None),float(reward or 0))
        self.reward_mean = baseline.value
        if mode != "frozen":
            self.updates += 1
        return float(result)
