"""Observable per-request input/workset, not a neural-memory capacity score."""
import hashlib
import json
import uuid
import numpy as np


def plan_workload(model, prompt='', pixels=None, budget=None, request_id=None):
    if not isinstance(prompt,str) or len(prompt)>2048:raise ValueError('Input≤2048characters')
    if request_id is not None and (not isinstance(request_id,str) or len(request_id)>80 or not request_id.isascii()):raise ValueError('Invalid request ID')
    # Input volume is a bounded workload heuristic, not a semantic difficulty
    # classifier. Keep the48protected base and never allocate because of error.
    words=len(prompt.split())
    connected=np.flatnonzero(np.any(model.core.w_out!=0,axis=0))
    protected=max(48,int(connected[-1])+1 if len(connected) else 48)
    requested=64 if pixels is not None else (48 if words<=12 else 64 if words<=40 else 96 if words<=100 else 128)
    requested=max(requested,protected)
    if budget is not None:
        if type(budget) is not int or not 48<=budget<=model.max_neurons:raise ValueError('Neuronbudget48–128')
        if budget<protected:raise ValueError(f'Cần ít nhất{protected}neuron để không tắt các readout đã học; chưa có kiểm chứng pruning.')
        requested=budget
    vector=model.context(prompt) if prompt.strip() and pixels is None else None
    return {'request_id':request_id or uuid.uuid4().hex,'phase':'processing','modality':'digit_image' if pixels is not None else 'text',
        'input_characters':len(prompt),'input_utf8_bytes':len(prompt.encode('utf-8')),'input_limit_characters':2048,
        'image_count':int(pixels is not None),'neural_image_pixels':64 if pixels is not None else 0,
        'prompt_features_nonzero':int(np.count_nonzero(vector)) if vector is not None else 0,
        'neural_input_channels':model.core.config.inputs,'neural_input_buffer_bytes':model.core.config.inputs*4,
        'requested_neurons':requested,'active_neurons':requested,'allocated_neurons':model.core.config.hidden,
        'max_neurons':model.max_neurons,'protected_neurons':protected,'replayed_messages':0,'retrieved_episodes':0,'tool_schemas':0,
        'allocation_reason':'explicit request budget' if budget is not None else 'observable input volume/modality heuristic, not validated task difficulty',
        'memory_capacity_measured':False,'capacity_improves_quality_proven':False}


def finish_workload(model, load, ram_bytes):
    c=model.core
    return {**load,'phase':'complete','active_neurons':model.active_count,'allocated_neurons':c.config.hidden,
        'output_connected_neurons':int(np.count_nonzero(np.any(c.w_out!=0,axis=0))),
        'neural_arrays_bytes':c.memory_bytes(),'process_rss_bytes':ram_bytes,
        'dormant_neurons':c.config.hidden-model.active_count,
        'note':'New neuron readout columns start zero and need learning; allocation alone adds no skill. All weights remain resident.'}
