"""Independent mechanical gates for repeated native ASR; not a quality scorer."""

import math
import numpy as np


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:
        raise ValueError('Expected finite nonnegative measured value')
    return value


def measured_gate(seconds,elapsed,calls,samples,failures):
    if type(seconds) is not int or seconds not in (30,1800):raise ValueError('Unregistered ASRsoak horizon')
    finite(elapsed)
    audio=0.;rtfs=[]
    for call in calls:
        duration=finite(call['audio_seconds'])
        if duration==0:raise ValueError('Audio duration must be positive')
        wall=finite(call['seconds']);ratio=wall/duration
        if abs(ratio-finite(call['RTF']))>1e-10:raise ValueError('Stored ASRRTF mismatch')
        if call['native_CUDA_FP32_verified'] is not True:raise ValueError('Missing CUDAFP32 evidence')
        audio+=duration;rtfs.append(ratio)
    free=[finite(row['free_MiB']) for row in samples]
    active=sum(row.get('ASR_active') is True for row in samples)
    p95=float(np.percentile(rtfs,95)) if rtfs else None
    gate=bool(not failures and elapsed>=seconds and len(calls)>=(1 if seconds==30 else 80)
        and (seconds==30 or audio>=1000 and p95<=1)
        and free and min(free)>=512 and active>0)
    return {'gate':gate,'completed_calls':len(calls),'processed_audio_seconds':audio,
            'RTF_p95':p95,'minimum_sampled_global_free_MiB':min(free) if free else None,
            'active_ASR_resource_samples':active,'new_independent_quality_samples':0}
