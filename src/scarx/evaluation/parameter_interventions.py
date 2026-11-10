"""Controlled inference-parameter swaps; diagnostics only, not learning."""
import numpy as np
from ..neural.unified import UnifiedCore

PARTS=('w_in','w_rec','w_out','bias')


def transplant_arrays(model,snapshot,parts):
    """Validate ALLrequestedarrays before any mutation; never copy othergroups."""
    parts=tuple(parts)
    if len(parts)!=len(set(parts)) or any(n not in PARTS for n in parts):raise ValueError('Unknown/duplicate parametergroup')
    for name in parts:
        a=snapshot.get(name);template=getattr(model.core,name)
        if not isinstance(a,np.ndarray) or a.shape!=template.shape or a.dtype!=template.dtype or not np.isfinite(a).all():raise ValueError('Transplant shape/type/finite mismatch: '+name)
    for name in parts:getattr(model.core,name)[:]=snapshot[name]
    return model


def from_payloads(before_payload,after_payload,parts):
    """Exactbeforestate with selectedafterweights; no source model mutated."""
    before=UnifiedCore.loads(before_payload);after=UnifiedCore.loads(after_payload)
    if before.vocabulary!=after.vocabulary or before.encoding!=after.encoding or before.core.config!=after.core.config or not np.array_equal(before.core.indices,after.core.indices):raise ValueError('Transplant requires samevocab/encoder/config/topology')
    return transplant_arrays(before,{n:getattr(after.core,n) for n in PARTS},parts)
