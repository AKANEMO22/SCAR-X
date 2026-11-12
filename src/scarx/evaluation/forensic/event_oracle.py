"""Independent chronological key/value oracle for audit data only.

No agent labels, latent metadata, neural weights or source scorer are consumed.
Known seeded noise may be removed by an audit caller; that is not model input.
"""
import numpy as np


class EventOracleError(ValueError):
    pass


def decode_events(inputs,expected_writes=2,threshold=.5):
    x=np.asarray(inputs)
    if x.ndim!=3 or x.shape[2]!=8 or len(x)==0 or not np.isfinite(x).all():
        raise EventOracleError('Finite nonempty [N,T,8] event tensor required')
    if type(expected_writes) is not int or expected_writes not in (2,3):
        raise EventOracleError('Registered two-write or three-write task required')
    a=[];b=[];queries=[];ticks=[];labels=[];first=[];write_ticks=[];writes=[]
    for index,row in enumerate(x):
        locations=np.argwhere(row[:,:2]>threshold)
        qloc=np.argwhere(row[:,4:6]>threshold)
        if len(locations)!=expected_writes or len(qloc)!=1:
            raise EventOracleError(f'row{index}: marker count ambiguous: writes={len(locations)}, query={len(qloc)}')
        qtick,q=map(int,qloc[0]);memory={};seen=[]
        for tick,key in locations:
            values=np.flatnonzero(row[tick,2:4]>threshold)
            if len(values)!=1 or tick>=qtick:
                raise EventOracleError(f'row{index}: ambiguous write or query preceding writes')
            memory[int(key)]=int(values[0]);seen.append((int(tick),int(key),int(values[0])))
        if set(memory)!={0,1}:raise EventOracleError(f'row{index}: both keys required')
        if expected_writes==2 and len({key for _,key,_ in seen})!=2:
            raise EventOracleError(f'row{index}: two-write task must have distinct keys')
        a.append(memory[0]);b.append(memory[1]);queries.append(q);ticks.append(qtick)
        labels.append(memory[q]);first.append(seen[0][2]);write_ticks.append([t for t,_,_ in seen]);writes.append(seen)
    return {k:np.asarray(v,np.int32) for k,v in [('vA',a),('vB',b),('q',queries),('query_tick',ticks),
        ('label',labels),('first_written_value',first),('write_ticks',write_ticks),('writes',writes)]}


def pulse_tensor(decoded,shape,pulse=1.5):
    result=np.zeros(shape,np.float32)
    for index,writes in enumerate(decoded['writes']):
        for tick,key,value in writes:
            result[index,tick,key]+=pulse;result[index,tick,2+value]+=pulse
        result[index,decoded['query_tick'][index],4+decoded['q'][index]]+=pulse
    return result


def move_known_query_tensor(inputs,decoded,delta=1):
    """Independent linear movement of a known query-bearing input tick.

    Uses decoded base event times, never labels. It deliberately does not infer
    query markers again from transformed noisy observations.
    """
    x=np.asarray(inputs);qt=np.unique(decoded['query_tick'])
    if len(qt)!=1 or delta<1 or qt[0]+delta>=x.shape[1]:raise EventOracleError('One in-range query-tick move required')
    out=np.zeros_like(x)
    for t in range(x.shape[1]):out[:,t+delta if t==qt[0] else t]+=x[:,t]
    return out


def pair_contract(raw_inputs,decoded,pulse=1.5,tolerance=2e-7):
    x=np.asarray(raw_inputs,np.float64)
    if len(x)%2:raise EventOracleError('Complete adjacent sibling pairs required')
    if not np.all(decoded['q'][::2]==0) or not np.all(decoded['q'][1::2]==1):
        raise EventOracleError('Expected query-A/query-B sibling ordering')
    for name in ('vA','vB','query_tick','write_ticks'):
        if not np.array_equal(decoded[name][::2],decoded[name][1::2]):
            raise EventOracleError('Sibling facts/timing differ: '+name)
    background=x.copy()
    background[np.arange(len(x)),decoded['query_tick'],4+decoded['q']]-=pulse
    delta=np.max(np.abs(background[::2]-background[1::2]),axis=(1,2))
    return {'pair_count':len(x)//2,'facts_and_query_pairing_valid':True,
            'same_nonquery_noise_pairs':int(np.sum(delta<=tolerance)),
            'different_nonquery_noise_pairs':int(np.sum(delta>tolerance)),
            'max_nonquery_difference':float(delta.max(initial=0)),
            'float32_pulse_subtraction_tolerance':tolerance}
