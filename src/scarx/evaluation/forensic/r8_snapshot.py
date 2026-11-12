"""Bounded R8 research-model snapshots; no candidate imports or learning changes.

Explicit caller-supplied model template only, never import classes from payload.
App/SQLite/session migration and arbitrary foreign optimizers remain unsupported.
"""
import copy,hashlib,inspect,io,json,math,os,tempfile,zipfile
from dataclasses import asdict
from pathlib import Path
import numpy as np

FORMAT='scarx-research-r8-snapshot-v1'
SOURCE='fe472b3b1257cdb264805bf79fd53e5647e3568d'
MATCHED_SOURCE='607ffca7c9222c5e3fd9741c3ebdf5939e10bedf'
MAX_BYTES=2_000_000
M_ARRAYS={'e_v','e_a','e_f','syn_trace','syn_energy','f_mean','f_var','w_in_0','w_rec_0','gamma_e_vec'}
B_ARRAYS={'u1_in','u2_in','u1_rec','u2_rec','u1_out','u2_out','bias','w_ctx','e_v','e_a','e_f','syn_trace','syn_energy','f_mean','f_var','gamma_e_vec'}
M_SETTINGS={'condition','sensory_threshold','calibrated','episodes_learned','hidden_updates','total_ticks'}
B_SETTINGS={'use_cascade','use_context_gating','kappa1','kappa2','sensory_threshold','n_context','calibrated','episodes_learned','hidden_updates'}
M_KINDS=('MultiTimescaleSensoryGatedEProp','Peer607MTSG','MatchedLocalCredit','StoppedResetCredit')
NEW_KINDS=('Peer607MTSG','Peer607BF','MatchedLocalCredit','StoppedResetCredit')
ACR_ARRAYS=M_ARRAYS|{'e_f_own','e_v_stop','e_a_stop','e_f_stop','alpha','router_q','router_bias','last_u','last_theta','last_a','last_signal','last_delta'}
ACR_SETTINGS=M_SETTINGS|{'trace_kind','gate_mode','profile'}


def mechanism_hash(model):
    return digest(Path(inspect.getfile(type(model))).read_bytes().replace(b'\r\n',b'\n'))


def encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')
def digest(raw):return hashlib.sha256(raw).hexdigest()


def validate_controller(value,depth=0):
    if depth>16:raise ValueError('Controller nesting bound')
    if value is None or type(value) in (str,bool,int):return
    if type(value) is float and math.isfinite(value):return
    if type(value) is list:
        for child in value:validate_controller(child,depth+1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for child in value.values():validate_controller(child,depth+1)
        return
    raise ValueError('Controller must use lossless JSON types and string keys')


def metadata(raw):
    if len(raw)>65536:raise ValueError('Metadata byte bound')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key]=value
        return result
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if not isinstance(value,dict):raise ValueError('Metadata object required')
    return value


def layout(model):
    name=type(model).__name__
    if name in ('MultiTimescaleSensoryGatedEProp','BennaFusiMetaplasticCascade'):
        parts=Path(inspect.getfile(type(model))).parts
        if 'review-gemini-r8-607ffca' in parts:name='Peer607MTSG' if name=='MultiTimescaleSensoryGatedEProp' else 'Peer607BF'
        elif 'review-gemini-r8-fe472b3' not in parts:raise ValueError('Explicit version-pinned peer template required')
    if name in ('MultiTimescaleSensoryGatedEProp','Peer607MTSG'):arrays,settings,rng=M_ARRAYS,M_SETTINGS,{'signal_rng'}
    elif name=='MatchedLocalCredit':arrays,settings,rng=M_ARRAYS,M_SETTINGS|{'trace_kind','gate_mode'},{'signal_rng'}
    elif name=='StoppedResetCredit':arrays,settings,rng=M_ARRAYS,M_SETTINGS|{'trace_kind','gate_mode','reset_mode'},{'signal_rng'}
    elif name=='AdaptiveCreditRouting':arrays,settings,rng=ACR_ARRAYS,ACR_SETTINGS,{'signal_rng','router_rng'}
    elif name=='BennaFusiMetaplasticCascade':arrays,settings,rng=B_ARRAYS,B_SETTINGS,set()
    elif name=='Peer607BF':arrays,settings,rng=B_ARRAYS|{'e_x'},B_SETTINGS,set()
    else:raise ValueError('Only explicit R8 research models supported')
    if set(vars(model))!=arrays|settings|rng|{'config','core'}:raise ValueError('Unknown learner state cannot be dropped')
    c=model.config
    if (c.hidden,c.inputs,c.outputs,c.fan_in,c.plastic_rows,c.neuron)!=(48,8,2,8,12,'alif'):
        raise ValueError('Snapshot scope H48 D8 O2 K8 R12 ALIF only')
    if model.core.config!=c:raise ValueError('Core/model config mismatch')
    expected=set(model.core.weight_names)|set(model.core.state_names)|{'config','indices','updates','reward_mean'}
    if set(vars(model.core))!=expected:raise ValueError('Unknown core state cannot be dropped')
    all_arrays={'core__'+key:val for key,val in model.core.arrays().items()}
    all_arrays.update({'model__'+key:getattr(model,key) for key in arrays})
    return name,settings,all_arrays


def validate_settings(kind,settings,core_settings):
    expected=ACR_SETTINGS if kind=='AdaptiveCreditRouting' else M_SETTINGS|{'trace_kind','gate_mode','reset_mode'} if kind=='StoppedResetCredit' else M_SETTINGS|{'trace_kind','gate_mode'} if kind=='MatchedLocalCredit' else M_SETTINGS if kind in ('MultiTimescaleSensoryGatedEProp','Peer607MTSG') else B_SETTINGS
    if set(settings)!=expected:raise ValueError('Settings schema')
    for key,value in settings.items():
        if key in ('calibrated','use_cascade','use_context_gating'):
            if type(value) is not bool:raise ValueError('Boolean setting')
        elif key in ('episodes_learned','hidden_updates','total_ticks','n_context'):
            if type(value) is not int or not 0<=value<2**63:raise ValueError('Counter bounds')
        elif key=='condition':
            if value not in ('frozen_hidden','zero_signal','shuffled_signal','readout_only','r35_baseline','ew_he_legacy','learned_mtsg'):
                raise ValueError('Unknown research condition')
        elif key=='profile':
            from ..acr import PROFILES
            if value not in PROFILES:raise ValueError('Unknown ACR profile')
        elif key=='trace_kind':
            if value not in ('fast','slow','multi'):raise ValueError('Unknown trace policy')
        elif key=='gate_mode':
            if value not in ('none','activity'):raise ValueError('Unknown gate policy')
        elif key=='reset_mode':
            if value!='stopped':raise ValueError('Unknown reset-derivative policy')
        elif type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=100:
            raise ValueError('Finite bounded setting required')
    expected_context=8 if kind=='Peer607BF' else 2
    if kind in ('BennaFusiMetaplasticCascade','Peer607BF') and (settings['n_context']!=expected_context or settings['kappa1']>1 or settings['kappa2']>1):
        raise ValueError('Cascade setting bounds')
    if settings['hidden_updates']>settings['episodes_learned']:raise ValueError('Hidden/episode counters inconsistent')
    if set(core_settings)!={'updates','reward_mean'} or type(core_settings['updates']) is not int or not 0<=core_settings['updates']<2**63:
        raise ValueError('Core learning counters')
    if type(core_settings['reward_mean']) not in (int,float) or not math.isfinite(core_settings['reward_mean']):
        raise ValueError('Core reward state')


def dumps(model,controller=None):
    kind,names,arrays=layout(model);settings={key:getattr(model,key) for key in names}
    core_settings={'updates':model.core.updates,'reward_mean':model.core.reward_mean}
    validate_settings(kind,settings,core_settings)
    validate_controller(controller)
    if controller is not None and (not isinstance(controller,dict) or len(encoded(controller))>16384):raise ValueError('Bounded explicit controller object required')
    members={}
    for name,array in arrays.items():
        if not isinstance(array,np.ndarray) or array.dtype not in (np.dtype('float32'),np.dtype('int32')) or not np.isfinite(array).all():
            raise ValueError('Invalid snapshot array')
        stream=io.BytesIO();np.save(stream,array,allow_pickle=False);members[name+'.npy']=stream.getvalue()
    acr=kind=='AdaptiveCreditRouting'
    rng=model.signal_rng.bit_generator.state if kind in M_KINDS or acr else None
    source=MATCHED_SOURCE if kind in NEW_KINDS or acr else SOURCE
    meta={'format':FORMAT,'source_sha':source,'kind':kind,'config':asdict(model.config),'settings':settings,
          'core_settings':core_settings,'rng':rng,'controller':controller,'sha256':{key:digest(raw) for key,raw in members.items()}}
    if acr:
        meta['format']='scarx-acr-snapshot-v1';meta['router_rng']=model.router_rng.bit_generator.state
        meta['mechanism_sha256']=mechanism_hash(model)
    meta['metadata_sha256']=digest(encoded(meta));members['metadata.json']=encoded(meta)
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(members.items()):archive.writestr(zipfile.ZipInfo(name),raw)
    payload=out.getvalue()
    if len(payload)>MAX_BYTES:raise ValueError('Snapshot size bound')
    return payload


def loads(template,payload):
    """Validate everything before cloning, then return model + controller copies."""
    kind,_,expected=layout(template)
    if not isinstance(payload,bytes) or len(payload)>MAX_BYTES:raise ValueError('Snapshot size bound')
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names=archive.namelist();fields={key+'.npy' for key in expected}|{'metadata.json'}
            if len(names)!=len(fields) or set(names)!=fields or sum(i.file_size for i in archive.infolist())>MAX_BYTES:
                raise ValueError('Snapshot member/bounds mismatch')
            if archive.getinfo('metadata.json').file_size>65536:raise ValueError('Metadata byte bound')
            meta=metadata(archive.read('metadata.json'));check=meta.pop('metadata_sha256')
            acr=kind=='AdaptiveCreditRouting'
            fields={'format','source_sha','kind','config','settings','core_settings','rng','controller','sha256'}
            if acr:fields|={'router_rng','mechanism_sha256'}
            if set(meta)!=fields or digest(encoded(meta))!=check:
                raise ValueError('Metadata schema/checksum')
            source=MATCHED_SOURCE if kind in NEW_KINDS or acr else SOURCE
            fmt='scarx-acr-snapshot-v1' if acr else FORMAT
            if meta['format']!=fmt or meta['source_sha']!=source or meta['kind']!=kind or meta['config']!=asdict(template.config):
                raise ValueError('Explicit template/source/config mismatch')
            if acr and (meta['mechanism_sha256']!=mechanism_hash(template) or meta['settings']['profile']!=template.profile or meta['settings']['condition']!=template.condition):
                raise ValueError('ACR mechanism/profile mismatch')
            validate_settings(kind,meta['settings'],meta['core_settings'])
            validate_controller(meta['controller'])
            if meta['controller'] is not None and (not isinstance(meta['controller'],dict) or len(encoded(meta['controller']))>16384):
                raise ValueError('Bounded controller state required')
            if set(meta['sha256'])!={key+'.npy' for key in expected}:raise ValueError('Array checksum fields')
            # Every declared NPY shape/type/body is checked before any np.load allocation.
            for name,prior in expected.items():
                filename=name+'.npy'
                with archive.open(filename) as stream:
                    version=np.lib.format.read_magic(stream)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=8192)
                    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=8192)
                    else:raise ValueError('Unsupported NPY header')
                    end=stream.tell()
                if shape!=prior.shape or dtype!=prior.dtype or fortran or dtype.hasobject:
                    raise ValueError('Declared array shape/dtype rejected before allocation')
                if end+math.prod(shape)*dtype.itemsize!=archive.getinfo(filename).file_size:raise ValueError('NPY body/header mismatch')
            arrays={}
            for name in expected:
                raw=archive.read(name+'.npy')
                if digest(raw)!=meta['sha256'][name+'.npy']:raise ValueError('Array checksum mismatch')
                array=np.load(io.BytesIO(raw),allow_pickle=False)
                if not np.isfinite(array).all():raise ValueError('Nonfinite array rejected')
                arrays[name]=array
        if np.any(arrays['core__indices']<0) or np.any(arrays['core__indices']>=48) or np.any(arrays['core__rows']<0) or np.any(arrays['core__rows']>=48):
            raise ValueError('Recurrent/plastic indices out of range')
        if np.any((arrays['model__gamma_e_vec']<0)|(arrays['model__gamma_e_vec']>=1)):raise ValueError('Eligibility decay range')
        if np.any(arrays['model__f_var']<0) or np.any(arrays['model__syn_energy']<0):raise ValueError('Variance/energy state')
        if not np.isin(arrays['core__spikes'],[0,1]).all():raise ValueError('ALIF binary spike state required')
        result=copy.deepcopy(template)
        if acr:
            if np.any((arrays['model__alpha']<0)|(arrays['model__alpha']>1)) or np.any(np.abs(arrays['model__router_q'])>1):
                raise ValueError('ACR router state range')
            if not isinstance(meta['router_rng'],dict) or meta['router_rng'].get('bit_generator')!='PCG64':raise ValueError('Router PCG64 state required')
        if kind in M_KINDS or acr:
            if not isinstance(meta['rng'],dict) or meta['rng'].get('bit_generator')!='PCG64':raise ValueError('PCG64 RNG state required')
            result.signal_rng.bit_generator.state=meta['rng']
            if acr:result.router_rng.bit_generator.state=meta['router_rng']
        elif meta['rng'] is not None:raise ValueError('Unexpected RNG state')
        for name,array in arrays.items():
            scope,key=name.split('__',1);getattr(result.core if scope=='core' else result,key)[:]=array
        for key,value in meta['settings'].items():setattr(result,key,value)
        for key,value in meta['core_settings'].items():setattr(result.core,key,value)
        return result,copy.deepcopy(meta['controller'])
    except (KeyError,TypeError,OverflowError,zipfile.BadZipFile,UnicodeError) as exc:raise ValueError('Malformed R8 research snapshot') from exc


def save(model,path,controller=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);payload=dumps(model,controller)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def load(template,path):
    path=Path(path)
    if path.stat().st_size>MAX_BYTES:raise ValueError('Snapshot file bound')
    return loads(template,path.read_bytes())
