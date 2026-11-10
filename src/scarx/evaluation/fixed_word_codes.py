"""Fixed train-prompt-only numeric word codes; no labels or neural weights."""
from collections import Counter
from functools import lru_cache
import hashlib,json,math
import numpy as np
from ..neural.language import tokenize

KINDS=('known-sparse128-v1','known-digest128-v1')


def words(prompt):return [t for t in tokenize(prompt) if any(ch.isalnum() for ch in t)]


def fit(prompts,kind):
    prompts=list(prompts)
    if not prompts or any(not isinstance(p,str) or not 1<=len(p.strip())<=2048 for p in prompts):raise ValueError('Expected training prompts')
    df=Counter(t for p in prompts for t in set(words(p)))
    metadata={'kind':kind,'train_documents':len(prompts),
        'training_prompt_sha256':hashlib.sha256(json.dumps(prompts,ensure_ascii=False).encode()).hexdigest(),
        'idf':{t:float(np.log((1+len(prompts))/(1+n))+1) for t,n in sorted(df.items())}}
    return validate(metadata)


def validate(metadata):
    if not isinstance(metadata,dict) or set(metadata)!={'kind','train_documents','training_prompt_sha256','idf'} or metadata['kind'] not in KINDS:raise ValueError('Unknown fixed input code')
    if type(metadata['train_documents']) is not int or not 1<=metadata['train_documents']<=1_000_000:raise ValueError('Training document count')
    digest=metadata['training_prompt_sha256']
    if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Training prompt checksum')
    idf=metadata['idf']
    if not isinstance(idf,dict) or not 1<=len(idf)<=4096:raise ValueError('Bounded word inventory required')
    for t,v in idf.items():
        if not isinstance(t,str) or not 1<=len(t)<=64 or words(t)!=[t] or type(v) not in (int,float) or not math.isfinite(v) or not 0<v<=20:raise ValueError('Invalid word/IDF')
    encoded=json.dumps(metadata,ensure_ascii=False,allow_nan=False)
    if len(encoded.encode())>65536:raise ValueError('Fixed code metadata exceeds65KB')
    return json.loads(encoded) # own metadata, no shared mutable caller object


@lru_cache(maxsize=4096)
def vector(word,kind):
    if kind=='known-sparse128-v1':
        code=int.from_bytes(hashlib.blake2b(word.encode(),digest_size=8).digest(),'little')
        out=np.zeros(128,np.float32);out[code%128]=1 if (code>>8)&1 else -1
    elif kind=='known-digest128-v1':
        code=hashlib.blake2b(word.encode(),digest_size=16).digest()
        bits=np.unpackbits(np.frombuffer(code,dtype=np.uint8),bitorder='big').astype(np.float32)
        out=(2*bits-1)/np.float32(np.sqrt(128.))
    else:raise ValueError('Unknown fixed input code')
    out.setflags(write=False);return out


def context(metadata,prompt):
    if not isinstance(prompt,str) or not 1<=len(prompt.strip())<=2048:raise ValueError('Prompt1..2048characters')
    out=np.zeros(128,np.float32)
    for token in words(prompt):
        if token in metadata['idf']:out+=metadata['idf'][token]*vector(token,metadata['kind'])
    return out/max(1.,float(np.linalg.norm(out)))


def coverage(metadata,prompts):
    values=[]
    for prompt in prompts:
        tokens=words(prompt);values.append(sum(t in metadata['idf'] for t in tokens)/max(1,len(tokens)))
    return {'mean_known_word_fraction':float(np.mean(values)),'all_unknown':sum(v==0 for v in values),
        'word_types':len(metadata['idf']),'channels':128,
        'metadata_bytes':len(json.dumps(metadata,ensure_ascii=False).encode())}
