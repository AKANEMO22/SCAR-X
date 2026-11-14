"""Pinned original-FP32 ASR assets; independent of a VLM or learning algorithm."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ASRProfile:
    name:str
    revision:str
    relative_path:str
    native_bytes:int
    native_sha256:str
    source_sha256:str
    tensor_count:int

    def identity(self):
        return {'name':self.name,'revision':self.revision,'precision':'originalFP32',
                'native_sha256':self.native_sha256}


PROFILES={
    'openai/whisper-base':ASRProfile(
        'openai/whisper-base','e37978b90ca9030d5170a5c07aadb050351a65bb',
        'models/native/whisper-base-f32.bin',290_960_250,
        '6a15df05602c21d513a0912bb02725280927ff72a32e150d1a053a2d8ec62cd1',
        '07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b',245),
    'openai/whisper-small':ASRProfile(
        'openai/whisper-small','973afd24965f72e36ca33b3055d56a652f456b4d',
        'candidates/whisper-small/whisper-small-f32.bin',967_535_936,
        'fcfe5ac3405eb920581f41444238ef9b5d86a6c57356c96cd1e486269c60072e',
        '1d7734884874f1a1513ed9aa760a4f8e97aaa02fd6d93a3a85d27b2ae9ca596b',479),
}


def selected_profile(name,revision,execution_mode):
    if not isinstance(name,str) or not isinstance(revision,str):
        raise ValueError('ASR identity fields must be explicit strings')
    profile=PROFILES.get(name)
    if profile is None or profile.revision!=revision:
        raise ValueError('Unregistered native ASR identity/revision')
    if execution_mode not in ('cpu','gpu'):
        raise ValueError('Expected explicit cpu/gpu ASR mode')
    if profile.name=='openai/whisper-small' and execution_mode!='gpu':
        raise ValueError('Whisper-small CPU profile is not resource-qualified; choose base or GPU')
    return profile


def model_path(directory,profile):
    if profile not in PROFILES.values():raise ValueError('Unregistered ASR profile')
    root=Path(directory).resolve()
    target=(root/profile.relative_path).resolve()
    if not target.is_relative_to(root):raise ValueError('ASR asset escapes owned data directory')
    return target


def read_json(path,limit=65536):
    if not path.is_file() or path.stat().st_size>limit:
        raise ValueError('Missing or oversized ASR storage receipt')
    try:value=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,UnicodeError) as exc:raise ValueError('Invalid ASR storage receipt') from exc
    if not isinstance(value,dict):raise ValueError('ASR storage receipt must be an object')
    return value


def validate_receipt(profile,receipt):
    """Check reported provenance against known pins; runtime still hashes the file."""
    if profile not in PROFILES.values() or not isinstance(receipt,dict):
        raise ValueError('Expected a pinned ASR profile and object receipt')
    if profile.name=='openai/whisper-base':
        if receipt.get('status')!='NATIVE_STORAGE_INTEGRITY_PASS' or receipt.get('quantized') is not False:
            raise ValueError('Baseline native storage receipt did not pass')
        exports,source=receipt.get('exports'),receipt.get('source')
        if not isinstance(exports,dict) or not isinstance(source,dict):
            raise ValueError('Baseline ASR manifest sections missing')
        source_entry=source.get('models/whisper-base/model.safetensors')
        if not isinstance(source_entry,dict) or source_entry.get('sha256')!=profile.source_sha256:
            raise ValueError('Baseline ASR source identity mismatch')
        entry=exports.get('whisper-base-f32.bin',{})
        if not isinstance(entry,dict):raise ValueError('Baseline ASR export must be an object')
        if entry.get('dtype')!='F32' or entry.get('direct_exact_tensor_readback_count')!=profile.tensor_count:
            raise ValueError('Baseline ASR dtype/tensor provenance mismatch')
    else:
        if (receipt.get('status')!='NATIVE_FP32_STORAGE_READY_INFERENCE_PENDING'
                or receipt.get('source_revision')!=profile.revision
                or receipt.get('source_sha256')!=profile.source_sha256
                or receipt.get('tensor_count')!=profile.tensor_count):
            raise ValueError('Candidate ASR source/tensor provenance mismatch')
        entry=receipt.get('export',{})
        if not isinstance(entry,dict):raise ValueError('Candidate ASR export must be an object')
        if entry.get('precision')!='originalFP32' or entry.get('every_tensor_exact_readback') is not True:
            raise ValueError('Candidate ASR original precision/readback evidence missing')
    if (type(entry.get('bytes')) is not int or entry['bytes']!=profile.native_bytes
            or entry.get('sha256')!=profile.native_sha256):
        raise ValueError('ASR export pin does not match selected profile')
    return {'storage_audit_run':receipt.get('run_id'),'quality_qualification_implied':False}


def verify_asr_storage(directory,name,revision,execution_mode):
    profile=selected_profile(name,revision,execution_mode)
    path=model_path(directory,profile)
    root=Path(directory).resolve()
    receipt_path=(root/'native-integrity.json' if profile.name=='openai/whisper-base' else
                  root/'candidates/whisper-small/native-storage-receipt.json').resolve()
    if not receipt_path.is_relative_to(root):raise ValueError('ASR receipt escapes owned data directory')
    provenance=validate_receipt(profile,read_json(receipt_path))
    if not path.is_file() or path.stat().st_size!=profile.native_bytes:
        raise ValueError('Selected native ASR file missing or byte-size changed')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        while block:=f.read(2<<20):digest.update(block)
    if digest.hexdigest()!=profile.native_sha256:
        raise ValueError('Selected native ASR checksum changed')
    return {'status':'PINNED_NATIVE_ASR_STORAGE_PASS','path':str(path),'bytes':profile.native_bytes,
            **profile.identity(),**provenance,'runtime_inference_verified_here':False}
