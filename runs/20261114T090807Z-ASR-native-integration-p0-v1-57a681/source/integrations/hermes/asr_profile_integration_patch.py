"""Exact-source ASR profile integration recipe, applied only after locked jobs end."""

import hashlib

EXPECTED={
    'config.py':'495a5a645c5ff3302fdcbe28bc3bbf538115c8f586fcaed2c2ad3b882fa04837',
    'backends.py':'d5967732d984f36826090b63dc68ff3fd63d6d1f9a4caa28da44d3cf54257b47',
    'qualification.py':'170a053eab6cf0d114bb56f53c4c361f71a711d9f7f5a243f0e864717786c849',
    'gateway.py':'7c0b93967ca44641aa238f4420b7f38276f651972f1b0e75e45c191d13ac375a',
}


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('ASR integration source anchor changed')
    return text.replace(before,after,1)


def config(text):
    text=replace(text,'import json\n','import json\nfrom ..runtime.asr_profiles import selected_profile\n')
    text=replace(text,'    asr_revision: str = ', '    asr_model_name: str = "openai/whisper-base"\n    asr_revision: str = ')
    anchor='        if type(self.searxng_enabled) is not bool:\n'
    text=replace(text,anchor,'        selected_profile(self.asr_model_name, self.asr_revision, self.execution_mode)\n'+anchor)
    anchor='        "model_revision",\n'
    return replace(text,anchor,anchor+'        "asr_model_name",\n        "asr_revision",\n')


ASR_METHODS='''    def asr_profile(self):
        return selected_profile(self.config.asr_model_name, self.config.asr_revision,
                                self.config.execution_mode)

    def asr_model_path(self):
        return model_path(self.directory, self.asr_profile())

    def ensure_asr_storage(self):
        path = self.asr_model_path()
        stat = path.stat()
        signature = (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        if signature != self.asr_asset_signature:
            self.asr_storage_receipt = None
            receipt = verify_asr_storage(self.directory, self.config.asr_model_name,
                                        self.config.asr_revision, self.config.execution_mode)
            current = path.stat()
            if signature != (str(path), current.st_size, current.st_mtime_ns, current.st_ctime_ns):
                raise RuntimeError("ASR asset changed while verifying its integrity")
            self.asr_storage_receipt = receipt
            self.asr_asset_signature = signature
        return path

    def wait_asr(self, timeout=120):
        deadline = time.monotonic() + timeout
        while True:
            code = self.asr.poll()
            if code is not None:
                return code
            self.monitor.check()
            if self.config.execution_mode == "gpu":
                if gpu_headroom(self.config.gpu_device)["free_MiB"] < 512:
                    raise MemoryError("ASR physical GPU reserve below 512 MiB; stop owned ASR without CPU fallback")
            if self.cancel_idle.is_set():
                raise InterruptedError("Foreground work preempts the owned background ASR chunk")
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned ASR chunk did not finish within its timeout")
            time.sleep(0.25)

'''


def backends(text):
    text=replace(text,'from .config import HybridConfig\n',
        'from .config import HybridConfig\nfrom ..runtime.asr_profiles import selected_profile, model_path, verify_asr_storage, verify_execution_log\n')
    text=replace(text,'        self.asset_signatures = None\n',
        '        self.asset_signatures = None\n        self.asr_asset_signature = None\n        self.asr_storage_receipt = None\n')
    text=replace(text,'    def whisper_executable(self):\n',ASR_METHODS+'    def whisper_executable(self):\n')
    anchor='    def pause_vlm_for_audio(self):\n'
    text=replace(text,anchor,anchor+'''        if (self.config.execution_mode == "gpu"
                and self.asr_profile().name == "openai/whisper-small"):
            headroom = gpu_headroom(self.config.gpu_device)
            if headroom["free_MiB"] < 1800:
                raise MemoryError("Whisper-small requires 1800 MiB physical launch headroom; VLM stays resident, no CPU fallback")
            return
''')
    text=replace(text,'        whisper = self.whisper_executable()\n',
        '        whisper = self.whisper_executable()\n        selected_asr = self.ensure_asr_storage()\n')
    text=replace(text,'            native / "whisper-base-f32.bin",\n','            selected_asr,\n')
    text=replace(text,'            str(self.directory / "models/native/whisper-base-f32.bin"),\n',
        '            str(self.asr_model_path()),\n')
    text=replace(text,'            self.pause_vlm_for_audio()\n',
        '            self.ensure_asr_storage()\n            self.pause_vlm_for_audio()\n')
    text=replace(text,'                if self.asr.wait(timeout=120):\n',
        '                if self.wait_asr(timeout=120):\n')
    anchor='            self.asr = self._spawn(command, "asr.log")\n'
    text=replace(text,anchor,'            asr_log = self.directory / "asr.log"\n'
        '            log_offset = asr_log.stat().st_size if asr_log.exists() else 0\n'+anchor)
    anchor='                path = output.with_suffix(".json")\n'
    text=replace(text,anchor,'''                with asr_log.open("rb") as current_log:
                    current_log.seek(log_offset)
                    fresh_log = current_log.read(262145)
                if len(fresh_log) > 262144:
                    raise ValueError("ASR worker log exceeds bounded current-call evidence")
                execution = verify_execution_log(fresh_log.decode("utf-8", errors="replace"),
                                                 self.config.execution_mode, self.config.gpu_device)
'''+anchor)
    anchor='                    "precision": "FP32",\n'
    return replace(text,anchor,anchor+'                    "ASR_identity": self.asr_profile().identity(),\n'
        '                    "model_path": str(self.asr_model_path()),\n'
        '                    "storage_verified": self.asr_storage_receipt is not None,\n'
        '                    **execution,\n')


def qualification(text):
    text=replace(text,'from .store import encode\n','from .store import encode\nfrom ..runtime.asr_profiles import verify_asr_storage\n')
    anchor='        "audit_run": report["run_id"],\n'
    return replace(text,anchor,anchor+'        "selected_ASR_storage": verify_asr_storage(directory, config.asr_model_name,\n'
        '            config.asr_revision, config.execution_mode) if config is not None else None,\n')


def gateway(text):
    anchor='            "pid": os.getpid(),\n'
    return replace(text,anchor,'''            "ASR_identity": {
                "name": self.config.asr_model_name,
                "revision": self.config.asr_revision,
                "precision": "originalFP32",
                "storage_verified": getattr(self.backend, "asr_storage_receipt", None) is not None,
            },
'''+anchor)


TRANSFORMS={'config.py':config,'backends.py':backends,'qualification.py':qualification,'gateway.py':gateway}


def patch_bytes(filename,original):
    if filename not in EXPECTED or hashlib.sha256(original).hexdigest()!=EXPECTED[filename]:
        raise ValueError('Frozen ASR integration source changed; review before patch')
    eol=b'\r\n' if b'\r\n' in original else b'\n'
    text=original.decode('utf-8').replace('\r\n','\n')
    result=TRANSFORMS[filename](text).encode().replace(b'\n',eol)
    compile(result,filename,'exec')
    return result
