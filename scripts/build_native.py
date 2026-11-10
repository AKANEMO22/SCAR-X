"""Build local CPU DLL/shared object; never download a compiler."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
build = root/"build"
build.mkdir(exist_ok=True)
source = root/"native"/"core.cpp"
if os.name == "nt":
    scripts = sorted(Path("C:/Program Files/Microsoft Visual Studio").glob("*/Community/VC/Auxiliary/Build/vcvars64.bat"))
    scripts += sorted(Path("C:/Program Files (x86)/Microsoft Visual Studio").glob("*/BuildTools/VC/Auxiliary/Build/vcvars64.bat"))
    if not scripts:
        raise SystemExit("MSVC x64 developer environment not found")
    # All values are fixed local paths discovered above, not user-provided shell fragments.
    command = f'call "{scripts[-1]}" >nul && cl /nologo /O2 /std:c++17 /EHsc /LD /arch:AVX2 "{source}" /Fe:scarx_cpu.dll'
    # A batch file avoids Python's Windows list quoting escaping embedded quotes.
    batch = build/"compile.cmd"
    batch.write_text("@echo off\n"+command+"\nexit /b %errorlevel%\n",encoding="utf-8")
    subprocess.run(["cmd.exe","/d","/c","compile.cmd"],cwd=build,check=True)
else:
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise SystemExit("C++ compiler required")
    subprocess.run([compiler,"-O3","-std=c++17","-fPIC","-shared",str(source),"-o",str(build/"libscarx_cpu.so")],check=True)
print(build)
