"""Read-only Windows PDH shared-GPU accounting; no driver/settings changes."""

import ctypes as c
import os
import re
import time


class SharedGPU:
    def __init__(self, counter_name="Shared Usage"):
        self.available = False
        self.error = None
        self.values = {}
        self.last = 0.0
        if os.name != "nt":
            self.error = "Not Windows"
            return
        try:
            from ctypes import wintypes as w

            self.pdh = c.WinDLL("pdh")
            self.query = c.c_void_p()
            self.counter = c.c_void_p()
            self.pdh.PdhOpenQueryW.argtypes = [
                w.LPCWSTR,
                c.c_size_t,
                c.POINTER(c.c_void_p),
            ]
            self.pdh.PdhAddEnglishCounterW.argtypes = [
                c.c_void_p,
                w.LPCWSTR,
                c.c_size_t,
                c.POINTER(c.c_void_p),
            ]
            self.pdh.PdhCollectQueryData.argtypes = [c.c_void_p]
            self.pdh.PdhGetFormattedCounterArrayW.argtypes = [
                c.c_void_p,
                w.DWORD,
                c.POINTER(w.DWORD),
                c.POINTER(w.DWORD),
                c.c_void_p,
            ]
            if self.pdh.PdhOpenQueryW(None, 0, c.byref(self.query)) != 0:
                raise OSError("PDH query unavailable")
            if (
                self.pdh.PdhAddEnglishCounterW(
                    self.query,
                    "\\GPU Process Memory(*)\\" + counter_name,
                    0,
                    c.byref(self.counter),
                )
                != 0
            ):
                raise OSError("GPU shared counter unavailable")
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def sample(self, pids):
        if self.available and time.monotonic() - self.last >= 2:
            self.last = time.monotonic()
            try:
                from ctypes import wintypes as w

                class Value(c.Union):
                    _fields_ = [
                        ("long", c.c_long),
                        ("double", c.c_double),
                        ("large", c.c_longlong),
                        ("pointer", c.c_void_p),
                    ]

                class Format(c.Structure):
                    _fields_ = [("status", w.DWORD), ("value", Value)]

                class Item(c.Structure):
                    _fields_ = [("name", w.LPWSTR), ("formatted", Format)]

                self.pdh.PdhCollectQueryData(self.query)
                size = w.DWORD()
                count = w.DWORD()
                self.pdh.PdhGetFormattedCounterArrayW(
                    self.counter, 0x400, c.byref(size), c.byref(count), None
                )
                if size.value:
                    buffer = c.create_string_buffer(size.value)
                    result = self.pdh.PdhGetFormattedCounterArrayW(
                        self.counter, 0x400, c.byref(size), c.byref(count), buffer
                    )
                    if result != 0:
                        raise OSError("PDH formatting failed: " + str(result))
                    array = c.cast(buffer, c.POINTER(Item))
                    values = {}
                    for i in range(count.value):
                        match = re.match(r"pid_(\d+)_", array[i].name or "")
                        if match and array[i].formatted.status in (0, 1):
                            pid = int(match.group(1))
                            values[pid] = values.get(pid, 0) + max(
                                0, array[i].formatted.value.large
                            )
                    self.values = values
            except Exception as exc:
                self.error = str(exc)
        return {
            "available": self.available and not self.error,
            "bytes": sum(self.values.get(pid, 0) for pid in pids),
            "sampling_seconds": 2,
            "error": self.error,
            "deduplication": "conservative addition to summed RSS; possible shared-page double count",
        }

    def close(self):
        if self.available:
            self.pdh.PdhCloseQuery.argtypes = [c.c_void_p]
            self.pdh.PdhCloseQuery(self.query)
            self.available = False


class DedicatedGPU(SharedGPU):
    def __init__(self):
        super().__init__("Dedicated Usage")
