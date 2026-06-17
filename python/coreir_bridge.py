"""
CoreIR Python Bridge
Direct ctypes bindings to the CoreIR C runtime.
"""
import ctypes
import os
from ctypes import c_double, c_int, c_size_t, c_bool, c_void_p, POINTER, Structure, byref

_lib_path = os.path.join(os.path.dirname(__file__), "..", "build", "libcoreir.so")
_lib = ctypes.CDLL(_lib_path)

class Tensor(Structure):
    _fields_ = [
        ("data", c_void_p),
        ("size", c_size_t),
        ("shape", POINTER(c_int)),
        ("ndim", c_int),
    ]

class Differentiable(Structure):
    pass

class SearchSpace(Structure):
    pass

_lib.coreir_init.argtypes = []
_lib.coreir_init.restype = None

_lib.coreir_shutdown.argtypes = []
_lib.coreir_shutdown.restype = None

_lib.coreir_diff_register.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t]
_lib.coreir_diff_register.restype = POINTER(Differentiable)

_lib.coreir_search_create.argtypes = [c_double]
_lib.coreir_search_create.restype = POINTER(SearchSpace)

_lib.coreir_search_add.argtypes = [POINTER(SearchSpace), c_void_p, c_double]
_lib.coreir_search_add.restype = None

_lib.coreir_search_sample.argtypes = [POINTER(SearchSpace)]
_lib.coreir_search_sample.restype = c_void_p

_lib.coreir_search_free.argtypes = [POINTER(SearchSpace)]
_lib.coreir_search_free.restype = None

_lib.coreir_learn_step.argtypes = []
_lib.coreir_learn_step.restype = None

_lib.coreir_get_performance.argtypes = []
_lib.coreir_get_performance.restype = c_double

class CoreIR:
    def __init__(self):
        self._model = None
        _lib.coreir_init()
    
    def shutdown(self):
        _lib.coreir_shutdown()
    
    def create_search(self, temperature=1.0):
        self._space = _lib.coreir_search_create(temperature)
        return self._space
    
    def search_add(self, candidate, score):
        _lib.coreir_search_add(self._space, candidate, score)
    
    def search_sample(self):
        return _lib.coreir_search_sample(self._space)
    
    def learn_step(self):
        _lib.coreir_learn_step()
    
    def get_performance(self):
        return _lib.coreir_get_performance()

if __name__ == "__main__":
    print("=== CoreIR Python Bridge Demo ===")
    runtime = CoreIR()
    runtime.create_search(0.5)
    runtime.search_add(ctypes.c_int(1), 0.7)
    runtime.search_add(ctypes.c_int(2), 0.9)
    best = runtime.search_sample()
    print(f"Best: {best}")
    for i in range(3):
        runtime.learn_step()
    print(f"Performance: {runtime.get_performance():.3f}")
    runtime.shutdown()
