# Centralized place to load Fortran and C++/CUDA extensions

_c2ray_lib = None
_c2ray_lib_loaded = None
_asora_lib = None
_asora_lib_loaded = None
_asora_he_lib = None
_asora_he_lib_loaded = None


# Load the f2py-compiled C2Ray subroutines
def load_c2ray():
    global _c2ray_lib
    global _c2ray_lib_loaded

    # Try once to load the library
    if _c2ray_lib_loaded is None:
        try:
            from .lib import libc2ray

            _c2ray_lib_loaded = True
            _c2ray_lib = libc2ray
            return _c2ray_lib
        except ImportError:
            # If C2Ray is not found, stop execution (the package cannot be used without)
            _c2ray_lib_loaded = False
            raise RuntimeError("Could not load c2ray library (image not found)")
    elif _c2ray_lib_loaded is True:
        return _c2ray_lib
    else:
        return None


# Load the ASORA raytracing library
def load_asora():
    global _asora_lib
    global _asora_lib_loaded

    # Try once to load the library
    if _asora_lib_loaded is None:
        try:
            from .lib import libasora

            _asora_lib_loaded = True
            _asora_lib = libasora
            return _asora_lib
        except ImportError:
            # If asora is not found, the package can still be used but we inform the user that GPU raytracing is not available
            print("Info: could not load ASORA library (image not found)")
            _asora_lib_loaded = False
    elif _asora_lib_loaded is True:
        return _asora_lib
    else:
        return None


# Load the ASORA (with Helium) raytracing library
def load_asora_he():
    global _asora_he_lib
    global _asora_he_lib_loaded

    # Try once to load the library
    if _asora_he_lib_loaded is None:
        try:
            from .lib import libasora_He

            _asora_he_lib_loaded = True
            _asora_he_lib = libasora_He
            return _asora_he_lib
        except ImportError:
            # If asora is not found, the package can still be used but we inform the user that GPU raytracing is not available
            print("Info: could not load ASORA (with Helium) library (image not found)")
            _asora_he_lib_loaded = False
    elif _asora_he_lib_loaded is True:
        return _asora_he_lib
    else:
        return None
