"""Small local mmcv compatibility shim for the CLRNet CPU/MPS demo."""


def jit(*args, **kwargs):
    def decorator(func):
        return func

    return decorator

