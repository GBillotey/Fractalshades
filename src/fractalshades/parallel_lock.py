# -*- coding: utf-8 -*-
import numba as nb
"""
Implements atomic compare-and-swap in user-land using llvm_call() and use it to
create a lock.
Ref:
https://gist.github.com/sklam/40f25167351832fe55b64232785d036d

https://llvm.org/docs/LangRef.html#cmpxchg-instruction
The ‘cmpxchg’ instruction is used to atomically modify memory. It loads a value
in memory and compares it to a given value. If they are equal, it tries to
store a new value into the memory.
"""

@nb.extending.intrinsic
def atomic_xchg(context, ptr, cmp, val):
    if isinstance(ptr, nb.types.CPointer):
        valtype = ptr.dtype
        sig = valtype(ptr, valtype, valtype)

        def codegen(context, builder, signature, args):
            [ptr, cmpval, value] = args
            res = builder.cmpxchg(ptr, cmpval, value, ordering='monotonic')
            oldval, succ = nb.core.cgutils.unpack_tuple(builder, res)
            return oldval
        return sig, codegen

@nb.extending.intrinsic
def cast_as_intp_ptr(context, ptrval):
    ptrty = nb.types.CPointer(nb.intp)
    sig = ptrty(nb.intp)

    def codegen(context, builder, signature, args):
        [val] = args
        llrety = context.get_value_type(signature.return_type)
        return builder.inttoptr(val, llrety)
    return sig, codegen

@nb.njit("intp(intp[:])")
def try_lock(lock):
    iptr = cast_as_intp_ptr(lock[0:].ctypes.data)
    old = atomic_xchg(iptr, 0, 1)
    return old == 0

@nb.njit("void(intp[:])")
def unlock(lock):
    iptr = cast_as_intp_ptr(lock[0:].ctypes.data)
    old = atomic_xchg(iptr, 1, 0)
    assert old == 1
