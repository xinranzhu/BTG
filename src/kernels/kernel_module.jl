"""
KernelModuleType sits at the top of the type hierarchy of kernel modules

All kernel module types subtype KernelModuleType, e.g.

RBFKernelModuleType <: KernelModuleType

"""
abstract type KernelModule end
