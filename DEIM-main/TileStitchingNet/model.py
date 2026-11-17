"""
Step 3: Boundary-Aware Local Relation Attention Network (TileStitchingNet)
Includes:
  - 3.1 Sparse Local Adjacency Mask (M_local & M_stitch)
  - 3.2 Pairwise 8D Geometric & Cut-Alignment Bias (g_ij)
  - 3.3 2-Layer Local Relation Self-Attention (BALR-Attention)
  - 3.4 Explicit Soft Bounding Envelope Operator & Dual-Branch Output Heads

Note: Uses native PyTorch (`torch`, `torch.nn`, `torch.nn.functional`) when installed,
and automatically falls back to a built-in NumPy Reverse-Mode Autograd Engine when
running in environments without PyTorch installed.

Module: tile_stitching_postprocess/model.py
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# ============================================================================
# 0. PyTorch Import with Built-in Differentiable Autograd Fallback
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_BACKEND = "pytorch"
    Tensor = torch.Tensor
except ImportError:
    TORCH_BACKEND = "numpy_autograd"

    class Tensor:
        """Lightweight reverse-mode Autograd Tensor compatible with PyTorch syntax."""

        def __init__(
            self,
            data: Union[np.ndarray, float, int, list],
            requires_grad: bool = False,
            _children: Tuple["Tensor", ...] = (),
            _op: str = "",
            dtype=np.float32,
        ):
            if isinstance(data, Tensor):
                arr = data.data.astype(dtype, copy=False)
            elif isinstance(data, np.ndarray):
                arr = data.astype(dtype, copy=False)
            else:
                arr = np.asarray(data, dtype=dtype)
            self.data: np.ndarray = arr
            self.requires_grad: bool = bool(requires_grad)
            self.grad: Optional[np.ndarray] = None
            self._backward = lambda: None
            self._prev = tuple(_children)
            self._op = _op

        @property
        def shape(self) -> Tuple[int, ...]:
            return self.data.shape

        @property
        def ndim(self) -> int:
            return self.data.ndim

        @property
        def dtype(self):
            return self.data.dtype

        @property
        def device(self) -> str:
            return "cpu"

        def numel(self) -> int:
            return int(self.data.size)

        def item(self) -> float:
            return float(self.data.item())

        def detach(self) -> "Tensor":
            return Tensor(self.data.copy(), requires_grad=False, dtype=self.data.dtype)

        def numpy(self) -> np.ndarray:
            return self.data

        def float(self) -> "Tensor":
            if self.data.dtype == np.float32:
                return self
            return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

        def long(self) -> "Tensor":
            return Tensor(self.data.astype(np.int64), requires_grad=False, dtype=np.int64)

        def bool(self) -> "Tensor":
            return Tensor(self.data.astype(bool), requires_grad=False, dtype=bool)

        def to(self, *args, **kwargs) -> "Tensor":
            return self

        def clone(self) -> "Tensor":
            out = Tensor(self.data.copy(), requires_grad=self.requires_grad, _children=(self,), _op="clone")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad)
                out._backward = _backward
            return out

        def _add_grad(self, g: np.ndarray):
            if not self.requires_grad:
                return
            g = np.asarray(g, dtype=np.float32)
            # Un-broadcast gradient to match self.data.shape
            while g.ndim > self.data.ndim:
                g = g.sum(axis=0)
            for ax, (dim_self, dim_g) in enumerate(zip(self.data.shape, g.shape)):
                if dim_self == 1 and dim_g > 1:
                    g = g.sum(axis=ax, keepdims=True)
            if self.grad is None:
                self.grad = g.copy()
            else:
                self.grad = self.grad + g

        def zero_grad(self):
            self.grad = None

        def backward(self, grad: Optional[np.ndarray] = None):
            topo: List[Tensor] = []
            visited = set()

            def build_topo(v: Tensor):
                if id(v) not in visited:
                    visited.add(id(v))
                    for child in v._prev:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)
            if grad is None:
                self.grad = np.ones_like(self.data, dtype=np.float32)
            else:
                self.grad = np.asarray(grad, dtype=np.float32)

            for v in reversed(topo):
                v._backward()

        def __getitem__(self, idx) -> "Tensor":
            if isinstance(idx, Tensor):
                idx = idx.data
            elif isinstance(idx, tuple):
                idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
            out = Tensor(self.data[idx], requires_grad=self.requires_grad, _children=(self,), _op="getitem", dtype=self.data.dtype)
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        g = np.zeros_like(self.data, dtype=np.float32)
                        np.add.at(g, idx, out.grad)
                        self._add_grad(g)
                out._backward = _backward
            return out

        def __neg__(self) -> "Tensor":
            return self * (-1.0)

        def __add__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            req = self.requires_grad or other.requires_grad
            out = Tensor(self.data + other.data, requires_grad=req, _children=(self, other), _op="+")
            if req:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad)
                        other._add_grad(out.grad)
                out._backward = _backward
            return out

        def __radd__(self, other) -> "Tensor":
            return self.__add__(other)

        def __sub__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            req = self.requires_grad or other.requires_grad
            out = Tensor(self.data - other.data, requires_grad=req, _children=(self, other), _op="-")
            if req:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad)
                        other._add_grad(-out.grad)
                out._backward = _backward
            return out

        def __rsub__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            return other.__sub__(self)

        def __mul__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            req = self.requires_grad or other.requires_grad
            out = Tensor(self.data * other.data, requires_grad=req, _children=(self, other), _op="*")
            if req:
                def _backward():
                    if out.grad is not None:
                        if self.requires_grad:
                            self._add_grad(out.grad * other.data)
                        if other.requires_grad:
                            other._add_grad(out.grad * self.data)
                out._backward = _backward
            return out

        def __rmul__(self, other) -> "Tensor":
            return self.__mul__(other)

        def __truediv__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            req = self.requires_grad or other.requires_grad
            out = Tensor(self.data / other.data, requires_grad=req, _children=(self, other), _op="/")
            if req:
                def _backward():
                    if out.grad is not None:
                        if self.requires_grad:
                            self._add_grad(out.grad / other.data)
                        if other.requires_grad:
                            other._add_grad(-out.grad * self.data / (other.data ** 2 + 1e-12))
                out._backward = _backward
            return out

        def __rtruediv__(self, other) -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            return other.__truediv__(self)

        def __pow__(self, power: float) -> "Tensor":
            req = self.requires_grad
            out = Tensor(self.data ** power, requires_grad=req, _children=(self,), _op=f"**{power}")
            if req:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad * power * (self.data ** (power - 1.0)))
                out._backward = _backward
            return out

        def __matmul__(self, other: "Tensor") -> "Tensor":
            other = other if isinstance(other, Tensor) else Tensor(other, dtype=np.float32)
            req = self.requires_grad or other.requires_grad
            out = Tensor(np.matmul(self.data, other.data), requires_grad=req, _children=(self, other), _op="@")
            if req:
                def _backward():
                    if out.grad is not None:
                        if self.requires_grad:
                            self._add_grad(np.matmul(out.grad, np.swapaxes(other.data, -1, -2)))
                        if other.requires_grad:
                            other._add_grad(np.matmul(np.swapaxes(self.data, -1, -2), out.grad))
                out._backward = _backward
            return out

        def transpose(self, dim0: int, dim1: int) -> "Tensor":
            axes = list(range(self.data.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            out = Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad, _children=(self,), _op="transpose")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(np.transpose(out.grad, axes))
                out._backward = _backward
            return out

        @property
        def T(self) -> "Tensor":
            return self.transpose(-2, -1)

        def reshape(self, *shape) -> "Tensor":
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            out = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, _children=(self,), _op="reshape")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad.reshape(self.data.shape))
                out._backward = _backward
            return out

        def view(self, *shape) -> "Tensor":
            return self.reshape(*shape)

        def unsqueeze(self, dim: int) -> "Tensor":
            out = Tensor(np.expand_dims(self.data, axis=dim), requires_grad=self.requires_grad, _children=(self,), _op="unsqueeze")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(np.squeeze(out.grad, axis=dim))
                out._backward = _backward
            return out

        def squeeze(self, dim: Optional[int] = None) -> "Tensor":
            out_data = np.squeeze(self.data, axis=dim) if dim is not None else np.squeeze(self.data)
            out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="squeeze")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(out.grad.reshape(self.data.shape))
                out._backward = _backward
            return out

        def sum(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "Tensor":
            out_data = self.data.sum(axis=dim, keepdims=keepdim)
            out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="sum")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        g = out.grad
                        if dim is not None and not keepdim:
                            axes = [dim] if isinstance(dim, int) else sorted([d % self.data.ndim for d in dim])
                            for ax in axes:
                                g = np.expand_dims(g, axis=ax)
                        self._add_grad(np.broadcast_to(g, self.data.shape))
                out._backward = _backward
            return out

        def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> "Tensor":
            if dim is None:
                count = max(1, self.data.size)
            elif isinstance(dim, int):
                count = max(1, self.data.shape[dim])
            else:
                count = max(1, int(np.prod([self.data.shape[d] for d in dim])))
            return self.sum(dim=dim, keepdim=keepdim) * (1.0 / float(count))

        def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "Tensor":
            out_data = np.clip(self.data, min, max)
            out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="clamp")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        mask = np.ones_like(self.data, dtype=np.float32)
                        if min is not None:
                            mask *= (self.data >= min)
                        if max is not None:
                            mask *= (self.data <= max)
                        self._add_grad(out.grad * mask)
                out._backward = _backward
            return out

        def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
            m = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
            out_data = np.where(m, value, self.data)
            out = Tensor(out_data, requires_grad=self.requires_grad, _children=(self,), _op="masked_fill")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        self._add_grad(np.where(m, 0.0, out.grad))
                out._backward = _backward
            return out

        def max(self, dim: Optional[int] = None, keepdim: bool = False):
            if dim is None:
                idx = np.argmax(self.data)
                out = Tensor(self.data.flat[idx], requires_grad=self.requires_grad, _children=(self,), _op="max")
                if self.requires_grad:
                    def _backward():
                        if out.grad is not None:
                            g = np.zeros_like(self.data, dtype=np.float32)
                            g.flat[idx] = out.grad
                            self._add_grad(g)
                    out._backward = _backward
                return out
            idx = np.argmax(self.data, axis=dim)
            vals = np.max(self.data, axis=dim, keepdims=keepdim)
            out = Tensor(vals, requires_grad=self.requires_grad, _children=(self,), _op="max_dim")
            if self.requires_grad:
                def _backward():
                    if out.grad is not None:
                        g_in = out.grad if keepdim else np.expand_dims(out.grad, axis=dim)
                        idx_exp = np.expand_dims(idx, axis=dim)
                        g = np.zeros_like(self.data, dtype=np.float32)
                        np.put_along_axis(g, idx_exp, g_in, axis=dim)
                        self._add_grad(g)
                out._backward = _backward
            return out, Tensor(idx, requires_grad=False, dtype=np.int64)

        def min(self, dim: Optional[int] = None, keepdim: bool = False):
            if dim is None:
                res = (-self).max()
                return -res
            vals, idx = (-self).max(dim=dim, keepdim=keepdim)
            return -vals, idx

        # Comparison operators returning boolean Tensors
        def __lt__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data < o, requires_grad=False, dtype=bool)

        def __le__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data <= o, requires_grad=False, dtype=bool)

        def __gt__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data > o, requires_grad=False, dtype=bool)

        def __ge__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data >= o, requires_grad=False, dtype=bool)

        def __eq__(self, other) -> "Tensor":  # type: ignore[override]
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data == o, requires_grad=False, dtype=bool)

        def __ne__(self, other) -> "Tensor":  # type: ignore[override]
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(self.data != o, requires_grad=False, dtype=bool)

        def __and__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(np.logical_and(self.data, o), requires_grad=False, dtype=bool)

        def __or__(self, other) -> "Tensor":
            o = other.data if isinstance(other, Tensor) else other
            return Tensor(np.logical_or(self.data, o), requires_grad=False, dtype=bool)

        def __invert__(self) -> "Tensor":
            return Tensor(np.logical_not(self.data), requires_grad=False, dtype=bool)

    class _NoGradContext:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def __call__(self, fn):
            def wrapper(*args, **kwargs):
                with self:
                    return fn(*args, **kwargs)
            return wrapper

    class _TorchNamespace:
        Tensor = Tensor
        float32 = np.float32
        int64 = np.int64
        bool = bool

        @staticmethod
        def tensor(data, dtype=np.float32, device=None, requires_grad: bool = False) -> Tensor:
            return Tensor(data, requires_grad=requires_grad, dtype=dtype)

        @staticmethod
        def from_numpy(arr: np.ndarray) -> Tensor:
            return Tensor(arr, requires_grad=False, dtype=arr.dtype)

        @staticmethod
        def zeros(*shape, dtype=np.float32, device=None, requires_grad: bool = False) -> Tensor:
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

        @staticmethod
        def ones(*shape, dtype=np.float32, device=None, requires_grad: bool = False) -> Tensor:
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad, dtype=dtype)

        @staticmethod
        def eye(n: int, dtype=np.float32, device=None) -> Tensor:
            return Tensor(np.eye(n, dtype=dtype), requires_grad=False, dtype=dtype)

        @staticmethod
        def arange(*args, dtype=np.int64, device=None) -> Tensor:
            return Tensor(np.arange(*args, dtype=dtype), requires_grad=False, dtype=dtype)

        @staticmethod
        def abs(x: Tensor) -> Tensor:
            out = Tensor(np.abs(x.data), requires_grad=x.requires_grad, _children=(x,), _op="abs")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad * np.sign(x.data))
                out._backward = _backward
            return out

        @staticmethod
        def sqrt(x: Tensor) -> Tensor:
            s = np.sqrt(np.maximum(x.data, 1e-12))
            out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="sqrt")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad * 0.5 / s)
                out._backward = _backward
            return out

        @staticmethod
        def log(x: Tensor) -> Tensor:
            safe_x = np.maximum(x.data, 1e-12)
            out = Tensor(np.log(safe_x), requires_grad=x.requires_grad, _children=(x,), _op="log")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad / safe_x)
                out._backward = _backward
            return out

        @staticmethod
        def exp(x: Tensor) -> Tensor:
            e = np.exp(np.clip(x.data, -40.0, 40.0))
            out = Tensor(e, requires_grad=x.requires_grad, _children=(x,), _op="exp")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad * e)
                out._backward = _backward
            return out

        @staticmethod
        def sigmoid(x: Tensor) -> Tensor:
            s = 1.0 / (1.0 + np.exp(-np.clip(x.data, -40.0, 40.0)))
            out = Tensor(s, requires_grad=x.requires_grad, _children=(x,), _op="sigmoid")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad * s * (1.0 - s))
                out._backward = _backward
            return out

        @staticmethod
        def clamp(x: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
            return x.clamp(min=min, max=max)

        @staticmethod
        def maximum(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
            ta = a if isinstance(a, Tensor) else Tensor(a)
            tb = b if isinstance(b, Tensor) else Tensor(b)
            req = ta.requires_grad or tb.requires_grad
            out = Tensor(np.maximum(ta.data, tb.data), requires_grad=req, _children=(ta, tb), _op="maximum")
            if req:
                def _backward():
                    if out.grad is not None:
                        m = (ta.data >= tb.data).astype(np.float32)
                        ta._add_grad(out.grad * m)
                        tb._add_grad(out.grad * (1.0 - m))
                out._backward = _backward
            return out

        @staticmethod
        def minimum(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
            ta = a if isinstance(a, Tensor) else Tensor(a)
            tb = b if isinstance(b, Tensor) else Tensor(b)
            req = ta.requires_grad or tb.requires_grad
            out = Tensor(np.minimum(ta.data, tb.data), requires_grad=req, _children=(ta, tb), _op="minimum")
            if req:
                def _backward():
                    if out.grad is not None:
                        m = (ta.data <= tb.data).astype(np.float32)
                        ta._add_grad(out.grad * m)
                        tb._add_grad(out.grad * (1.0 - m))
                out._backward = _backward
            return out

        @staticmethod
        def where(cond: Tensor, x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
            c = cond.data.astype(bool) if isinstance(cond, Tensor) else np.asarray(cond, dtype=bool)
            tx = x if isinstance(x, Tensor) else Tensor(x)
            ty = y if isinstance(y, Tensor) else Tensor(y)
            req = tx.requires_grad or ty.requires_grad
            out = Tensor(np.where(c, tx.data, ty.data), requires_grad=req, _children=(tx, ty), _op="where")
            if req:
                def _backward():
                    if out.grad is not None:
                        if tx.requires_grad:
                            tx._add_grad(np.where(c, out.grad, 0.0))
                        if ty.requires_grad:
                            ty._add_grad(np.where(c, 0.0, out.grad))
                out._backward = _backward
            return out

        @staticmethod
        def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
            req = any(t.requires_grad for t in tensors)
            out_data = np.stack([t.data for t in tensors], axis=dim)
            out = Tensor(out_data, requires_grad=req, _children=tuple(tensors), _op="stack")
            if req:
                def _backward():
                    if out.grad is not None:
                        grads = np.split(out.grad, len(tensors), axis=dim)
                        for t, g in zip(tensors, grads):
                            t._add_grad(np.squeeze(g, axis=dim))
                out._backward = _backward
            return out

        @staticmethod
        def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
            req = any(t.requires_grad for t in tensors)
            out_data = np.concatenate([t.data for t in tensors], axis=dim)
            out = Tensor(out_data, requires_grad=req, _children=tuple(tensors), _op="cat")
            if req:
                sizes = np.cumsum([t.data.shape[dim] for t in tensors[:-1]])
                def _backward():
                    if out.grad is not None:
                        grads = np.split(out.grad, sizes, axis=dim)
                        for t, g in zip(tensors, grads):
                            t._add_grad(g)
                out._backward = _backward
            return out

        @staticmethod
        def topk(x: Tensor, k: int, dim: int = -1, largest: bool = True):
            arr = x.data if largest else -x.data
            k = min(k, arr.shape[dim])
            idx = np.argsort(arr, axis=dim)
            idx = np.take(idx, np.arange(arr.shape[dim] - 1, arr.shape[dim] - 1 - k, -1), axis=dim)
            vals = np.take_along_axis(x.data, idx, axis=dim)
            return Tensor(vals, requires_grad=False), Tensor(idx, requires_grad=False, dtype=np.int64)

        @staticmethod
        def atan(x: Tensor) -> Tensor:
            out = Tensor(np.arctan(x.data), requires_grad=x.requires_grad, _children=(x,), _op="atan")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad / (1.0 + x.data ** 2))
                out._backward = _backward
            return out

        @staticmethod
        def no_grad():
            return _NoGradContext()

        @staticmethod
        def manual_seed(seed: int):
            np.random.seed(seed)

    class _FunctionalNamespace:
        @staticmethod
        def relu(x: Tensor) -> Tensor:
            out = Tensor(np.maximum(0.0, x.data), requires_grad=x.requires_grad, _children=(x,), _op="relu")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        x._add_grad(out.grad * (x.data > 0).astype(np.float32))
                out._backward = _backward
            return out

        @staticmethod
        def gelu(x: Tensor) -> Tensor:
            # Fast tanh approximation of GELU
            c = math.sqrt(2.0 / math.pi)
            u = c * (x.data + 0.044715 * (x.data ** 3))
            t = np.tanh(np.clip(u, -15.0, 15.0))
            out_data = 0.5 * x.data * (1.0 + t)
            out = Tensor(out_data, requires_grad=x.requires_grad, _children=(x,), _op="gelu")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        du = c * (1.0 + 3.0 * 0.044715 * (x.data ** 2))
                        grad_x = 0.5 * (1.0 + t) + 0.5 * x.data * (1.0 - t ** 2) * du
                        x._add_grad(out.grad * grad_x)
                out._backward = _backward
            return out

        @staticmethod
        def softmax(x: Tensor, dim: int = -1) -> Tensor:
            shifted = x.data - np.max(x.data, axis=dim, keepdims=True)
            exps = np.exp(np.clip(shifted, -40.0, 40.0))
            probs = exps / (np.sum(exps, axis=dim, keepdims=True) + 1e-12)
            out = Tensor(probs, requires_grad=x.requires_grad, _children=(x,), _op="softmax")
            if x.requires_grad:
                def _backward():
                    if out.grad is not None:
                        dot = np.sum(out.grad * probs, axis=dim, keepdims=True)
                        x._add_grad(probs * (out.grad - dot))
                out._backward = _backward
            return out

    class Parameter(Tensor):
        def __init__(self, data):
            super().__init__(data, requires_grad=True, dtype=np.float32)

    class Module:
        def __init__(self):
            self.training = True

        def parameters(self) -> List[Parameter]:
            params: List[Parameter] = []
            visited = set()

            def collect(obj):
                if id(obj) in visited:
                    return
                visited.add(id(obj))
                if isinstance(obj, Parameter):
                    params.append(obj)
                elif isinstance(obj, Module):
                    for val in obj.__dict__.values():
                        collect(val)
                elif isinstance(obj, (list, tuple)):
                    for item in obj:
                        collect(item)
                elif isinstance(obj, dict):
                    for item in obj.values():
                        collect(item)

            collect(self)
            return params

        def train(self, mode: bool = True):
            self.training = mode
            for p in self.__dict__.values():
                if isinstance(p, Module):
                    p.train(mode)
                elif isinstance(p, (list, tuple)):
                    for item in p:
                        if isinstance(item, Module):
                            item.train(mode)
            return self

        def eval(self):
            return self.train(False)

        def zero_grad(self):
            for p in self.parameters():
                p.zero_grad()

        def state_dict(self) -> Dict[str, np.ndarray]:
            return {f"param_{i}": p.data.copy() for i, p in enumerate(self.parameters())}

        def load_state_dict(self, state: Dict[str, np.ndarray]):
            for i, p in enumerate(self.parameters()):
                if f"param_{i}" in state:
                    p.data[...] = state[f"param_{i}"]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class Linear(Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True):
            super().__init__()
            bound = 1.0 / math.sqrt(max(1, in_features))
            self.weight = Parameter(np.random.uniform(-bound, bound, (out_features, in_features)).astype(np.float32))
            self.bias = Parameter(np.random.uniform(-bound, bound, (out_features,)).astype(np.float32)) if bias else None

        def forward(self, x: Tensor) -> Tensor:
            out = x @ self.weight.T
            if self.bias is not None:
                out = out + self.bias
            return out

    class LayerNorm(Module):
        def __init__(self, normalized_shape: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = Parameter(np.ones((normalized_shape,), dtype=np.float32))
            self.bias = Parameter(np.zeros((normalized_shape,), dtype=np.float32))

        def forward(self, x: Tensor) -> Tensor:
            mean = x.mean(dim=-1, keepdim=True)
            diff = x - mean
            var = (diff * diff).mean(dim=-1, keepdim=True)
            normed = diff / _TorchNamespace.sqrt(var + self.eps)
            return normed * self.weight + self.bias

    class ReLU(Module):
        def forward(self, x: Tensor) -> Tensor:
            return _FunctionalNamespace.relu(x)

    class GELU(Module):
        def forward(self, x: Tensor) -> Tensor:
            return _FunctionalNamespace.gelu(x)

    class Sequential(Module):
        def __init__(self, *layers: Module):
            super().__init__()
            self.layers = list(layers)

        def forward(self, x: Tensor) -> Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

    class ModuleList(Module):
        def __init__(self, modules: List[Module]):
            super().__init__()
            self.modules_list = list(modules)

        def __iter__(self):
            return iter(self.modules_list)

        def __len__(self):
            return len(self.modules_list)

        def __getitem__(self, idx: int) -> Module:
            return self.modules_list[idx]

    class AdamW:
        """Standard AdamW optimizer with decoupled weight decay and gradient clipping."""

        def __init__(
            self,
            params: List[Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-4,
        ):
            self.params = list(params)
            self.lr = lr
            self.beta1, self.beta2 = betas
            self.eps = eps
            self.weight_decay = weight_decay
            self.t = 0
            self.m = [np.zeros_like(p.data) for p in self.params]
            self.v = [np.zeros_like(p.data) for p in self.params]

        def zero_grad(self):
            for p in self.params:
                p.zero_grad()

        def step(self):
            self.t += 1
            b1_corr = 1.0 - (self.beta1 ** self.t)
            b2_corr = 1.0 - (self.beta2 ** self.t)
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                g = np.clip(p.grad, -5.0, 5.0)
                if self.weight_decay > 0:
                    p.data -= self.lr * self.weight_decay * p.data
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g * g)
                m_hat = self.m[i] / b1_corr
                v_hat = self.v[i] / b2_corr
                p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    class _NNNamespace:
        Module = Module
        Parameter = Parameter
        Linear = Linear
        LayerNorm = LayerNorm
        ReLU = ReLU
        GELU = GELU
        Sequential = Sequential
        ModuleList = ModuleList

    class _OptimNamespace:
        AdamW = AdamW

    torch = _TorchNamespace()  # type: ignore[assignment]
    nn = _NNNamespace()        # type: ignore[assignment]
    F = _FunctionalNamespace() # type: ignore[assignment]
    torch.nn = nn              # type: ignore[attr-defined]
    torch.optim = _OptimNamespace()  # type: ignore[attr-defined]


# ============================================================================
# 1. Step 3.1 & 3.2: Sparse Adjacency Mask & 8D Pairwise Cut-Alignment Geometry
# ============================================================================

def compute_sparse_masks_and_pairwise_geometry(
    boxes_global: "torch.Tensor",
    patch_ids: "torch.Tensor",
    is_boundary_risk: "torch.Tensor",
    max_neighbors: int = 16,
    gap_thresh_pixels: float = 16.0,
    giou_thresh: float = -0.5,
    use_iou1d: bool = True,
    eps: float = 1e-6,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """
    Step 3.1 & 3.2:
    Constructs:
      1. M_local in {0, 1}^{N x N}: Sparse local adjacency mask where shortest boundary
         distance < gap_thresh_pixels (16px) OR GIoU(b_i, b_j) > -0.5, capped at top K=16
         nearest neighbors per box (always retaining self loop i == j).
      2. M_stitch in {0, 1}^{N x N}: Cross-tile stitching mask (either-side boundary gate):
         M_stitch[i, j] = M_local[i, j] & (patch_id[i] != patch_id[j]) & (I_bound[i] | I_bound[j])
         (③放行“完整框<->单侧截断框”的互补正边：截断侧 flag=1 即可成边，不再要求两侧都贴边界)
      3. g_ij in R^{N x N x 8}: 8-dimensional Pairwise Geometric & Cut-Alignment vector:
         - [0:4] Standard relative geometry:
                 log(|x_i - x_j| / sqrt(w_i w_j)), log(|y_i - y_j| / sqrt(h_i h_j)),
                 log(w_j / w_i), log(h_j / h_i)
         - [4:6] Area overlap: Standard IoU(b_i, b_j) and GIoU(b_i, b_j)
         - [6:7] 1D Cut-Projection Overlap: max(IoU_1D(x), IoU_1D(y))
         - [7:8] Cross-intersection corner convergence distance:
                 min_{u, v in corners} ||c_{i,u} - c_{j,v}||_2 / sqrt(w_i h_i + w_j h_j)
    """
    n = boxes_global.shape[0]
    if n == 0:
        return (
            torch.zeros((0, 0), dtype=torch.bool),
            torch.zeros((0, 0), dtype=torch.bool),
            torch.zeros((0, 0, 8), dtype=torch.float32),
        )

    b_np = boxes_global.detach().numpy() if hasattr(boxes_global, "detach") else np.asarray(boxes_global)
    p_np = patch_ids.detach().numpy() if hasattr(patch_ids, "detach") else np.asarray(patch_ids)
    ib_np = is_boundary_risk.detach().numpy() if hasattr(is_boundary_risk, "detach") else np.asarray(is_boundary_risk)

    x1 = b_np[:, 0]
    y1 = b_np[:, 1]
    x2 = b_np[:, 2]
    y2 = b_np[:, 3]
    w = np.maximum(eps, x2 - x1)
    h = np.maximum(eps, y2 - y1)
    xc = x1 + 0.5 * w
    yc = y1 + 0.5 * h
    area = w * h

    # Pairwise intersections and enclosing boxes [N, N]
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih

    union = area[:, None] + area[None, :] - inter + eps
    iou = inter / union

    cx1 = np.minimum(x1[:, None], x1[None, :])
    cy1 = np.minimum(y1[:, None], y1[None, :])
    cx2 = np.maximum(x2[:, None], x2[None, :])
    cy2 = np.maximum(y2[:, None], y2[None, :])
    cw = np.maximum(eps, cx2 - cx1)
    ch = np.maximum(eps, cy2 - cy1)
    c_area = cw * ch + eps
    giou = iou - (c_area - union) / c_area

    # Shortest boundary gap distance between box i and box j
    gap_x = np.maximum(0.0, ix1 - ix2)
    gap_y = np.maximum(0.0, iy1 - iy2)
    boundary_gap = np.sqrt(gap_x ** 2 + gap_y ** 2)

    # Candidate local adjacency condition: shortest boundary gap < 16px OR GIoU > -0.5
    cand_local = (boundary_gap < gap_thresh_pixels) | (giou > giou_thresh)
    np.fill_diagonal(cand_local, True)

    # Retain at most K = 16 nearest neighbors per box (ranked by boundary_gap - 10 * giou)
    dist_score = boundary_gap - 10.0 * giou
    np.fill_diagonal(dist_score, -1e9)  # Ensure self is always rank 0
    dist_score_masked = np.where(cand_local, dist_score, 1e9)

    k_use = min(max_neighbors, n)
    topk_idx = np.argsort(dist_score_masked, axis=1)[:, :k_use]
    m_local_np = np.zeros((n, n), dtype=bool)
    row_idx = np.arange(n)[:, None]
    m_local_np[row_idx, topk_idx] = cand_local[row_idx, topk_idx]
    np.fill_diagonal(m_local_np, True)

    # Cross-slice stitching mask:
    # ③放宽门控：原 both_bound 要求两框都贴边界(I_bound[i] & I_bound[j])，会把“完整框<->截断框”
    # 这类真正需要拼接的正边(截断侧 flag=1、完整侧 flag=0)整体丢弃 -> target_edge=1 却 m_stitch=0，
    # 既不进边损失也不进包络。改为 either_bound(至少一侧贴边界)放行单侧截断的互补边，
    # m_local 的 top-K 近邻约束仍控制负边规模。
    # M_stitch[i, j] = M_local[i, j] & (patch_id[i] != patch_id[j]) & (I_bound[i] | I_bound[j])
    diff_patch = p_np[:, None] != p_np[None, :]
    either_bound = (ib_np[:, None] > 0.5) | (ib_np[None, :] > 0.5)
    m_stitch_np = m_local_np & diff_patch & either_bound

    # ------------------------------------------------------------------------
    # 3.2 Pairwise 8D Geometric & Cut-Alignment Vector g_ij
    # ------------------------------------------------------------------------
    # (1-4) Standard relative geometry (4D)
    denom_xy = np.sqrt(w[:, None] * w[None, :]) + eps
    denom_yx = np.sqrt(h[:, None] * h[None, :]) + eps
    g0 = np.log((np.abs(xc[:, None] - xc[None, :]) + 1.0) / denom_xy)
    g1 = np.log((np.abs(yc[:, None] - yc[None, :]) + 1.0) / denom_yx)
    g2 = np.log(w[None, :] / w[:, None])
    g3 = np.log(h[None, :] / h[:, None])

    # (5-6) Area overlap: Standard IoU and GIoU (2D)
    g4 = iou
    g5 = giou

    # (7) 1D Cut-Projection Overlap: max(IoU_1D(x), IoU_1D(y)) (1D)
    iou_1d_x = iw / cw
    iou_1d_y = ih / ch
    if use_iou1d:
        g6 = np.maximum(iou_1d_x, iou_1d_y)
    else:
        g6 = np.zeros_like(iou_1d_x)
        g0 = np.zeros_like(iou_1d_x)
        g1 = np.zeros_like(iou_1d_x)

    # (8) Cross-intersection corner convergence distance (1D)
    # min_{u, v in corners} ||c_{i,u} - c_{j,v}||_2 / sqrt(w_i h_i + w_j h_j)
    corners = np.stack([
        np.stack([x1, y1], axis=-1),
        np.stack([x1, y2], axis=-1),
        np.stack([x2, y1], axis=-1),
        np.stack([x2, y2], axis=-1),
    ], axis=1)  # [N, 4, 2]

    # Compute pairwise distances between 4 corners of i and 4 corners of j: [N, N, 4, 4]
    c_diff = corners[:, None, :, None, :] - corners[None, :, None, :, :]  # [N, N, 4, 4, 2]
    c_dist = np.sqrt(np.sum(c_diff ** 2, axis=-1))                       # [N, N, 4, 4]
    min_corner_dist = np.min(c_dist, axis=(-2, -1))                      # [N, N]
    norm_scale = np.sqrt(area[:, None] + area[None, :]) + eps
    g7 = min_corner_dist / norm_scale

    g_np = np.stack([g0, g1, g2, g3, g4, g5, g6, g7], axis=-1).astype(np.float32)
    g_np = np.clip(g_np, -10.0, 10.0)

    return (
        torch.tensor(m_local_np, dtype=torch.bool),
        torch.tensor(m_stitch_np, dtype=torch.bool),
        torch.tensor(g_np, dtype=torch.float32),
    )


# ============================================================================
# 2. Step 3.3: Boundary-Aware Local Relation Self-Attention Layer
# ============================================================================

class BALRAttentionLayer(nn.Module):
    """
    Single Layer of Boundary-Aware Local Relation Self-Attention (BALR-Attention):
        AttnLogits_{ij}^{(l)} = (Q_i^{(l)} (K_j^{(l)})^T) / sqrt(d_k) + MLP_geo(g_{ij})
    Positions where M_local[i, j] == 0 are masked with -inf (-1e9) prior to Softmax.
    """

    def __init__(self, d_model: int = 64, geo_dim: int = 8):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(float(d_model))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Two-layer MLP mapping 8D pairwise geometry g_ij to scalar attention bias B_ij
        self.mlp_geo = nn.Sequential(
            nn.Linear(geo_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h: "torch.Tensor",
        g_ij: "torch.Tensor",
        m_local: "torch.Tensor",
    ) -> "torch.Tensor":
        """
        Args:
            h:       [N, d_model] node embeddings
            g_ij:    [N, N, 8] pairwise geometric & cut-alignment features
            m_local: [N, N] boolean local adjacency mask
        Returns:
            h_next:  [N, d_model] updated node embeddings
        """
        if h.shape[0] == 0:
            return h

        q = self.q_proj(h)  # [N, d]
        k = self.k_proj(h)  # [N, d]
        v = self.v_proj(h)  # [N, d]

        # Pairwise geometric bias B_ij: [N, N, 1] -> [N, N]
        b_ij = self.mlp_geo(g_ij).squeeze(-1)

        # AttnLogits_ij = Q_i K_j^T / sqrt(d_k) + MLP_geo(g_ij)
        attn_logits = (q @ k.T) * self.scale + b_ij

        # Mask out non-neighbors (M_local[i, j] == 0) with -1e9 (-inf)
        invalid_mask = ~m_local
        attn_logits = attn_logits.masked_fill(invalid_mask, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)  # [N, N]
        agg = attn_weights @ v                         # [N, d]

        h_mid = self.norm1(h + self.out_proj(agg))
        h_out = self.norm2(h_mid + self.ffn(h_mid))
        return h_out


# ============================================================================
# 3. Step 3.4: TileStitchingNet with Soft Bounding Envelope Operator
# ============================================================================

class TileStitchingNet(nn.Module):
    """
    Complete Boundary-Aware Local Relation Attention Post-Processing Network.
    Replaces handcrafted NMS / NMM / WBF with:
      1. 2-Layer BALR-Attention (connecting 1-hop adjacent fragments and 2-hop
         diagonal/cross-4-tile fragments).
      2. Explicit Soft Bounding Envelope Operator (absorbs multi-tile fragments into
         a differentiable soft union bounding box b_i^env).
      3. Dual-Branch Prediction Heads (Box Refinement Head + Confidence Rescoring Head).

    Supports Ablation switches:
      - num_layers: 2 (default) vs 1 (ablation)
      - use_cutline_dist: True (default) vs False (ablation: zeroes out 4D cut-line distances & I_bound)
      - use_iou1d: True (default) vs False (ablation: zeroes out 1D cut-projection overlap)
      - use_soft_envelope: True (default) vs False (ablation: bypasses Soft Bounding Envelope)
    """

    def __init__(
        self,
        in_dim: int = 12,
        roi_dim: int = 0,
        d_model: int = 64,
        num_layers: int = 2,
        max_neighbors: int = 16,
        gap_thresh_pixels: float = 16.0,
        giou_thresh: float = -0.5,
        use_cutline_dist: bool = True,
        use_iou1d: bool = True,
        use_soft_envelope: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim + roi_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_neighbors = max_neighbors
        self.gap_thresh_pixels = gap_thresh_pixels
        self.giou_thresh = giou_thresh
        self.use_cutline_dist = use_cutline_dist
        self.use_iou1d = use_iou1d
        self.use_soft_envelope = use_soft_envelope

        # Node input embedding projection (patch_id column 6 is zeroed before linear projection)
        self.input_proj = nn.Sequential(
            nn.Linear(self.in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Stack of BALR-Attention layers (default: 2 layers for 2-hop cross-4-tile message passing)
        self.attn_layers = nn.ModuleList([
            BALRAttentionLayer(d_model=d_model, geo_dim=8)
            for _ in range(num_layers)
        ])

        # Affinity head for same-source stitching probability A_ij:
        # A_ij = Sigmoid(Q_i K_j^T / sqrt(d) + MLP_edge(g_ij)) * M_stitch[i, j]
        self.edge_q = nn.Linear(d_model, d_model)
        self.edge_k = nn.Linear(d_model, d_model)
        self.mlp_edge = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # Context encoder for soft envelope expansion features (helps rescoring & refinement)
        # Encodes [env_gain_w, env_gain_h, max_stitch_prob, sum_stitch_prob, score_diff_max]
        self.env_context_proj = nn.Sequential(
            nn.Linear(d_model + 6, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Branch 1: Box Refinement Head predicting residual offsets (dx1, dy1, dx2, dy2)
        self.box_refine_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 4),
        )
        # Initialize last layer of refinement head near zero so initial prediction starts at b_i^env
        if TORCH_BACKEND == "pytorch":
            nn.init.zeros_(self.box_refine_head[-1].weight)
            nn.init.zeros_(self.box_refine_head[-1].bias)
        else:
            self.box_refine_head.layers[-1].weight.data.fill(0.0)
            self.box_refine_head.layers[-1].bias.data.fill(0.0)

        # Branch 2: Confidence Rescoring Head predicting retention score s_i in (0, 1)
        self.rescore_head = nn.Sequential(
            nn.Linear(d_model + 6, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.rescore_direct = nn.Linear(6, 1)

    @staticmethod
    def compute_soft_bounding_envelope(
        boxes_global: "torch.Tensor",
        affinity_matrix: "torch.Tensor",
        m_stitch: "torch.Tensor",
        num_layers: int = 2,
    ) -> "torch.Tensor":
        """
        Step 3.4 Step A: Explicit Soft Bounding Envelope Operator.
        For each box i, absorbs coordinates of neighboring same-source fragments j in N(i):
            x_{1,i}^{env} = min_{j in N(i)} (x_{1,i} + A_{ij} * (x_{1,j} - x_{1,i}))
            y_{1,i}^{env} = min_{j in N(i)} (y_{1,i} + A_{ij} * (y_{1,j} - y_{1,i}))
            x_{2,i}^{env} = max_{j in N(i)} (x_{2,i} + A_{ij} * (x_{2,j} - x_{2,i}))
            y_{2,i}^{env} = max_{j in N(i)} (y_{2,i} + A_{ij} * (y_{2,j} - y_{2,i}))
        When num_layers >= 2, 2-Hop Message Passing connects diagonal/chain fragments across 3~4 tiles;
        when num_layers == 1, only direct 1-hop cut-aligned fragments (IoU_1D > 0) are connected.
        """
        n = boxes_global.shape[0]
        if n == 0:
            return boxes_global

        a_eff = affinity_matrix * m_stitch.float()

        x1 = boxes_global[:, 0]  # [N]
        y1 = boxes_global[:, 1]
        x2 = boxes_global[:, 2]
        y2 = boxes_global[:, 3]

        cand_x1 = x1.unsqueeze(1) + a_eff * (x1.unsqueeze(0) - x1.unsqueeze(1))
        cand_y1 = y1.unsqueeze(1) + a_eff * (y1.unsqueeze(0) - y1.unsqueeze(1))
        cand_x2 = x2.unsqueeze(1) + a_eff * (x2.unsqueeze(0) - x2.unsqueeze(1))
        cand_y2 = y2.unsqueeze(1) + a_eff * (y2.unsqueeze(0) - y2.unsqueeze(1))

        env_x1, _ = cand_x1.min(dim=1)
        env_y1, _ = cand_y1.min(dim=1)
        env_x2, _ = cand_x2.max(dim=1)
        env_y2, _ = cand_y2.max(dim=1)

        return torch.stack([env_x1, env_y1, env_x2, env_y2], dim=-1)

    def forward(
        self,
        node_features: "torch.Tensor",
        boxes_global: "torch.Tensor",
        patch_ids: Optional["torch.Tensor"] = None,
        is_boundary_risk: Optional["torch.Tensor"] = None,
    ) -> Dict[str, "torch.Tensor"]:
        """
        Forward pass of TileStitchingNet for a single high-resolution image's N candidate boxes.
        """
        n = boxes_global.shape[0]
        if n == 0:
            empty_mat = torch.zeros((0, 0), dtype=torch.float32)
            empty_bool = torch.zeros((0, 0), dtype=torch.bool)
            return {
                "pred_boxes": boxes_global,
                "env_boxes": boxes_global,
                "pred_scores": torch.zeros((0,), dtype=torch.float32),
                "rescore_logits": torch.zeros((0,), dtype=torch.float32),
                "edge_probs": empty_mat,
                "edge_logits": empty_mat,
                "m_local": empty_bool,
                "m_stitch": empty_bool,
            }

        if patch_ids is None:
            patch_ids = node_features[:, 6].long()
        if is_boundary_risk is None:
            if self.use_cutline_dist:
                is_boundary_risk = node_features[:, 11]
            else:
                # Without cut-line distance encoding, I_bound prior is unavailable
                is_boundary_risk = torch.ones((n,), dtype=torch.float32)

        feat_mask = np.ones((1, node_features.shape[1]), dtype=np.float32)
        feat_mask[0, 6] = 0.0
        if not self.use_cutline_dist:
            feat_mask[0, 7:12] = 0.0
        clean_features = node_features * torch.tensor(feat_mask, dtype=torch.float32)

        # Step 3.1 & 3.2: Sparse masks and 8D pairwise cut-alignment geometry
        m_local, m_stitch, g_ij = compute_sparse_masks_and_pairwise_geometry(
            boxes_global=boxes_global,
            patch_ids=patch_ids,
            is_boundary_risk=is_boundary_risk,
            max_neighbors=self.max_neighbors,
            gap_thresh_pixels=self.gap_thresh_pixels,
            giou_thresh=self.giou_thresh,
            use_iou1d=self.use_iou1d,
        )

        # For 1-Layer Attention, only 1-hop direct cut-face neighbors (IoU_1D > 0.05) can be linked;
        # For 2-Layer Attention, 2-hop message passing bridges diagonal/3~4-tile fragments
        if self.num_layers == 1:
            direct_cut_mask = (g_ij[:, :, 6] > 0.05)
            m_stitch_eff = m_stitch & direct_cut_mask
        else:
            m_stitch_eff = m_stitch

        # Node Embedding + BALR-Attention Layers
        h = self.input_proj(clean_features)
        for layer in self.attn_layers:
            h = layer(h, g_ij, m_local)

        # Step 3.4: Pairwise Same-Source Stitching Affinity Head
        q_edge = self.edge_q(h)
        k_edge = self.edge_k(h)
        scale = 1.0 / math.sqrt(float(self.d_model))
        edge_bias = self.mlp_edge(g_ij).squeeze(-1)

        # Incorporate explicit cut-alignment & 2-hop structural prior into edge logits
        iou1d_feat = g_ij[:, :, 6]  # max(IoU_1D(x), IoU_1D(y))
        corner_feat = g_ij[:, :, 7] # normalized min corner distance
        if self.use_iou1d and self.use_cutline_dist:
            # 1-hop cut-face alignment or 2-hop corner convergence (only active when num_layers >= 2)
            hop2_align = (corner_feat < 0.12).float() if self.num_layers >= 2 else torch.zeros(corner_feat.shape, dtype=torch.float32)
            align_prior = 3.0 * (iou1d_feat + 0.8 * hop2_align) - 1.5
        else:
            align_prior = 0.0

        edge_logits = (q_edge @ k_edge.T) * scale + edge_bias + align_prior
        edge_logits = 0.5 * (edge_logits + edge_logits.T)

        edge_probs_raw = torch.sigmoid(edge_logits)
        edge_probs = edge_probs_raw * m_stitch_eff.float()

        # Step 3.4 Step A: Soft Bounding Envelope Box b_i^{env}
        if self.use_soft_envelope:
            env_boxes = self.compute_soft_bounding_envelope(
                boxes_global, edge_probs_raw, m_stitch_eff, num_layers=self.num_layers
            )
        else:
            env_boxes = boxes_global

        orig_w = torch.clamp(boxes_global[:, 2] - boxes_global[:, 0], min=1.0)
        orig_h = torch.clamp(boxes_global[:, 3] - boxes_global[:, 1], min=1.0)
        env_w = torch.clamp(env_boxes[:, 2] - env_boxes[:, 0], min=1.0)
        env_h = torch.clamp(env_boxes[:, 3] - env_boxes[:, 1], min=1.0)

        gain_w = torch.log(env_w / orig_w)
        gain_h = torch.log(env_h / orig_h)
        max_stitch_prob, _ = edge_probs.max(dim=1)

        # Compute containment indicator (whether box i is geometrically nested inside a larger neighbor j)
        # so the rescoring head can learn either to preserve nested targets (Cropped-GT IoU) or suppress them (IoS)
        b_np = boxes_global.detach().numpy()
        lt = np.maximum(b_np[:, None, :2], b_np[None, :, :2])
        rb = np.minimum(b_np[:, None, 2:], b_np[None, :, 2:])
        wh = np.maximum(0.0, rb - lt)
        inter = wh[:, :, 0] * wh[:, :, 1]
        area_np = (orig_w.detach().numpy() * orig_h.detach().numpy())
        ioa_self = inter / (area_np[:, None] + 1e-6)  # fraction of box i inside box j
        larger_j = (area_np[None, :] > area_np[:, None] * 1.8) & (g_ij[:, :, 4].detach().numpy() < 0.45)
        np.fill_diagonal(larger_j, False)
        is_nested_inside = torch.tensor(np.max(np.where(larger_j, ioa_self, 0.0), axis=1), dtype=torch.float32)

        raw_scores = node_features[:, 4]
        quality_proxy = raw_scores * (orig_w * orig_h)
        q_diff = (quality_proxy.unsqueeze(0) - quality_proxy.unsqueeze(1)) / (quality_proxy.unsqueeze(1) + 1.0)
        iou_mat = g_ij[:, :, 4]
        same_obj_weight = torch.maximum(edge_probs, (iou_mat > 0.5).float() * m_local.float() * (1.0 - torch.eye(n)))
        neighbor_dominance, _ = (q_diff * same_obj_weight).max(dim=1)

        env_ctx = torch.stack([
            raw_scores,
            gain_w,
            gain_h,
            max_stitch_prob,
            is_nested_inside,
            neighbor_dominance,
        ], dim=-1)

        h_final = self.env_context_proj(torch.cat([h, env_ctx], dim=-1))

        # Step 3.4 Step B: Dual-Branch Prediction Heads
        raw_deltas = self.box_refine_head(h_final)  # [N, 4]
        scale_wh = torch.stack([env_w, env_h, env_w, env_h], dim=-1)
        deltas = raw_deltas.clamp(min=-0.35, max=0.35) * scale_wh
        pred_boxes = env_boxes + deltas

        rescore_logits = (
            self.rescore_head(torch.cat([h_final, env_ctx], dim=-1)).squeeze(-1)
            + self.rescore_direct(env_ctx).squeeze(-1)
        )
        pred_scores = torch.sigmoid(rescore_logits)

        return {
            "raw_deltas": raw_deltas,
            "pred_boxes": pred_boxes,
            "env_boxes": env_boxes,
            "pred_scores": pred_scores,
            "rescore_logits": rescore_logits,
            "edge_probs": edge_probs,
            "edge_logits": edge_logits,
            "m_local": m_local,
            "m_stitch": m_stitch_eff,
        }