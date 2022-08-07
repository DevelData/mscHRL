import gin
import torch
import numpy as np

# Copied from https://github.com/jesbu1/alf/blob/def59fe39bdbca70a6c80e9b8f2c7c785cb59ea7/alf/tensor_specs.py

@gin.configurable
class TensorSpec(object):
    """Describes a torch.Tensor.
    A TensorSpec allows an API to describe the Tensors that it accepts or
    returns, before that Tensor exists. This allows dynamic and flexible graph
    construction and configuration.
    """

    __slots__ = ["_shape", "_dtype"]

    def __init__(self, shape, dtype=torch.float32):
        """Creates a TensorSpec.
        Args:
            shape (tuple[int]): The shape of the tensor.
            dtype (str or torch.dtype): The type of the tensor values,
                e.g., "int32" or torch.int32
        """
        self._shape = tuple(shape)
        if isinstance(dtype, str):
            self._dtype = getattr(torch, dtype)
        else:
            assert isinstance(dtype, torch.dtype)
            self._dtype = dtype

    @classmethod
    def from_spec(cls, spec):
        assert isinstance(spec, TensorSpec)
        return cls(spec.shape, spec.dtype)

    @classmethod
    def from_tensor(cls, tensor, from_dim=0):
        """Create TensorSpec from tensor.
        Args:
            tensor (Tensor): tensor from which the spec is extracted
            from_dim (int): use tensor.shape[from_dim:] as shape
        Returns:
            TensorSpec
        """
        assert isinstance(tensor, torch.Tensor)
        return TensorSpec(tensor.shape[from_dim:], tensor.dtype)

    @classmethod
    def from_array(cls, array, from_dim=0):
        """Create TensorSpec from numpy array.
        Args:
            array (np.ndarray|np.number): array from which the spec is extracted
            from_dim (int): use ``array.shape[from_dim:]`` as shape
        Returns:
            TensorSpec
        """
        assert isinstance(array, (np.ndarray, np.number))
        return TensorSpec(array.shape[from_dim:], str(array.dtype))

    @classmethod
    def is_bounded(cls):
        #del cls
        return False

    @property
    def shape(self):
        """Returns the `TensorShape` that represents the shape of the tensor."""
        return self._shape

    @property
    def ndim(self):
        """Return the rank of the tensor."""
        return len(self._shape)

    @property
    def dtype(self):
        """Returns the `dtype` of elements in the tensor."""
        return self._dtype

    @property
    def is_continuous(self):
        """Whether spec is continuous or not."""
        # Modified from original
        # Deleted is_discrete method
        return self.dtype.is_floating_point

    def __repr__(self):
        return "TensorSpec(shape={}, dtype={})".format(self.shape,
                                                       repr(self.dtype))

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    def __ne__(self, other):
        return not self == other

    def __reduce__(self):
        return TensorSpec, (self._shape, self._dtype)

    def constant(self, value, outer_dims=None):
        """Create a constant tensor from the spec.
        Args:
            value : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        value = torch.as_tensor(value).to(self._dtype)
        assert len(value.size()) == 0, "The input value must be a scalar!"
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.ones(size=shape, dtype=self._dtype) * value

    def zeros(self, outer_dims=None):
        """Create a zero tensor from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return self.constant(0, outer_dims)

    def numpy_constant(self, value, outer_dims=None):
        """Create a constant np.ndarray from the spec.
        Args:
            value (Number) : a scalar
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return np.ones(shape, dtype=torch_dtype_to_str(self._dtype)) * value

    def numpy_zeros(self, outer_dims=None):
        """Create a zero numpy.ndarray from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            np.ndarray: an array of ``self._dtype``.
        """
        return self.numpy_constant(0, outer_dims)

    def ones(self, outer_dims=None):
        """Create an all-one tensor from the spec.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        return self.constant(1, outer_dims)

    def randn(self, outer_dims=None):
        """Create a tensor filled with random numbers from a std normal dist.
        Args:
            outer_dims (tuple[int]): an optional list of integers specifying outer
                dimensions to add to the spec shape before sampling.
        Returns:
            tensor (torch.Tensor): a tensor of ``self._dtype``.
        """
        shape = self._shape
        if outer_dims is not None:
            shape = tuple(outer_dims) + shape
        return torch.randn(*shape, dtype=self._dtype)