# NumPy API Reference — selected functions

Source: NumPy 2.3.5 installed Python API docstrings. Reorganized as Markdown; examples wrapped in code fences. See CORPUS_LICENSE.txt.

## numpy.reshape

Gives a new shape to an array without changing its data.

Parameters
----------
a : array_like
    Array to be reshaped.
shape : int or tuple of ints
    The new shape should be compatible with the original shape. If
    an integer, then the result will be a 1-D array of that length.
    One shape dimension can be -1. In this case, the value is
    inferred from the length of the array and remaining dimensions.
order : {'C', 'F', 'A'}, optional
    Read the elements of ``a`` using this index order, and place the
    elements into the reshaped array using this index order. 'C'
    means to read / write the elements using C-like index order,
    with the last axis index changing fastest, back to the first
    axis index changing slowest. 'F' means to read / write the
    elements using Fortran-like index order, with the first index
    changing fastest, and the last index changing slowest. Note that
    the 'C' and 'F' options take no account of the memory layout of
    the underlying array, and only refer to the order of indexing.
    'A' means to read / write the elements in Fortran-like index
    order if ``a`` is Fortran *contiguous* in memory, C-like order
    otherwise.
newshape : int or tuple of ints
    .. deprecated:: 2.1
        Replaced by ``shape`` argument. Retained for backward
        compatibility.
copy : bool, optional
    If ``True``, then the array data is copied. If ``None``, a copy will
    only be made if it's required by ``order``. For ``False`` it raises
    a ``ValueError`` if a copy cannot be avoided. Default: ``None``.

Returns
-------
reshaped_array : ndarray
    This will be a new view object if possible; otherwise, it will
    be a copy.  Note there is no guarantee of the *memory layout* (C- or
    Fortran- contiguous) of the returned array.

See Also
--------
ndarray.reshape : Equivalent method.

Notes
-----
It is not always possible to change the shape of an array without copying
the data.

The ``order`` keyword gives the index ordering both for *fetching*
the values from ``a``, and then *placing* the values into the output
array. For example, let's say you have an array:

>>> a = np.arange(6).reshape((3, 2))
>>> a
array([[0, 1],
       [2, 3],
       [4, 5]])

You can think of reshaping as first raveling the array (using the given
index order), then inserting the elements from the raveled array into the
new array using the same kind of index ordering as was used for the
raveling.

>>> np.reshape(a, (2, 3)) # C-like index ordering
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
array([[0, 1, 2],
       [3, 4, 5]])
>>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
array([[0, 4, 3],
       [2, 1, 5]])
>>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
array([[0, 4, 3],
       [2, 1, 5]])

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1,2,3], [4,5,6]])
>>> np.reshape(a, 6)
array([1, 2, 3, 4, 5, 6])
>>> np.reshape(a, 6, order='F')
array([1, 4, 2, 5, 3, 6])

>>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
array([[1, 2],
       [3, 4],
       [5, 6]])
````

## numpy.transpose

Returns an array with axes transposed.

For a 1-D array, this returns an unchanged view of the original array, as a
transposed vector is simply the same vector.
To convert a 1-D array into a 2-D column vector, an additional dimension
must be added, e.g., ``np.atleast_2d(a).T`` achieves this, as does
``a[:, np.newaxis]``.
For a 2-D array, this is the standard matrix transpose.
For an n-D array, if axes are given, their order indicates how the
axes are permuted (see Examples). If axes are not provided, then
``transpose(a).shape == a.shape[::-1]``.

Parameters
----------
a : array_like
    Input array.
axes : tuple or list of ints, optional
    If specified, it must be a tuple or list which contains a permutation
    of [0, 1, ..., N-1] where N is the number of axes of `a`. Negative
    indices can also be used to specify axes. The i-th axis of the returned
    array will correspond to the axis numbered ``axes[i]`` of the input.
    If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses
    the order of the axes.

Returns
-------
p : ndarray
    `a` with its axes permuted. A view is returned whenever possible.

See Also
--------
ndarray.transpose : Equivalent method.
moveaxis : Move axes of an array to new positions.
argsort : Return the indices that would sort an array.

Notes
-----
Use ``transpose(a, argsort(axes))`` to invert the transposition of tensors
when using the `axes` keyword argument.

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> a
array([[1, 2],
       [3, 4]])
>>> np.transpose(a)
array([[1, 3],
       [2, 4]])

>>> a = np.array([1, 2, 3, 4])
>>> a
array([1, 2, 3, 4])
>>> np.transpose(a)
array([1, 2, 3, 4])

>>> a = np.ones((1, 2, 3))
>>> np.transpose(a, (1, 0, 2)).shape
(2, 1, 3)

>>> a = np.ones((2, 3, 4, 5))
>>> np.transpose(a).shape
(5, 4, 3, 2)

>>> a = np.arange(3*4*5).reshape((3, 4, 5))
>>> np.transpose(a, (-1, 0, -2)).shape
(5, 3, 4)
````

## numpy.broadcast_to

Broadcast an array to a new shape.

Parameters
----------
array : array_like
    The array to broadcast.
shape : tuple or int
    The shape of the desired array. A single integer ``i`` is interpreted
    as ``(i,)``.
subok : bool, optional
    If True, then sub-classes will be passed-through, otherwise
    the returned array will be forced to be a base-class array (default).

Returns
-------
broadcast : array
    A readonly view on the original array with the given shape. It is
    typically not contiguous. Furthermore, more than one element of a
    broadcasted array may refer to a single memory location.

Raises
------
ValueError
    If the array is not compatible with the new shape according to NumPy's
    broadcasting rules.

See Also
--------
broadcast
broadcast_arrays
broadcast_shapes

Examples
--------

````python
>>> import numpy as np
>>> x = np.array([1, 2, 3])
>>> np.broadcast_to(x, (3, 3))
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]])
````

## numpy.concatenate

concatenate(
    (a1, a2, ...),
    axis=0,
    out=None,
    dtype=None,
    casting="same_kind"
)

Join a sequence of arrays along an existing axis.

Parameters
----------
a1, a2, ... : sequence of array_like
    The arrays must have the same shape, except in the dimension
    corresponding to `axis` (the first, by default).
axis : int, optional
    The axis along which the arrays will be joined.  If axis is None,
    arrays are flattened before use.  Default is 0.
out : ndarray, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what concatenate would have returned if no
    out argument were specified.
dtype : str or dtype
    If provided, the destination array will have this dtype. Cannot be
    provided together with `out`.

    .. versionadded:: 1.20.0

casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'same_kind'.
    For a description of the options, please see :term:`casting`.

    .. versionadded:: 1.20.0

Returns
-------
res : ndarray
    The concatenated array.

See Also
--------
ma.concatenate : Concatenate function that preserves input masks.
array_split : Split an array into multiple sub-arrays of equal or
              near-equal size.
split : Split array into a list of multiple sub-arrays of equal size.
hsplit : Split array into multiple sub-arrays horizontally (column wise).
vsplit : Split array into multiple sub-arrays vertically (row wise).
dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
stack : Stack a sequence of arrays along a new axis.
block : Assemble arrays from blocks.
hstack : Stack arrays in sequence horizontally (column wise).
vstack : Stack arrays in sequence vertically (row wise).
dstack : Stack arrays in sequence depth wise (along third dimension).
column_stack : Stack 1-D arrays as columns into a 2-D array.

Notes
-----
When one or more of the arrays to be concatenated is a MaskedArray,
this function will return a MaskedArray object instead of an ndarray,
but the input masks are *not* preserved. In cases where a MaskedArray
is expected as input, use the ma.concatenate function from the masked
array module instead.

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])

This function will not preserve masking of MaskedArray inputs.

>>> a = np.ma.arange(3)
>>> a[1] = np.ma.masked
>>> b = np.arange(2, 5)
>>> a
masked_array(data=[0, --, 2],
             mask=[False,  True, False],
       fill_value=999999)
>>> b
array([2, 3, 4])
>>> np.concatenate([a, b])
masked_array(data=[0, 1, 2, 2, 3, 4],
             mask=False,
       fill_value=999999)
>>> np.ma.concatenate([a, b])
masked_array(data=[0, --, 2, 2, 3, 4],
             mask=[False,  True, False, False, False, False],
       fill_value=999999)
````

## numpy.stack

Join a sequence of arrays along a new axis.

The ``axis`` parameter specifies the index of the new axis in the
dimensions of the result. For example, if ``axis=0`` it will be the first
dimension and if ``axis=-1`` it will be the last dimension.

Parameters
----------
arrays : sequence of ndarrays
    Each array must have the same shape. In the case of a single ndarray
    array_like input, it will be treated as a sequence of arrays; i.e.,
    each element along the zeroth axis is treated as a separate array.

axis : int, optional
    The axis in the result array along which the input arrays are stacked.

out : ndarray, optional
    If provided, the destination to place the result. The shape must be
    correct, matching that of what stack would have returned if no
    out argument were specified.

dtype : str or dtype
    If provided, the destination array will have this dtype. Cannot be
    provided together with `out`.

    .. versionadded:: 1.24

casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur. Defaults to 'same_kind'.

    .. versionadded:: 1.24


Returns
-------
stacked : ndarray
    The stacked array has one more dimension than the input arrays.

See Also
--------
concatenate : Join a sequence of arrays along an existing axis.
block : Assemble an nd-array from nested lists of blocks.
split : Split array into a list of multiple sub-arrays of equal size.
unstack : Split an array into a tuple of sub-arrays along an axis.

Examples
--------

````python
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> arrays = [rng.normal(size=(3,4)) for _ in range(10)]
>>> np.stack(arrays, axis=0).shape
(10, 3, 4)

>>> np.stack(arrays, axis=1).shape
(3, 10, 4)

>>> np.stack(arrays, axis=2).shape
(3, 4, 10)

>>> a = np.array([1, 2, 3])
>>> b = np.array([4, 5, 6])
>>> np.stack((a, b))
array([[1, 2, 3],
       [4, 5, 6]])

>>> np.stack((a, b), axis=-1)
array([[1, 4],
       [2, 5],
       [3, 6]])
````

## numpy.split

Split an array into multiple sub-arrays as views into `ary`.

Parameters
----------
ary : ndarray
    Array to be divided into sub-arrays.
indices_or_sections : int or 1-D array
    If `indices_or_sections` is an integer, N, the array will be divided
    into N equal arrays along `axis`.  If such a split is not possible,
    an error is raised.

    If `indices_or_sections` is a 1-D array of sorted integers, the entries
    indicate where along `axis` the array is split.  For example,
    ``[2, 3]`` would, for ``axis=0``, result in

    - ary[:2]
    - ary[2:3]
    - ary[3:]

    If an index exceeds the dimension of the array along `axis`,
    an empty sub-array is returned correspondingly.
axis : int, optional
    The axis along which to split, default is 0.

Returns
-------
sub-arrays : list of ndarrays
    A list of sub-arrays as views into `ary`.

Raises
------
ValueError
    If `indices_or_sections` is given as an integer, but
    a split does not result in equal division.

See Also
--------
array_split : Split an array into multiple sub-arrays of equal or
              near-equal size.  Does not raise an exception if
              an equal division cannot be made.
hsplit : Split array into multiple sub-arrays horizontally (column-wise).
vsplit : Split array into multiple sub-arrays vertically (row wise).
dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
concatenate : Join a sequence of arrays along an existing axis.
stack : Join a sequence of arrays along a new axis.
hstack : Stack arrays in sequence horizontally (column wise).
vstack : Stack arrays in sequence vertically (row wise).
dstack : Stack arrays in sequence depth wise (along third dimension).

Examples
--------

````python
>>> import numpy as np
>>> x = np.arange(9.0)
>>> np.split(x, 3)
[array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.,  8.])]

>>> x = np.arange(8.0)
>>> np.split(x, [3, 5, 6, 10])
[array([0.,  1.,  2.]),
 array([3.,  4.]),
 array([5.]),
 array([6.,  7.]),
 array([], dtype=float64)]
````

## numpy.where

where(condition, [x, y], /)

Return elements chosen from `x` or `y` depending on `condition`.

.. note::
    When only `condition` is provided, this function is a shorthand for
    ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
    preferred, as it behaves correctly for subclasses. The rest of this
    documentation covers only the case where all three arguments are
    provided.

Parameters
----------
condition : array_like, bool
    Where True, yield `x`, otherwise yield `y`.
x, y : array_like
    Values from which to choose. `x`, `y` and `condition` need to be
    broadcastable to some shape.

Returns
-------
out : ndarray
    An array with elements from `x` where `condition` is True, and elements
    from `y` elsewhere.

See Also
--------
choose
nonzero : The function that is called when x and y are omitted

Notes
-----
If all the arrays are 1-D, `where` is equivalent to::

    [xv if c else yv
     for c, xv, yv in zip(condition, x, y)]

Examples
--------

````python
>>> import numpy as np
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.where(a < 5, a, 10*a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

This can be used on multidimensional arrays too:

>>> np.where([[True, False], [True, True]],
...          [[1, 2], [3, 4]],
...          [[9, 8], [7, 6]])
array([[1, 8],
       [3, 4]])

The shapes of x, y, and the condition are broadcast together:

>>> x, y = np.ogrid[:3, :4]
>>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
array([[10,  0,  0,  0],
       [10, 11,  1,  1],
       [10, 11, 12,  2]])

>>> a = np.array([[0, 1, 2],
...               [0, 2, 4],
...               [0, 3, 6]])
>>> np.where(a < 4, a, -1)  # -1 is broadcast
array([[ 0,  1,  2],
       [ 0,  2, -1],
       [ 0,  3, -1]])
````

## numpy.nonzero

Return the indices of the elements that are non-zero.

Returns a tuple of arrays, one for each dimension of `a`,
containing the indices of the non-zero elements in that
dimension. The values in `a` are always tested and returned in
row-major, C-style order.

To group the indices by element, rather than dimension, use `argwhere`,
which returns a row for each non-zero element.

.. note::

   When called on a zero-d array or scalar, ``nonzero(a)`` is treated
   as ``nonzero(atleast_1d(a))``.

   .. deprecated:: 1.17.0

      Use `atleast_1d` explicitly if this behavior is deliberate.

Parameters
----------
a : array_like
    Input array.

Returns
-------
tuple_of_arrays : tuple
    Indices of elements that are non-zero.

See Also
--------
flatnonzero :
    Return indices that are non-zero in the flattened version of the input
    array.
ndarray.nonzero :
    Equivalent ndarray method.
count_nonzero :
    Counts the number of non-zero elements in the input array.

Notes
-----
While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
will correctly handle 0-d arrays.

Examples
--------

````python
>>> import numpy as np
>>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
>>> x
array([[3, 0, 0],
       [0, 4, 0],
       [5, 6, 0]])
>>> np.nonzero(x)
(array([0, 1, 2, 2]), array([0, 1, 0, 1]))

>>> x[np.nonzero(x)]
array([3, 4, 5, 6])
>>> np.transpose(np.nonzero(x))
array([[0, 0],
       [1, 1],
       [2, 0],
       [2, 1]])

A common use for ``nonzero`` is to find the indices of an array, where
a condition is True.  Given an array `a`, the condition `a` > 3 is a
boolean array and since False is interpreted as 0, np.nonzero(a > 3)
yields the indices of the `a` where the condition is true.

>>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
>>> a > 3
array([[False, False, False],
       [ True,  True,  True],
       [ True,  True,  True]])
>>> np.nonzero(a > 3)
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

Using this result to index `a` is equivalent to using the mask directly:

>>> a[np.nonzero(a > 3)]
array([4, 5, 6, 7, 8, 9])
>>> a[a > 3]  # prefer this spelling
array([4, 5, 6, 7, 8, 9])

``nonzero`` can also be called as a method of the array.

>>> (a > 3).nonzero()
(array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
````

## numpy.argmax

Returns the indices of the maximum values along an axis.

Parameters
----------
a : array_like
    Input array.
axis : int, optional
    By default, the index is into the flattened array, otherwise
    along the specified axis.
out : array, optional
    If provided, the result will be inserted into this array. It should
    be of the appropriate shape and dtype.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the array.

    .. versionadded:: 1.22.0

Returns
-------
index_array : ndarray of ints
    Array of indices into the array. It has the same shape as ``a.shape``
    with the dimension along `axis` removed. If `keepdims` is set to True,
    then the size of `axis` will be 1 with the resulting array having same
    shape as ``a.shape``.

See Also
--------
ndarray.argmax, argmin
amax : The maximum value along a given axis.
unravel_index : Convert a flat index into an index tuple.
take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                  from argmax to an array as if by calling max.

Notes
-----
In case of multiple occurrences of the maximum values, the indices
corresponding to the first occurrence are returned.

Examples
--------

````python
>>> import numpy as np
>>> a = np.arange(6).reshape(2,3) + 10
>>> a
array([[10, 11, 12],
       [13, 14, 15]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])

Indexes of the maximal elements of a N-dimensional array:

>>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
>>> ind
(1, 2)
>>> a[ind]
15

>>> b = np.arange(6)
>>> b[1] = 5
>>> b
array([0, 5, 2, 3, 4, 5])
>>> np.argmax(b)  # Only the first occurrence is returned.
1

>>> x = np.array([[4,2,3], [1,0,3]])
>>> index_array = np.argmax(x, axis=-1)
>>> # Same as np.amax(x, axis=-1, keepdims=True)
>>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
array([[4],
       [3]])
>>> # Same as np.amax(x, axis=-1)
>>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
...     axis=-1).squeeze(axis=-1)
array([4, 3])

Setting `keepdims` to `True`,

>>> x = np.arange(24).reshape((2, 3, 4))
>>> res = np.argmax(x, axis=1, keepdims=True)
>>> res.shape
(2, 1, 4)
````

## numpy.argsort

Returns the indices that would sort an array.

Perform an indirect sort along the given axis using the algorithm specified
by the `kind` keyword. It returns an array of indices of the same shape as
`a` that index data along the given axis in sorted order.

Parameters
----------
a : array_like
    Array to sort.
axis : int or None, optional
    Axis along which to sort.  The default is -1 (the last axis). If None,
    the flattened array is used.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort under the covers and, in general, the
    actual implementation will vary with data type. The 'mergesort' option
    is retained for backwards compatibility.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.
stable : bool, optional
    Sort stability. If ``True``, the returned array will maintain
    the relative order of ``a`` values which compare as equal.
    If ``False`` or ``None``, this is not guaranteed. Internally,
    this option selects ``kind='stable'``. Default: ``None``.

    .. versionadded:: 2.0.0

Returns
-------
index_array : ndarray, int
    Array of indices that sort `a` along the specified `axis`.
    If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
    More generally, ``np.take_along_axis(a, index_array, axis=axis)``
    always yields the sorted `a`, irrespective of dimensionality.

See Also
--------
sort : Describes sorting algorithms used.
lexsort : Indirect stable sort with multiple keys.
ndarray.sort : Inplace sort.
argpartition : Indirect partial sort.
take_along_axis : Apply ``index_array`` from argsort
                  to an array as if by calling sort.

Notes
-----
See `sort` for notes on the different sorting algorithms.

As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
nan values. The enhanced sort order is documented in `sort`.

Examples
--------

````python
One dimensional array:

>>> import numpy as np
>>> x = np.array([3, 1, 2])
>>> np.argsort(x)
array([1, 2, 0])

Two-dimensional array:

>>> x = np.array([[0, 3], [2, 2]])
>>> x
array([[0, 3],
       [2, 2]])

>>> ind = np.argsort(x, axis=0)  # sorts along first axis (down)
>>> ind
array([[0, 1],
       [1, 0]])
>>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
array([[0, 2],
       [2, 3]])

>>> ind = np.argsort(x, axis=1)  # sorts along last axis (across)
>>> ind
array([[0, 1],
       [0, 1]])
>>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
array([[0, 3],
       [2, 2]])

Indices of the sorted elements of a N-dimensional array:

>>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
>>> ind
(array([0, 1, 1, 0]), array([0, 0, 1, 1]))
>>> x[ind]  # same as np.sort(x, axis=None)
array([0, 2, 2, 3])

Sorting with keys:

>>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
>>> x
array([(1, 0), (0, 1)],
      dtype=[('x', '<i4'), ('y', '<i4')])

>>> np.argsort(x, order=('x','y'))
array([1, 0])

>>> np.argsort(x, order=('y','x'))
array([0, 1])
````

## numpy.sort

Return a sorted copy of an array.

Parameters
----------
a : array_like
    Array to be sorted.
axis : int or None, optional
    Axis along which to sort. If None, the array is flattened before
    sorting. The default is -1, which sorts along the last axis.
kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
    Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
    and 'mergesort' use timsort or radix sort under the covers and,
    in general, the actual implementation will vary with data type.
    The 'mergesort' option is retained for backwards compatibility.
order : str or list of str, optional
    When `a` is an array with fields defined, this argument specifies
    which fields to compare first, second, etc.  A single field can
    be specified as a string, and not all fields need be specified,
    but unspecified fields will still be used, in the order in which
    they come up in the dtype, to break ties.
stable : bool, optional
    Sort stability. If ``True``, the returned array will maintain
    the relative order of ``a`` values which compare as equal.
    If ``False`` or ``None``, this is not guaranteed. Internally,
    this option selects ``kind='stable'``. Default: ``None``.

    .. versionadded:: 2.0.0

Returns
-------
sorted_array : ndarray
    Array of the same type and shape as `a`.

See Also
--------
ndarray.sort : Method to sort an array in-place.
argsort : Indirect sort.
lexsort : Indirect stable sort on multiple keys.
searchsorted : Find elements in a sorted array.
partition : Partial sort.

Notes
-----
The various sorting algorithms are characterized by their average speed,
worst case performance, work space size, and whether they are stable. A
stable sort keeps items with the same key in the same relative
order. The four algorithms implemented in NumPy have the following
properties:

=========== ======= ============= ============ ========
   kind      speed   worst case    work space   stable
=========== ======= ============= ============ ========
'quicksort'    1     O(n^2)            0          no
'heapsort'     3     O(n*log(n))       0          no
'mergesort'    2     O(n*log(n))      ~n/2        yes
'timsort'      2     O(n*log(n))      ~n/2        yes
=========== ======= ============= ============ ========

.. note:: The datatype determines which of 'mergesort' or 'timsort'
   is actually used, even if 'mergesort' is specified. User selection
   at a finer scale is not currently available.

For performance, ``sort`` makes a temporary copy if needed to make the data
`contiguous <https://numpy.org/doc/stable/glossary.html#term-contiguous>`_
in memory along the sort axis. For even better performance and reduced
memory consumption, ensure that the array is already contiguous along the
sort axis.

The sort order for complex numbers is lexicographic. If both the real
and imaginary parts are non-nan then the order is determined by the
real parts except when they are equal, in which case the order is
determined by the imaginary parts.

Previous to numpy 1.4.0 sorting real and complex arrays containing nan
values led to undefined behaviour. In numpy versions >= 1.4.0 nan
values are sorted to the end. The extended sort order is:

  * Real: [R, nan]
  * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

where R is a non-nan real value. Complex values with the same nan
placements are sorted according to the non-nan part if it exists.
Non-nan values are sorted as before.

quicksort has been changed to:
`introsort <https://en.wikipedia.org/wiki/Introsort>`_.
When sorting does not make enough progress it switches to
`heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
This implementation makes quicksort O(n*log(n)) in the worst case.

'stable' automatically chooses the best stable sorting algorithm
for the data type being sorted.
It, along with 'mergesort' is currently mapped to
`timsort <https://en.wikipedia.org/wiki/Timsort>`_
or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
depending on the data type.
API forward compatibility currently limits the
ability to select the implementation and it is hardwired for the different
data types.

Timsort is added for better performance on already or nearly
sorted data. On random data timsort is almost identical to
mergesort. It is now used for stable sort while quicksort is still the
default sort if none is chosen. For timsort details, refer to
`CPython listsort.txt
<https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_
'mergesort' and 'stable' are mapped to radix sort for integer data types.
Radix sort is an O(n) sort instead of O(n log n).

NaT now sorts to the end of arrays for consistency with NaN.

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1,4],[3,1]])
>>> np.sort(a)                # sort along the last axis
array([[1, 4],
       [1, 3]])
>>> np.sort(a, axis=None)     # sort the flattened array
array([1, 1, 3, 4])
>>> np.sort(a, axis=0)        # sort along the first axis
array([[1, 1],
       [3, 4]])

Use the `order` keyword to specify a field to use when sorting a
structured array:

>>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
>>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
...           ('Galahad', 1.7, 38)]
>>> a = np.array(values, dtype=dtype)       # create a structured array
>>> np.sort(a, order='height')                        # doctest: +SKIP
array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
       ('Lancelot', 1.8999999999999999, 38)],
      dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

Sort by age, then height if ages are equal:

>>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
       ('Arthur', 1.8, 41)],
      dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])
````

## numpy.unique

Find the unique elements of an array.

Returns the sorted unique elements of an array. There are three optional
outputs in addition to the unique elements:

* the indices of the input array that give the unique values
* the indices of the unique array that reconstruct the input array
* the number of times each unique value comes up in the input array

Parameters
----------
ar : array_like
    Input array. Unless `axis` is specified, this will be flattened if it
    is not already 1-D.
return_index : bool, optional
    If True, also return the indices of `ar` (along the specified axis,
    if provided, or in the flattened array) that result in the unique array.
return_inverse : bool, optional
    If True, also return the indices of the unique array (for the specified
    axis, if provided) that can be used to reconstruct `ar`.
return_counts : bool, optional
    If True, also return the number of times each unique item appears
    in `ar`.
axis : int or None, optional
    The axis to operate on. If None, `ar` will be flattened. If an integer,
    the subarrays indexed by the given axis will be flattened and treated
    as the elements of a 1-D array with the dimension of the given axis,
    see the notes for more details.  Object arrays or structured arrays
    that contain objects are not supported if the `axis` kwarg is used. The
    default is None.

equal_nan : bool, optional
    If True, collapses multiple NaN values in the return array into one.

    .. versionadded:: 1.24

sorted : bool, optional
    If True, the unique elements are sorted. Elements may be sorted in
    practice even if ``sorted=False``, but this could change without
    notice.

    .. versionadded:: 2.3

Returns
-------
unique : ndarray
    The sorted unique values.
unique_indices : ndarray, optional
    The indices of the first occurrences of the unique values in the
    original array. Only provided if `return_index` is True.
unique_inverse : ndarray, optional
    The indices to reconstruct the original array from the
    unique array. Only provided if `return_inverse` is True.
unique_counts : ndarray, optional
    The number of times each of the unique values comes up in the
    original array. Only provided if `return_counts` is True.

See Also
--------
repeat : Repeat elements of an array.
sort : Return a sorted copy of an array.

Notes
-----
When an axis is specified the subarrays indexed by the axis are sorted.
This is done by making the specified axis the first dimension of the array
(move the axis to the first dimension to keep the order of the other axes)
and then flattening the subarrays in C order. The flattened subarrays are
then viewed as a structured type with each element given a label, with the
effect that we end up with a 1-D array of structured types that can be
treated in the same way as any other 1-D array. The result is that the
flattened subarrays are sorted in lexicographic order starting with the
first element.

.. versionchanged:: 1.21
    Like np.sort, NaN will sort to the end of the values.
    For complex arrays all NaN values are considered equivalent
    (no matter whether the NaN is in the real or imaginary part).
    As the representant for the returned array the smallest one in the
    lexicographical order is chosen - see np.sort for how the lexicographical
    order is defined for complex arrays.

.. versionchanged:: 2.0
    For multi-dimensional inputs, ``unique_inverse`` is reshaped
    such that the input can be reconstructed using
    ``np.take(unique, unique_inverse, axis=axis)``. The result is
    now not 1-dimensional when ``axis=None``.

    Note that in NumPy 2.0.0 a higher dimensional array was returned also
    when ``axis`` was not ``None``.  This was reverted, but
    ``inverse.reshape(-1)`` can be used to ensure compatibility with both
    versions.

Examples
--------

````python
>>> import numpy as np
>>> np.unique([1, 1, 2, 2, 3, 3])
array([1, 2, 3])
>>> a = np.array([[1, 1], [2, 3]])
>>> np.unique(a)
array([1, 2, 3])

Return the unique rows of a 2D array

>>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
>>> np.unique(a, axis=0)
array([[1, 0, 0], [2, 3, 4]])

Return the indices of the original array that give the unique values:

>>> a = np.array(['a', 'b', 'b', 'c', 'a'])
>>> u, indices = np.unique(a, return_index=True)
>>> u
array(['a', 'b', 'c'], dtype='<U1')
>>> indices
array([0, 1, 3])
>>> a[indices]
array(['a', 'b', 'c'], dtype='<U1')

Reconstruct the input array from the unique values and inverse:

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> u, indices = np.unique(a, return_inverse=True)
>>> u
array([1, 2, 3, 4, 6])
>>> indices
array([0, 1, 4, 3, 1, 2, 1])
>>> u[indices]
array([1, 2, 6, 4, 2, 3, 2])

Reconstruct the input values from the unique values and counts:

>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> values, counts = np.unique(a, return_counts=True)
>>> values
array([1, 2, 3, 4, 6])
>>> counts
array([1, 3, 1, 1, 1])
>>> np.repeat(values, counts)
array([1, 2, 2, 2, 3, 4, 6])    # original order not preserved
````

## numpy.searchsorted

Find indices where elements should be inserted to maintain order.

Find the indices into a sorted array `a` such that, if the
corresponding elements in `v` were inserted before the indices, the
order of `a` would be preserved.

Assuming that `a` is sorted:

======  ============================
`side`  returned index `i` satisfies
======  ============================
left    ``a[i-1] < v <= a[i]``
right   ``a[i-1] <= v < a[i]``
======  ============================

Parameters
----------
a : 1-D array_like
    Input array. If `sorter` is None, then it must be sorted in
    ascending order, otherwise `sorter` must be an array of indices
    that sort it.
v : array_like
    Values to insert into `a`.
side : {'left', 'right'}, optional
    If 'left', the index of the first suitable location found is given.
    If 'right', return the last such index.  If there is no suitable
    index, return either 0 or N (where N is the length of `a`).
sorter : 1-D array_like, optional
    Optional array of integer indices that sort array a into ascending
    order. They are typically the result of argsort.

Returns
-------
indices : int or array of ints
    Array of insertion points with the same shape as `v`,
    or an integer if `v` is a scalar.

See Also
--------
sort : Return a sorted copy of an array.
histogram : Produce histogram from 1-D data.

Notes
-----
Binary search is used to find the required insertion points.

As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
`nan` values. The enhanced sort order is documented in `sort`.

This function uses the same algorithm as the builtin python
`bisect.bisect_left` (``side='left'``) and `bisect.bisect_right`
(``side='right'``) functions, which is also vectorized
in the `v` argument.

Examples
--------

````python
>>> import numpy as np
>>> np.searchsorted([11,12,13,14,15], 13)
2
>>> np.searchsorted([11,12,13,14,15], 13, side='right')
3
>>> np.searchsorted([11,12,13,14,15], [-10, 20, 12, 13])
array([0, 5, 1, 2])

When `sorter` is used, the returned indices refer to the sorted
array of `a` and not `a` itself:

>>> a = np.array([40, 10, 20, 30])
>>> sorter = np.argsort(a)
>>> sorter
array([1, 2, 3, 0])  # Indices that would sort the array 'a'
>>> result = np.searchsorted(a, 25, sorter=sorter)
>>> result
2
>>> a[sorter[result]]
30  # The element at index 2 of the sorted array is 30.
````

## numpy.clip

Clip (limit) the values in an array.

Given an interval, values outside the interval are clipped to
the interval edges.  For example, if an interval of ``[0, 1]``
is specified, values smaller than 0 become 0, and values larger
than 1 become 1.

Equivalent to but faster than ``np.minimum(a_max, np.maximum(a, a_min))``.

No check is performed to ensure ``a_min < a_max``.

Parameters
----------
a : array_like
    Array containing elements to clip.
a_min, a_max : array_like or None
    Minimum and maximum value. If ``None``, clipping is not performed on
    the corresponding edge. If both ``a_min`` and ``a_max`` are ``None``,
    the elements of the returned array stay the same. Both are broadcasted
    against ``a``.
out : ndarray, optional
    The results will be placed in this array. It may be the input
    array for in-place clipping.  `out` must be of the right shape
    to hold the output.  Its type is preserved.
min, max : array_like or None
    Array API compatible alternatives for ``a_min`` and ``a_max``
    arguments. Either ``a_min`` and ``a_max`` or ``min`` and ``max``
    can be passed at the same time. Default: ``None``.

    .. versionadded:: 2.1.0
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
clipped_array : ndarray
    An array with the elements of `a`, but where values
    < `a_min` are replaced with `a_min`, and those > `a_max`
    with `a_max`.

See Also
--------
:ref:`ufuncs-output-type`

Notes
-----
When `a_min` is greater than `a_max`, `clip` returns an
array in which all values are equal to `a_max`,
as shown in the second example.

Examples
--------

````python
>>> import numpy as np
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, 1, 8)
array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
>>> np.clip(a, 8, 1)
array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> np.clip(a, 3, 6, out=a)
array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
>>> a
array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
>>> a = np.arange(10)
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])
````

## numpy.mean

Compute the arithmetic mean along the specified axis.

Returns the average of the array elements.  The average is taken over
the flattened array by default, otherwise over the specified axis.
`float64` intermediate and return values are used for integer inputs.

Parameters
----------
a : array_like
    Array containing numbers whose mean is desired. If `a` is not an
    array, a conversion is attempted.
axis : None or int or tuple of ints, optional
    Axis or axes along which the means are computed. The default is to
    compute the mean of the flattened array.

    If this is a tuple of ints, a mean is performed over multiple axes,
    instead of a single axis or all the axes as before.
dtype : data-type, optional
    Type to use in computing the mean.  For integer inputs, the default
    is `float64`; for floating point inputs, it is the same as the
    input dtype.
out : ndarray, optional
    Alternate output array in which to place the result.  The default
    is ``None``; if provided, it must have the same shape as the
    expected output, but the type will be cast if necessary.
    See :ref:`ufuncs-output-type` for more details.
    See :ref:`ufuncs-output-type` for more details.

keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `mean` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.

where : array_like of bool, optional
    Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.20.0

Returns
-------
m : ndarray, see dtype parameter above
    If `out=None`, returns a new array containing the mean values,
    otherwise a reference to the output array is returned.

See Also
--------
average : Weighted average
std, var, nanmean, nanstd, nanvar

Notes
-----
The arithmetic mean is the sum of the elements along the axis divided
by the number of elements.

Note that for floating-point input, the mean is computed using the
same precision the input has.  Depending on the input data, this can
cause the results to be inaccurate, especially for `float32` (see
example below).  Specifying a higher-precision accumulator using the
`dtype` keyword can alleviate this issue.

By default, `float16` results are computed using `float32` intermediates
for extra precision.

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> np.mean(a)
2.5
>>> np.mean(a, axis=0)
array([2., 3.])
>>> np.mean(a, axis=1)
array([1.5, 3.5])

In single precision, `mean` can be inaccurate:

>>> a = np.zeros((2, 512*512), dtype=np.float32)
>>> a[0, :] = 1.0
>>> a[1, :] = 0.1
>>> np.mean(a)
np.float32(0.54999924)

Computing the mean in float64 is more accurate:

>>> np.mean(a, dtype=np.float64)
0.55000000074505806 # may vary

Computing the mean in timedelta64 is available:

>>> b = np.array([1, 3], dtype="timedelta64[D]")
>>> np.mean(b)
np.timedelta64(2,'D')

Specifying a where argument:

>>> a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]])
>>> np.mean(a)
12.0
>>> np.mean(a, where=[[True], [False], [False]])
9.0
````

## numpy.std

Compute the standard deviation along the specified axis.

Returns the standard deviation, a measure of the spread of a distribution,
of the array elements. The standard deviation is computed for the
flattened array by default, otherwise over the specified axis.

Parameters
----------
a : array_like
    Calculate the standard deviation of these values.
axis : None or int or tuple of ints, optional
    Axis or axes along which the standard deviation is computed. The
    default is to compute the standard deviation of the flattened array.
    If this is a tuple of ints, a standard deviation is performed over
    multiple axes, instead of a single axis or all the axes as before.
dtype : dtype, optional
    Type to use in computing the standard deviation. For arrays of
    integer type the default is float64, for arrays of float types it is
    the same as the array type.
out : ndarray, optional
    Alternative output array in which to place the result. It must have
    the same shape as the expected output but the type (of the calculated
    values) will be cast if necessary.
    See :ref:`ufuncs-output-type` for more details.
ddof : {int, float}, optional
    Means Delta Degrees of Freedom.  The divisor used in calculations
    is ``N - ddof``, where ``N`` represents the number of elements.
    By default `ddof` is zero. See Notes for details about use of `ddof`.
keepdims : bool, optional
    If this is set to True, the axes which are reduced are left
    in the result as dimensions with size one. With this option,
    the result will broadcast correctly against the input array.

    If the default value is passed, then `keepdims` will not be
    passed through to the `std` method of sub-classes of
    `ndarray`, however any non-default value will be.  If the
    sub-class' method does not implement `keepdims` any
    exceptions will be raised.
where : array_like of bool, optional
    Elements to include in the standard deviation.
    See `~numpy.ufunc.reduce` for details.

    .. versionadded:: 1.20.0

mean : array_like, optional
    Provide the mean to prevent its recalculation. The mean should have
    a shape as if it was calculated with ``keepdims=True``.
    The axis for the calculation of the mean should be the same as used in
    the call to this std function.

    .. versionadded:: 2.0.0

correction : {int, float}, optional
    Array API compatible name for the ``ddof`` parameter. Only one of them
    can be provided at the same time.

    .. versionadded:: 2.0.0

Returns
-------
standard_deviation : ndarray, see dtype parameter above.
    If `out` is None, return a new array containing the standard deviation,
    otherwise return a reference to the output array.

See Also
--------
var, mean, nanmean, nanstd, nanvar
:ref:`ufuncs-output-type`

Notes
-----
There are several common variants of the array standard deviation
calculation. Assuming the input `a` is a one-dimensional NumPy array
and ``mean`` is either provided as an argument or computed as
``a.mean()``, NumPy computes the standard deviation of an array as::

    N = len(a)
    d2 = abs(a - mean)**2  # abs is for complex `a`
    var = d2.sum() / (N - ddof)  # note use of `ddof`
    std = var**0.5

Different values of the argument `ddof` are useful in different
contexts. NumPy's default ``ddof=0`` corresponds with the expression:

.. math::

    \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}

which is sometimes called the "population standard deviation" in the field
of statistics because it applies the definition of standard deviation to
`a` as if `a` were a complete population of possible observations.

Many other libraries define the standard deviation of an array
differently, e.g.:

.. math::

    \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N - 1}}

In statistics, the resulting quantity is sometimes called the "sample
standard deviation" because if `a` is a random sample from a larger
population, this calculation provides the square root of an unbiased
estimate of the variance of the population. The use of :math:`N-1` in the
denominator is often called "Bessel's correction" because it corrects for
bias (toward lower values) in the variance estimate introduced when the
sample mean of `a` is used in place of the true mean of the population.
The resulting estimate of the standard deviation is still biased, but less
than it would have been without the correction. For this quantity, use
``ddof=1``.

Note that, for complex numbers, `std` takes the absolute
value before squaring, so that the result is always real and nonnegative.

For floating-point input, the standard deviation is computed using the same
precision the input has. Depending on the input data, this can cause
the results to be inaccurate, especially for float32 (see example below).
Specifying a higher-accuracy accumulator using the `dtype` keyword can
alleviate this issue.

Examples
--------

````python
>>> import numpy as np
>>> a = np.array([[1, 2], [3, 4]])
>>> np.std(a)
1.1180339887498949 # may vary
>>> np.std(a, axis=0)
array([1.,  1.])
>>> np.std(a, axis=1)
array([0.5,  0.5])

In single precision, std() can be inaccurate:

>>> a = np.zeros((2, 512*512), dtype=np.float32)
>>> a[0, :] = 1.0
>>> a[1, :] = 0.1
>>> np.std(a)
np.float32(0.45000005)

Computing the standard deviation in float64 is more accurate:

>>> np.std(a, dtype=np.float64)
0.44999999925494177 # may vary

Specifying a where argument:

>>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
>>> np.std(a)
2.614064523559687 # may vary
>>> np.std(a, where=[[True], [True], [False]])
2.0

Using the mean keyword to save computation time:

>>> import numpy as np
>>> from timeit import timeit
>>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
>>> mean = np.mean(a, axis=1, keepdims=True)
>>>
>>> g = globals()
>>> n = 10000
>>> t1 = timeit("std = np.std(a, axis=1, mean=mean)", globals=g, number=n)
>>> t2 = timeit("std = np.std(a, axis=1)", globals=g, number=n)
>>> print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
#doctest: +SKIP
Percentage execution time saved 30%
````

## numpy.einsum

einsum(subscripts, *operands, out=None, dtype=None, order='K',
       casting='safe', optimize=False)

Evaluates the Einstein summation convention on the operands.

Using the Einstein summation convention, many common multi-dimensional,
linear algebraic array operations can be represented in a simple fashion.
In *implicit* mode `einsum` computes these values.

In *explicit* mode, `einsum` provides further flexibility to compute
other array operations that might not be considered classical Einstein
summation operations, by disabling, or forcing summation over specified
subscript labels.

See the notes and examples for clarification.

Parameters
----------
subscripts : str
    Specifies the subscripts for summation as comma separated list of
    subscript labels. An implicit (classical Einstein summation)
    calculation is performed unless the explicit indicator '->' is
    included as well as subscript labels of the precise output form.
operands : list of array_like
    These are the arrays for the operation.
out : ndarray, optional
    If provided, the calculation is done into this array.
dtype : {data-type, None}, optional
    If provided, forces the calculation to use the data type specified.
    Note that you may have to also give a more liberal `casting`
    parameter to allow the conversions. Default is None.
order : {'C', 'F', 'A', 'K'}, optional
    Controls the memory layout of the output. 'C' means it should
    be C contiguous. 'F' means it should be Fortran contiguous,
    'A' means it should be 'F' if the inputs are all 'F', 'C' otherwise.
    'K' means it should be as close to the layout as the inputs as
    is possible, including arbitrarily permuted axes.
    Default is 'K'.
casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
    Controls what kind of data casting may occur.  Setting this to
    'unsafe' is not recommended, as it can adversely affect accumulations.

    * 'no' means the data types should not be cast at all.
    * 'equiv' means only byte-order changes are allowed.
    * 'safe' means only casts which can preserve values are allowed.
    * 'same_kind' means only safe casts or casts within a kind,
      like float64 to float32, are allowed.
    * 'unsafe' means any data conversions may be done.

    Default is 'safe'.
optimize : {False, True, 'greedy', 'optimal'}, optional
    Controls if intermediate optimization should occur. No optimization
    will occur if False and True will default to the 'greedy' algorithm.
    Also accepts an explicit contraction list from the ``np.einsum_path``
    function. See ``np.einsum_path`` for more details. Defaults to False.

Returns
-------
output : ndarray
    The calculation based on the Einstein summation convention.

See Also
--------
einsum_path, dot, inner, outer, tensordot, linalg.multi_dot
einsum:
    Similar verbose interface is provided by the
    `einops <https://github.com/arogozhnikov/einops>`_ package to cover
    additional operations: transpose, reshape/flatten, repeat/tile,
    squeeze/unsqueeze and reductions.
    The `opt_einsum <https://optimized-einsum.readthedocs.io/en/stable/>`_
    optimizes contraction order for einsum-like expressions
    in backend-agnostic manner.

Notes
-----
The Einstein summation convention can be used to compute
many multi-dimensional, linear algebraic array operations. `einsum`
provides a succinct way of representing these.

A non-exhaustive list of these operations,
which can be computed by `einsum`, is shown below along with examples:

* Trace of an array, :py:func:`numpy.trace`.
* Return a diagonal, :py:func:`numpy.diag`.
* Array axis summations, :py:func:`numpy.sum`.
* Transpositions and permutations, :py:func:`numpy.transpose`.
* Matrix multiplication and dot product, :py:func:`numpy.matmul`
    :py:func:`numpy.dot`.
* Vector inner and outer products, :py:func:`numpy.inner`
    :py:func:`numpy.outer`.
* Broadcasting, element-wise and scalar multiplication,
    :py:func:`numpy.multiply`.
* Tensor contractions, :py:func:`numpy.tensordot`.
* Chained array operations, in efficient calculation order,
    :py:func:`numpy.einsum_path`.

The subscripts string is a comma-separated list of subscript labels,
where each label refers to a dimension of the corresponding operand.
Whenever a label is repeated it is summed, so ``np.einsum('i,i', a, b)``
is equivalent to :py:func:`np.inner(a,b) <numpy.inner>`. If a label
appears only once, it is not summed, so ``np.einsum('i', a)``
produces a view of ``a`` with no changes. A further example
``np.einsum('ij,jk', a, b)`` describes traditional matrix multiplication
and is equivalent to :py:func:`np.matmul(a,b) <numpy.matmul>`.
Repeated subscript labels in one operand take the diagonal.
For example, ``np.einsum('ii', a)`` is equivalent to
:py:func:`np.trace(a) <numpy.trace>`.

In *implicit mode*, the chosen subscripts are important
since the axes of the output are reordered alphabetically.  This
means that ``np.einsum('ij', a)`` doesn't affect a 2D array, while
``np.einsum('ji', a)`` takes its transpose. Additionally,
``np.einsum('ij,jk', a, b)`` returns a matrix multiplication, while,
``np.einsum('ij,jh', a, b)`` returns the transpose of the
multiplication since subscript 'h' precedes subscript 'i'.

In *explicit mode* the output can be directly controlled by
specifying output subscript labels.  This requires the
identifier '->' as well as the list of output subscript labels.
This feature increases the flexibility of the function since
summing can be disabled or forced when required. The call
``np.einsum('i->', a)`` is like :py:func:`np.sum(a) <numpy.sum>`
if ``a`` is a 1-D array, and ``np.einsum('ii->i', a)``
is like :py:func:`np.diag(a) <numpy.diag>` if ``a`` is a square 2-D array.
The difference is that `einsum` does not allow broadcasting by default.
Additionally ``np.einsum('ij,jh->ih', a, b)`` directly specifies the
order of the output subscript labels and therefore returns matrix
multiplication, unlike the example above in implicit mode.

To enable and control broadcasting, use an ellipsis.  Default
NumPy-style broadcasting is done by adding an ellipsis
to the left of each term, like ``np.einsum('...ii->...i', a)``.
``np.einsum('...i->...', a)`` is like
:py:func:`np.sum(a, axis=-1) <numpy.sum>` for array ``a`` of any shape.
To take the trace along the first and last axes,
you can do ``np.einsum('i...i', a)``, or to do a matrix-matrix
product with the left-most indices instead of rightmost, one can do
``np.einsum('ij...,jk...->ik...', a, b)``.

When there is only one operand, no axes are summed, and no output
parameter is provided, a view into the operand is returned instead
of a new array.  Thus, taking the diagonal as ``np.einsum('ii->i', a)``
produces a view (changed in version 1.10.0).

`einsum` also provides an alternative way to provide the subscripts and
operands as ``einsum(op0, sublist0, op1, sublist1, ..., [sublistout])``.
If the output shape is not provided in this format `einsum` will be
calculated in implicit mode, otherwise it will be performed explicitly.
The examples below have corresponding `einsum` calls with the two
parameter methods.

Views returned from einsum are now writeable whenever the input array
is writeable. For example, ``np.einsum('ijk...->kji...', a)`` will now
have the same effect as :py:func:`np.swapaxes(a, 0, 2) <numpy.swapaxes>`
and ``np.einsum('ii->i', a)`` will return a writeable view of the diagonal
of a 2D array.

Added the ``optimize`` argument which will optimize the contraction order
of an einsum expression. For a contraction with three or more operands
this can greatly increase the computational efficiency at the cost of
a larger memory footprint during computation.

Typically a 'greedy' algorithm is applied which empirical tests have shown
returns the optimal path in the majority of cases. In some cases 'optimal'
will return the superlative path through a more expensive, exhaustive
search. For iterative calculations it may be advisable to calculate
the optimal path once and reuse that path by supplying it as an argument.
An example is given below.

See :py:func:`numpy.einsum_path` for more details.

Examples
--------

````python
>>> a = np.arange(25).reshape(5,5)
>>> b = np.arange(5)
>>> c = np.arange(6).reshape(2,3)

Trace of a matrix:

>>> np.einsum('ii', a)
60
>>> np.einsum(a, [0,0])
60
>>> np.trace(a)
60

Extract the diagonal (requires explicit form):

>>> np.einsum('ii->i', a)
array([ 0,  6, 12, 18, 24])
>>> np.einsum(a, [0,0], [0])
array([ 0,  6, 12, 18, 24])
>>> np.diag(a)
array([ 0,  6, 12, 18, 24])

Sum over an axis (requires explicit form):

>>> np.einsum('ij->i', a)
array([ 10,  35,  60,  85, 110])
>>> np.einsum(a, [0,1], [0])
array([ 10,  35,  60,  85, 110])
>>> np.sum(a, axis=1)
array([ 10,  35,  60,  85, 110])

For higher dimensional arrays summing a single axis can be done
with ellipsis:

>>> np.einsum('...j->...', a)
array([ 10,  35,  60,  85, 110])
>>> np.einsum(a, [Ellipsis,1], [Ellipsis])
array([ 10,  35,  60,  85, 110])

Compute a matrix transpose, or reorder any number of axes:

>>> np.einsum('ji', c)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.einsum('ij->ji', c)
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.einsum(c, [1,0])
array([[0, 3],
       [1, 4],
       [2, 5]])
>>> np.transpose(c)
array([[0, 3],
       [1, 4],
       [2, 5]])

Vector inner products:

>>> np.einsum('i,i', b, b)
30
>>> np.einsum(b, [0], b, [0])
30
>>> np.inner(b,b)
30

Matrix vector multiplication:

>>> np.einsum('ij,j', a, b)
array([ 30,  80, 130, 180, 230])
>>> np.einsum(a, [0,1], b, [1])
array([ 30,  80, 130, 180, 230])
>>> np.dot(a, b)
array([ 30,  80, 130, 180, 230])
>>> np.einsum('...j,j', a, b)
array([ 30,  80, 130, 180, 230])

Broadcasting and scalar multiplication:

>>> np.einsum('..., ...', 3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.einsum(',ij', 3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.einsum(3, [Ellipsis], c, [Ellipsis])
array([[ 0,  3,  6],
       [ 9, 12, 15]])
>>> np.multiply(3, c)
array([[ 0,  3,  6],
       [ 9, 12, 15]])

Vector outer product:

>>> np.einsum('i,j', np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.einsum(np.arange(2)+1, [0], b, [1])
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])
>>> np.outer(np.arange(2)+1, b)
array([[0, 1, 2, 3, 4],
       [0, 2, 4, 6, 8]])

Tensor contraction:

>>> a = np.arange(60.).reshape(3,4,5)
>>> b = np.arange(24.).reshape(4,3,2)
>>> np.einsum('ijk,jil->kl', a, b)
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.einsum(a, [0,1,2], b, [1,0,3], [2,3])
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])
>>> np.tensordot(a,b, axes=([1,0],[0,1]))
array([[4400., 4730.],
       [4532., 4874.],
       [4664., 5018.],
       [4796., 5162.],
       [4928., 5306.]])

Writeable returned arrays (since version 1.10.0):

>>> a = np.zeros((3, 3))
>>> np.einsum('ii->i', a)[:] = 1
>>> a
array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]])

Example of ellipsis use:

>>> a = np.arange(6).reshape((3,2))
>>> b = np.arange(12).reshape((4,3))
>>> np.einsum('ki,jk->ij', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('ki,...k->i...', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])
>>> np.einsum('k...,jk', a, b)
array([[10, 28, 46, 64],
       [13, 40, 67, 94]])

Chained array operations. For more complicated contractions, speed ups
might be achieved by repeatedly computing a 'greedy' path or pre-computing
the 'optimal' path and repeatedly applying it, using an `einsum_path`
insertion (since version 1.12.0). Performance improvements can be
particularly significant with larger arrays:

>>> a = np.ones(64).reshape(2,4,8)

Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a)

Sub-optimal `einsum` (due to repeated path calculation time): ~330ms

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
...         optimize='optimal')

Greedy `einsum` (faster optimal path approximation): ~160ms

>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize='greedy')

Optimal `einsum` (best usage pattern in some use cases): ~110ms

>>> path = np.einsum_path('ijk,ilm,njm,nlk,abc->',a,a,a,a,a,
...     optimize='optimal')[0]
>>> for iteration in range(500):
...     _ = np.einsum('ijk,ilm,njm,nlk,abc->',a,a,a,a,a, optimize=path)
````

## numpy.matmul

matmul(x1, x2, /, out=None, *, casting='same_kind', order='K', dtype=None, subok=True[, signature, axes, axis])

Matrix product of two arrays.

Parameters
----------
x1, x2 : array_like
    Input arrays, scalars not allowed.
out : ndarray, optional
    A location into which the result is stored. If provided, it must have
    a shape that matches the signature `(n,k),(k,m)->(n,m)`. If not
    provided or None, a freshly-allocated array is returned.
**kwargs
    For other keyword-only arguments, see the
    :ref:`ufunc docs <ufuncs.kwargs>`.

Returns
-------
y : ndarray
    The matrix product of the inputs.
    This is a scalar only when both x1, x2 are 1-d vectors.

Raises
------
ValueError
    If the last dimension of `x1` is not the same size as
    the second-to-last dimension of `x2`.

    If a scalar value is passed in.

See Also
--------
vecdot : Complex-conjugating dot product for stacks of vectors.
matvec : Matrix-vector product for stacks of matrices and vectors.
vecmat : Vector-matrix product for stacks of vectors and matrices.
tensordot : Sum products over arbitrary axes.
einsum : Einstein summation convention.
dot : alternative matrix product with different broadcasting rules.

Notes
-----
The behavior depends on the arguments in the following way.

- If both arguments are 2-D they are multiplied like conventional
  matrices.
- If either argument is N-D, N > 2, it is treated as a stack of
  matrices residing in the last two indexes and broadcast accordingly.
- If the first argument is 1-D, it is promoted to a matrix by
  prepending a 1 to its dimensions. After matrix multiplication
  the prepended 1 is removed. (For stacks of vectors, use ``vecmat``.)
- If the second argument is 1-D, it is promoted to a matrix by
  appending a 1 to its dimensions. After matrix multiplication
  the appended 1 is removed. (For stacks of vectors, use ``matvec``.)

``matmul`` differs from ``dot`` in two important ways:

- Multiplication by scalars is not allowed, use ``*`` instead.
- Stacks of matrices are broadcast together as if the matrices
  were elements, respecting the signature ``(n,k),(k,m)->(n,m)``:

  >>> a = np.ones([9, 5, 7, 4])
  >>> c = np.ones([9, 5, 4, 3])
  >>> np.dot(a, c).shape
  (9, 5, 7, 9, 5, 3)
  >>> np.matmul(a, c).shape
  (9, 5, 7, 3)
  >>> # n is 7, k is 4, m is 3

The matmul function implements the semantics of the ``@`` operator
defined in :pep:`465`.

It uses an optimized BLAS library when possible (see `numpy.linalg`).

Examples
--------

````python
For 2-D arrays it is the matrix product:

>>> import numpy as np
>>> a = np.array([[1, 0],
...               [0, 1]])
>>> b = np.array([[4, 1],
...               [2, 2]])
>>> np.matmul(a, b)
array([[4, 1],
       [2, 2]])

For 2-D mixed with 1-D, the result is the usual.

>>> a = np.array([[1, 0],
...               [0, 1]])
>>> b = np.array([1, 2])
>>> np.matmul(a, b)
array([1, 2])
>>> np.matmul(b, a)
array([1, 2])


Broadcasting is conventional for stacks of arrays

>>> a = np.arange(2 * 2 * 4).reshape((2, 2, 4))
>>> b = np.arange(2 * 2 * 4).reshape((2, 4, 2))
>>> np.matmul(a,b).shape
(2, 2, 2)
>>> np.matmul(a, b)[0, 1, 1]
98
>>> sum(a[0, 1, :] * b[0 , :, 1])
98

Vector, vector returns the scalar inner product, but neither argument
is complex-conjugated:

>>> np.matmul([2j, 3j], [2j, 3j])
(-13+0j)

Scalar multiplication raises an error.

>>> np.matmul([1,2], 3)
Traceback (most recent call last):
...
ValueError: matmul: Input operand 1 does not have enough dimensions ...

The ``@`` operator can be used as a shorthand for ``np.matmul`` on
ndarrays.

>>> x1 = np.array([2j, 3j])
>>> x2 = np.array([2j, 3j])
>>> x1 @ x2
(-13+0j)
````

## numpy.linspace

Return evenly spaced numbers over a specified interval.

Returns `num` evenly spaced samples, calculated over the
interval [`start`, `stop`].

The endpoint of the interval can optionally be excluded.

.. versionchanged:: 1.20.0
    Values are rounded towards ``-inf`` instead of ``0`` when an
    integer ``dtype`` is specified. The old behavior can
    still be obtained with ``np.linspace(start, stop, num).astype(int)``

Parameters
----------
start : array_like
    The starting value of the sequence.
stop : array_like
    The end value of the sequence, unless `endpoint` is set to False.
    In that case, the sequence consists of all but the last of ``num + 1``
    evenly spaced samples, so that `stop` is excluded.  Note that the step
    size changes when `endpoint` is False.
num : int, optional
    Number of samples to generate. Default is 50. Must be non-negative.
endpoint : bool, optional
    If True, `stop` is the last sample. Otherwise, it is not included.
    Default is True.
retstep : bool, optional
    If True, return (`samples`, `step`), where `step` is the spacing
    between samples.
dtype : dtype, optional
    The type of the output array.  If `dtype` is not given, the data type
    is inferred from `start` and `stop`. The inferred dtype will never be
    an integer; `float` is chosen even if the arguments would produce an
    array of integers.
axis : int, optional
    The axis in the result to store the samples.  Relevant only if start
    or stop are array-like.  By default (0), the samples will be along a
    new axis inserted at the beginning. Use -1 to get an axis at the end.
device : str, optional
    The device on which to place the created array. Default: None.
    For Array-API interoperability only, so must be ``"cpu"`` if passed.

    .. versionadded:: 2.0.0

Returns
-------
samples : ndarray
    There are `num` equally spaced samples in the closed interval
    ``[start, stop]`` or the half-open interval ``[start, stop)``
    (depending on whether `endpoint` is True or False).
step : float, optional
    Only returned if `retstep` is True

    Size of spacing between samples.


See Also
--------
arange : Similar to `linspace`, but uses a step size (instead of the
         number of samples).
geomspace : Similar to `linspace`, but with numbers spaced evenly on a log
            scale (a geometric progression).
logspace : Similar to `geomspace`, but with the end points specified as
           logarithms.
:ref:`how-to-partition`

Examples
--------

````python
>>> import numpy as np
>>> np.linspace(2.0, 3.0, num=5)
array([2.  , 2.25, 2.5 , 2.75, 3.  ])
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([2. ,  2.2,  2.4,  2.6,  2.8])
>>> np.linspace(2.0, 3.0, num=5, retstep=True)
(array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

Graphical illustration:

>>> import matplotlib.pyplot as plt
>>> N = 8
>>> y = np.zeros(N)
>>> x1 = np.linspace(0, 10, N, endpoint=True)
>>> x2 = np.linspace(0, 10, N, endpoint=False)
>>> plt.plot(x1, y, 'o')
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.plot(x2, y + 0.5, 'o')
[<matplotlib.lines.Line2D object at 0x...>]
>>> plt.ylim([-0.5, 1])
(-0.5, 1)
>>> plt.show()
````

## numpy.isclose

Returns a boolean array where two arrays are element-wise equal within a
tolerance.

The tolerance values are positive, typically very small numbers.  The
relative difference (`rtol` * abs(`b`)) and the absolute difference
`atol` are added together to compare against the absolute difference
between `a` and `b`.

.. warning:: The default `atol` is not appropriate for comparing numbers
             with magnitudes much smaller than one (see Notes).

Parameters
----------
a, b : array_like
    Input arrays to compare.
rtol : array_like
    The relative tolerance parameter (see Notes).
atol : array_like
    The absolute tolerance parameter (see Notes).
equal_nan : bool
    Whether to compare NaN's as equal.  If True, NaN's in `a` will be
    considered equal to NaN's in `b` in the output array.

Returns
-------
y : array_like
    Returns a boolean array of where `a` and `b` are equal within the
    given tolerance. If both `a` and `b` are scalars, returns a single
    boolean value.

See Also
--------
allclose
math.isclose

Notes
-----
For finite values, isclose uses the following equation to test whether
two floating point values are equivalent.::

 absolute(a - b) <= (atol + rtol * absolute(b))

Unlike the built-in `math.isclose`, the above equation is not symmetric
in `a` and `b` -- it assumes `b` is the reference value -- so that
`isclose(a, b)` might be different from `isclose(b, a)`.

The default value of `atol` is not appropriate when the reference value
`b` has magnitude smaller than one. For example, it is unlikely that
``a = 1e-9`` and ``b = 2e-9`` should be considered "close", yet
``isclose(1e-9, 2e-9)`` is ``True`` with default settings. Be sure
to select `atol` for the use case at hand, especially for defining the
threshold below which a non-zero value in `a` will be considered "close"
to a very small or zero value in `b`.

`isclose` is not defined for non-numeric data types.
:class:`bool` is considered a numeric data-type for this purpose.

Examples
--------

````python
>>> import numpy as np
>>> np.isclose([1e10,1e-7], [1.00001e10,1e-8])
array([ True, False])

>>> np.isclose([1e10,1e-8], [1.00001e10,1e-9])
array([ True, True])

>>> np.isclose([1e10,1e-8], [1.0001e10,1e-9])
array([False,  True])

>>> np.isclose([1.0, np.nan], [1.0, np.nan])
array([ True, False])

>>> np.isclose([1.0, np.nan], [1.0, np.nan], equal_nan=True)
array([ True, True])

>>> np.isclose([1e-8, 1e-7], [0.0, 0.0])
array([ True, False])

>>> np.isclose([1e-100, 1e-7], [0.0, 0.0], atol=0.0)
array([False, False])

>>> np.isclose([1e-10, 1e-10], [1e-20, 0.0])
array([ True,  True])

>>> np.isclose([1e-10, 1e-10], [1e-20, 0.999999e-10], atol=0.0)
array([False,  True])
````

## numpy.linalg.solve

Solve a linear matrix equation, or system of linear scalar equations.

Computes the "exact" solution, `x`, of the well-determined, i.e., full
rank, linear matrix equation `ax = b`.

Parameters
----------
a : (..., M, M) array_like
    Coefficient matrix.
b : {(M,), (..., M, K)}, array_like
    Ordinate or "dependent variable" values.

Returns
-------
x : {(..., M,), (..., M, K)} ndarray
    Solution to the system a x = b.  Returned shape is (..., M) if b is
    shape (M,) and (..., M, K) if b is (..., M, K), where the "..." part is
    broadcasted between a and b.

Raises
------
LinAlgError
    If `a` is singular or not square.

See Also
--------
scipy.linalg.solve : Similar function in SciPy.

Notes
-----
Broadcasting rules apply, see the `numpy.linalg` documentation for
details.

The solutions are computed using LAPACK routine ``_gesv``.

`a` must be square and of full-rank, i.e., all rows (or, equivalently,
columns) must be linearly independent; if either is not true, use
`lstsq` for the least-squares best "solution" of the
system/equation.

.. versionchanged:: 2.0

   The b array is only treated as a shape (M,) column vector if it is
   exactly 1-dimensional. In all other instances it is treated as a stack
   of (M, K) matrices. Previously b would be treated as a stack of (M,)
   vectors if b.ndim was equal to a.ndim - 1.

References
----------
.. [1] G. Strang, *Linear Algebra and Its Applications*, 2nd Ed., Orlando,
       FL, Academic Press, Inc., 1980, pg. 22.

Examples
--------

````python
Solve the system of equations:
``x0 + 2 * x1 = 1`` and
``3 * x0 + 5 * x1 = 2``:

>>> import numpy as np
>>> a = np.array([[1, 2], [3, 5]])
>>> b = np.array([1, 2])
>>> x = np.linalg.solve(a, b)
>>> x
array([-1.,  1.])

Check that the solution is correct:

>>> np.allclose(np.dot(a, x), b)
True
````

## numpy.linalg.svd

Singular Value Decomposition.

When `a` is a 2D array, and ``full_matrices=False``, then it is
factorized as ``u @ np.diag(s) @ vh = (u * s) @ vh``, where
`u` and the Hermitian transpose of `vh` are 2D arrays with
orthonormal columns and `s` is a 1D array of `a`'s singular
values. When `a` is higher-dimensional, SVD is applied in
stacked mode as explained below.

Parameters
----------
a : (..., M, N) array_like
    A real or complex array with ``a.ndim >= 2``.
full_matrices : bool, optional
    If True (default), `u` and `vh` have the shapes ``(..., M, M)`` and
    ``(..., N, N)``, respectively.  Otherwise, the shapes are
    ``(..., M, K)`` and ``(..., K, N)``, respectively, where
    ``K = min(M, N)``.
compute_uv : bool, optional
    Whether or not to compute `u` and `vh` in addition to `s`.  True
    by default.
hermitian : bool, optional
    If True, `a` is assumed to be Hermitian (symmetric if real-valued),
    enabling a more efficient method for finding singular values.
    Defaults to False.

Returns
-------
When `compute_uv` is True, the result is a namedtuple with the following
attribute names:

U : { (..., M, M), (..., M, K) } array
    Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
    size as those of the input `a`. The size of the last two dimensions
    depends on the value of `full_matrices`. Only returned when
    `compute_uv` is True.
S : (..., K) array
    Vector(s) with the singular values, within each vector sorted in
    descending order. The first ``a.ndim - 2`` dimensions have the same
    size as those of the input `a`.
Vh : { (..., N, N), (..., K, N) } array
    Unitary array(s). The first ``a.ndim - 2`` dimensions have the same
    size as those of the input `a`. The size of the last two dimensions
    depends on the value of `full_matrices`. Only returned when
    `compute_uv` is True.

Raises
------
LinAlgError
    If SVD computation does not converge.

See Also
--------
scipy.linalg.svd : Similar function in SciPy.
scipy.linalg.svdvals : Compute singular values of a matrix.

Notes
-----
The decomposition is performed using LAPACK routine ``_gesdd``.

SVD is usually described for the factorization of a 2D matrix :math:`A`.
The higher-dimensional case will be discussed below. In the 2D case, SVD is
written as :math:`A = U S V^H`, where :math:`A = a`, :math:`U= u`,
:math:`S= \mathtt{np.diag}(s)` and :math:`V^H = vh`. The 1D array `s`
contains the singular values of `a` and `u` and `vh` are unitary. The rows
of `vh` are the eigenvectors of :math:`A^H A` and the columns of `u` are
the eigenvectors of :math:`A A^H`. In both cases the corresponding
(possibly non-zero) eigenvalues are given by ``s**2``.

If `a` has more than two dimensions, then broadcasting rules apply, as
explained in :ref:`routines.linalg-broadcasting`. This means that SVD is
working in "stacked" mode: it iterates over all indices of the first
``a.ndim - 2`` dimensions and for each combination SVD is applied to the
last two indices. The matrix `a` can be reconstructed from the
decomposition with either ``(u * s[..., None, :]) @ vh`` or
``u @ (s[..., None] * vh)``. (The ``@`` operator can be replaced by the
function ``np.matmul`` for python versions below 3.5.)

If `a` is a ``matrix`` object (as opposed to an ``ndarray``), then so are
all the return values.

Examples
--------

````python
>>> import numpy as np
>>> rng = np.random.default_rng()
>>> a = rng.normal(size=(9, 6)) + 1j*rng.normal(size=(9, 6))
>>> b = rng.normal(size=(2, 7, 8, 3)) + 1j*rng.normal(size=(2, 7, 8, 3))


Reconstruction based on full SVD, 2D case:

>>> U, S, Vh = np.linalg.svd(a, full_matrices=True)
>>> U.shape, S.shape, Vh.shape
((9, 9), (6,), (6, 6))
>>> np.allclose(a, np.dot(U[:, :6] * S, Vh))
True
>>> smat = np.zeros((9, 6), dtype=complex)
>>> smat[:6, :6] = np.diag(S)
>>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
True

Reconstruction based on reduced SVD, 2D case:

>>> U, S, Vh = np.linalg.svd(a, full_matrices=False)
>>> U.shape, S.shape, Vh.shape
((9, 6), (6,), (6, 6))
>>> np.allclose(a, np.dot(U * S, Vh))
True
>>> smat = np.diag(S)
>>> np.allclose(a, np.dot(U, np.dot(smat, Vh)))
True

Reconstruction based on full SVD, 4D case:

>>> U, S, Vh = np.linalg.svd(b, full_matrices=True)
>>> U.shape, S.shape, Vh.shape
((2, 7, 8, 8), (2, 7, 3), (2, 7, 3, 3))
>>> np.allclose(b, np.matmul(U[..., :3] * S[..., None, :], Vh))
True
>>> np.allclose(b, np.matmul(U[..., :3], S[..., None] * Vh))
True

Reconstruction based on reduced SVD, 4D case:

>>> U, S, Vh = np.linalg.svd(b, full_matrices=False)
>>> U.shape, S.shape, Vh.shape
((2, 7, 8, 3), (2, 7, 3), (2, 7, 3, 3))
>>> np.allclose(b, np.matmul(U * S[..., None, :], Vh))
True
>>> np.allclose(b, np.matmul(U, S[..., None] * Vh))
True
````

## numpy.random.default_rng

default_rng(seed=None)

Construct a new Generator with the default BitGenerator (PCG64).

Parameters
----------
seed : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator, RandomState}, optional
    A seed to initialize the `BitGenerator`. If None, then fresh,
    unpredictable entropy will be pulled from the OS. If an ``int`` or
    ``array_like[ints]`` is passed, then all values must be non-negative and will be
    passed to `SeedSequence` to derive the initial `BitGenerator` state. One may also
    pass in a `SeedSequence` instance.
    Additionally, when passed a `BitGenerator`, it will be wrapped by
    `Generator`. If passed a `Generator`, it will be returned unaltered.
    When passed a legacy `RandomState` instance it will be coerced to a `Generator`.

Returns
-------
Generator
    The initialized generator object.

Notes
-----
If ``seed`` is not a `BitGenerator` or a `Generator`, a new `BitGenerator`
is instantiated. This function does not manage a default global instance.

See :ref:`seeding_and_entropy` for more information about seeding.

Examples
--------

````python
`default_rng` is the recommended constructor for the random number class
`Generator`. Here are several ways we can construct a random 
number generator using `default_rng` and the `Generator` class. 

Here we use `default_rng` to generate a random float:

>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> print(rng)
Generator(PCG64)
>>> rfloat = rng.random()
>>> rfloat
0.22733602246716966
>>> type(rfloat)
<class 'float'>
 
Here we use `default_rng` to generate 3 random integers between 0 
(inclusive) and 10 (exclusive):
    
>>> import numpy as np
>>> rng = np.random.default_rng(12345)
>>> rints = rng.integers(low=0, high=10, size=3)
>>> rints
array([6, 2, 7])
>>> type(rints[0])
<class 'numpy.int64'>

Here we specify a seed so that we have reproducible results:

>>> import numpy as np
>>> rng = np.random.default_rng(seed=42)
>>> print(rng)
Generator(PCG64)
>>> arr1 = rng.random((3, 3))
>>> arr1
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363]])

If we exit and restart our Python interpreter, we'll see that we
generate the same random numbers again:

>>> import numpy as np
>>> rng = np.random.default_rng(seed=42)
>>> arr2 = rng.random((3, 3))
>>> arr2
array([[0.77395605, 0.43887844, 0.85859792],
       [0.69736803, 0.09417735, 0.97562235],
       [0.7611397 , 0.78606431, 0.12811363]])
````

## numpy.load

Load arrays or pickled objects from ``.npy``, ``.npz`` or pickled files.

.. warning:: Loading files that contain object arrays uses the ``pickle``
             module, which is not secure against erroneous or maliciously
             constructed data. Consider passing ``allow_pickle=False`` to
             load data that is known not to contain object arrays for the
             safer handling of untrusted sources.

Parameters
----------
file : file-like object, string, or pathlib.Path
    The file to read. File-like objects must support the
    ``seek()`` and ``read()`` methods and must always
    be opened in binary mode.  Pickled files require that the
    file-like object support the ``readline()`` method as well.
mmap_mode : {None, 'r+', 'r', 'w+', 'c'}, optional
    If not None, then memory-map the file, using the given mode (see
    `numpy.memmap` for a detailed description of the modes).  A
    memory-mapped array is kept on disk. However, it can be accessed
    and sliced like any ndarray.  Memory mapping is especially useful
    for accessing small fragments of large files without reading the
    entire file into memory.
allow_pickle : bool, optional
    Allow loading pickled object arrays stored in npy files. Reasons for
    disallowing pickles include security, as loading pickled data can
    execute arbitrary code. If pickles are disallowed, loading object
    arrays will fail. Default: False
fix_imports : bool, optional
    Only useful when loading Python 2 generated pickled files,
    which includes npy/npz files containing object arrays. If `fix_imports`
    is True, pickle will try to map the old Python 2 names to the new names
    used in Python 3.
encoding : str, optional
    What encoding to use when reading Python 2 strings. Only useful when
    loading Python 2 generated pickled files, which includes
    npy/npz files containing object arrays. Values other than 'latin1',
    'ASCII', and 'bytes' are not allowed, as they can corrupt numerical
    data. Default: 'ASCII'
max_header_size : int, optional
    Maximum allowed size of the header.  Large headers may not be safe
    to load securely and thus require explicitly passing a larger value.
    See :py:func:`ast.literal_eval()` for details.
    This option is ignored when `allow_pickle` is passed.  In that case
    the file is by definition trusted and the limit is unnecessary.

Returns
-------
result : array, tuple, dict, etc.
    Data stored in the file. For ``.npz`` files, the returned instance
    of NpzFile class must be closed to avoid leaking file descriptors.

Raises
------
OSError
    If the input file does not exist or cannot be read.
UnpicklingError
    If ``allow_pickle=True``, but the file cannot be loaded as a pickle.
ValueError
    The file contains an object array, but ``allow_pickle=False`` given.
EOFError
    When calling ``np.load`` multiple times on the same file handle,
    if all data has already been read

See Also
--------
save, savez, savez_compressed, loadtxt
memmap : Create a memory-map to an array stored in a file on disk.
lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

Notes
-----
- If the file contains pickle data, then whatever object is stored
  in the pickle is returned.
- If the file is a ``.npy`` file, then a single array is returned.
- If the file is a ``.npz`` file, then a dictionary-like object is
  returned, containing ``{filename: array}`` key-value pairs, one for
  each file in the archive.
- If the file is a ``.npz`` file, the returned value supports the
  context manager protocol in a similar fashion to the open function::

    with load('foo.npz') as data:
        a = data['a']

  The underlying file descriptor is closed when exiting the 'with'
  block.

Examples
--------

````python
>>> import numpy as np

Store data to disk, and load it again:

>>> np.save('/tmp/123', np.array([[1, 2, 3], [4, 5, 6]]))
>>> np.load('/tmp/123.npy')
array([[1, 2, 3],
       [4, 5, 6]])

Store compressed data to disk, and load it again:

>>> a=np.array([[1, 2, 3], [4, 5, 6]])
>>> b=np.array([1, 2])
>>> np.savez('/tmp/123.npz', a=a, b=b)
>>> data = np.load('/tmp/123.npz')
>>> data['a']
array([[1, 2, 3],
       [4, 5, 6]])
>>> data['b']
array([1, 2])
>>> data.close()

Mem-map the stored array, and then access the second row
directly from disk:

>>> X = np.load('/tmp/123.npy', mmap_mode='r')
>>> X[1, :]
memmap([4, 5, 6])
````

## numpy.memmap

Create a memory-map to an array stored in a *binary* file on disk.

Memory-mapped files are used for accessing small segments of large files
on disk, without reading the entire file into memory.  NumPy's
memmap's are array-like objects.  This differs from Python's ``mmap``
module, which uses file-like objects.

This subclass of ndarray has some unpleasant interactions with
some operations, because it doesn't quite fit properly as a subclass.
An alternative to using this subclass is to create the ``mmap``
object yourself, then create an ndarray with ndarray.__new__ directly,
passing the object created in its 'buffer=' parameter.

This class may at some point be turned into a factory function
which returns a view into an mmap buffer.

Flush the memmap instance to write the changes to the file. Currently there
is no API to close the underlying ``mmap``. It is tricky to ensure the
resource is actually closed, since it may be shared between different
memmap instances.


Parameters
----------
filename : str, file-like object, or pathlib.Path instance
    The file name or file object to be used as the array data buffer.
dtype : data-type, optional
    The data-type used to interpret the file contents.
    Default is `uint8`.
mode : {'r+', 'r', 'w+', 'c'}, optional
    The file is opened in this mode:

    +------+-------------------------------------------------------------+
    | 'r'  | Open existing file for reading only.                        |
    +------+-------------------------------------------------------------+
    | 'r+' | Open existing file for reading and writing.                 |
    +------+-------------------------------------------------------------+
    | 'w+' | Create or overwrite existing file for reading and writing.  |
    |      | If ``mode == 'w+'`` then `shape` must also be specified.    |
    +------+-------------------------------------------------------------+
    | 'c'  | Copy-on-write: assignments affect data in memory, but       |
    |      | changes are not saved to disk.  The file on disk is         |
    |      | read-only.                                                  |
    +------+-------------------------------------------------------------+

    Default is 'r+'.
offset : int, optional
    In the file, array data starts at this offset. Since `offset` is
    measured in bytes, it should normally be a multiple of the byte-size
    of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
    file are valid; The file will be extended to accommodate the
    additional data. By default, ``memmap`` will start at the beginning of
    the file, even if ``filename`` is a file pointer ``fp`` and
    ``fp.tell() != 0``.
shape : int or sequence of ints, optional
    The desired shape of the array. If ``mode == 'r'`` and the number
    of remaining bytes after `offset` is not a multiple of the byte-size
    of `dtype`, you must specify `shape`. By default, the returned array
    will be 1-D with the number of elements determined by file size
    and data-type.

    .. versionchanged:: 2.0
     The shape parameter can now be any integer sequence type, previously
     types were limited to tuple and int.

order : {'C', 'F'}, optional
    Specify the order of the ndarray memory layout:
    :term:`row-major`, C-style or :term:`column-major`,
    Fortran-style.  This only has an effect if the shape is
    greater than 1-D.  The default order is 'C'.

Attributes
----------
filename : str or pathlib.Path instance
    Path to the mapped file.
offset : int
    Offset position in the file.
mode : str
    File mode.

Methods
-------
flush
    Flush any changes in memory to file on disk.
    When you delete a memmap object, flush is called first to write
    changes to disk.


See also
--------
lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

Notes
-----
The memmap object can be used anywhere an ndarray is accepted.
Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
``True``.

Memory-mapped files cannot be larger than 2GB on 32-bit systems.

When a memmap causes a file to be created or extended beyond its
current size in the filesystem, the contents of the new part are
unspecified. On systems with POSIX filesystem semantics, the extended
part will be filled with zero bytes.

Examples
--------

````python
>>> import numpy as np
>>> data = np.arange(12, dtype='float32')
>>> data.resize((3,4))

This example uses a temporary file so that doctest doesn't write
files to your directory. You would use a 'normal' filename.

>>> from tempfile import mkdtemp
>>> import os.path as path
>>> filename = path.join(mkdtemp(), 'newfile.dat')

Create a memmap with dtype and shape that matches our data:

>>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
>>> fp
memmap([[0., 0., 0., 0.],
        [0., 0., 0., 0.],
        [0., 0., 0., 0.]], dtype=float32)

Write data to memmap array:

>>> fp[:] = data[:]
>>> fp
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

>>> fp.filename == path.abspath(filename)
True

Flushes memory changes to disk in order to read them back

>>> fp.flush()

Load the memmap and verify data was stored:

>>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
>>> newfp
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

Read-only memmap:

>>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
>>> fpr.flags.writeable
False

Copy-on-write memmap:

>>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
>>> fpc.flags.writeable
True

It's possible to assign to copy-on-write array, but values are only
written into the memory copy of the array, and not written to disk:

>>> fpc
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)
>>> fpc[0,:] = 0
>>> fpc
memmap([[  0.,   0.,   0.,   0.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

File on disk is unchanged:

>>> fpr
memmap([[  0.,   1.,   2.,   3.],
        [  4.,   5.,   6.,   7.],
        [  8.,   9.,  10.,  11.]], dtype=float32)

Offset into a memmap:

>>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
>>> fpo
memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)
````

