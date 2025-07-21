from __future__ import absolute_import

import functools
import numpy as np


def unpack(pyramid, backend='numpy'):

    backend = backend.lower()
    if backend == 'numpy':
        yield pyramid.lowpass
        yield pyramid.highpasses
        if pyramid.scales is not None:
            yield pyramid.scales
    elif backend == 'opencl':
        yield pyramid.cl_lowpass
        yield pyramid.cl_highpasses
        if pyramid.cl_scales is not None:
            yield pyramid.cl_scales
    elif backend == 'tf':
        yield pyramid.lowpass_op
        yield pyramid.highpasses_ops
        if pyramid.scales_ops is not None:
            yield pyramid.scales_ops


def drawedge(theta,r,w,N):

    # convert theta from degrees to radians
    thetar = np.array(theta * np.pi / 180)

    # Calculate image centre from given width
    imCentre = (np.array([N,N]).T - 1) / 2 + 1

    # Calculate values to subtract from the plane
    r = np.array([np.cos(thetar), np.sin(thetar)])*(-1) * (r - imCentre)

    # check width of raised cosine section
    w = np.maximum(1,w)

    ramp = np.arange(0,N) - (N+1)/2
    hgrad = np.sin(thetar)*(-1) * np.ones([N,1])
    vgrad = np.cos(thetar)*(-1) * np.ones([1,N])
    plane = ((hgrad * ramp) - r[0]) + ((ramp * vgrad).T - r[1])
    x = 0.5 + 0.5 * np.sin(np.minimum(np.maximum(
        plane*(np.pi/w), np.pi/(-2)), np.pi/2))

    return x


def drawcirc(r,w,du,dv,N):

    # check value of w to avoid dividing by zero
    w = np.maximum(w,1)

    # x plane
    x = np.ones([N,1]) * ((np.arange(0,N,1, dtype='float') -
                          (N+1) / 2 - dv) / r)

    # y vector
    y = (((np.arange(0,N,1, dtype='float') - (N+1) / 2 - du) / r) *
         np.ones([1,N])).T

    # Final circle image plane
    p = 0.5 + 0.5 * np.sin(np.minimum(np.maximum((
            np.exp(np.array([-0.5]) * (x**2 + y**2)).T - np.exp((-0.5))) * (r * 3 / w),  # noqa
        np.pi/(-2)), np.pi/2))
    return p


def asfarray(X):

    X = np.asanyarray(X)
    return np.asfarray(X, dtype=X.dtype)


def appropriate_complex_type_for(X):

    X = asfarray(X)

    if np.issubsctype(X.dtype, np.complex64) or \
            np.issubsctype(X.dtype, np.complex128):
        return X.dtype
    elif np.issubsctype(X.dtype, np.float32):
        return np.complex64
    elif np.issubsctype(X.dtype, np.float64):
        return np.complex128

    # God knows, err on the side of caution
    return np.complex128


def as_column_vector(v):

    v = np.atleast_2d(v)
    if v.shape[0] == 1:
        return v.T
    else:
        return v


def reflect(x, minx, maxx):

    x = np.asanyarray(x)
    rng = maxx - minx
    rng_by_2 = 2 * rng
    mod = np.fmod(x - minx, rng_by_2)
    normed_mod = np.where(mod < 0, mod + rng_by_2, mod)
    out = np.where(normed_mod >= rng, rng_by_2 - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)


def symm_pad_1d(l, m):

    xe = reflect(np.arange(-m, l+m, dtype='int32'), -0.5, l-0.5)
    return xe


def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer


def stacked_2d_matrix_vector_prod(mats, vecs):

    return np.einsum('...ij,...j->...i', mats, vecs)


def stacked_2d_vector_matrix_prod(vecs, mats):

    vecshape = np.array(vecs.shape + (1,))
    vecshape[-1:-3:-1] = vecshape[-2:]
    outshape = mats.shape[:-2] + (mats.shape[-1],)
    return stacked_2d_matrix_matrix_prod(vecs.reshape(vecshape), mats).reshape(outshape)  # noqa


def stacked_2d_matrix_matrix_prod(mats1, mats2):

    return np.einsum('...ij,...jk->...ik', mats1, mats2)
