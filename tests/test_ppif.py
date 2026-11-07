"""Reference tests for the pyppif / pyppifcuda bindings.

Every check compares against a small NumPy (float64) implementation of the
paper's definitions, so a bug in either backend shows up as a mismatch.
The CUDA tests are skipped when pyppifcuda is not installed.

Run with: python -m pytest tests
"""
import numpy as np
import pytest

import pyppif

try:
    import pyppifcuda
except ImportError:
    pyppifcuda = None

DIM = 128
# The bindings compute in float32; squared distances are O(1).
ATOL = 1e-4

requires_cuda = pytest.mark.skipif(pyppifcuda is None, reason='pyppifcuda is not installed')


# ---------------------------------------------------------------------------
# NumPy references.
# ---------------------------------------------------------------------------

def ref_point_to_subspace(p, S):
    """Squared distance from point p to the affine subspace S (basis rows, then offset)."""
    S = S.astype(np.float64)
    basis, offset = S[:-1], S[-1]
    d = p.astype(np.float64) - offset
    return np.sum((d - basis.T @ (basis @ d)) ** 2)


def ref_subspace_to_subspace(A, B):
    """Squared distance between two affine subspaces.

    This is the part of (offset_B - offset_A) orthogonal to span(basis_A, basis_B).
    The SVD rank threshold keeps it exact when the subspaces share a direction.
    """
    A, B = A.astype(np.float64), B.astype(np.float64)
    X = np.concatenate([A[:-1], B[:-1]]).T
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    U = U[:, s > 1e-9]
    r = B[-1] - A[-1]
    return np.sum((r - U @ (U.T @ r)) ** 2)


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def unit_rows(rng, n):
    x = rng.standard_normal((n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def noisy_copies(rng, x, noise=0.1):
    y = x + noise * unit_rows(rng, x.shape[0])
    return (y / np.linalg.norm(y, axis=1, keepdims=True)).astype(np.float32)


def lift(alg, descriptors, subspace_dim, database, num_sub_databases=1, seed=0):
    if alg == 'random':
        return pyppif.random_lifting(descriptors, subspace_dim, seed=seed)
    if alg == 'adversarial':
        return pyppif.adversarial_lifting(
            descriptors, subspace_dim, database, num_sub_databases=num_sub_databases, seed=seed)
    if alg == 'hybrid':
        return pyppif.hybrid_lifting(
            descriptors, subspace_dim, database, num_sub_databases=num_sub_databases, seed=seed)
    raise ValueError(alg)


def num_adversarial_directions(alg, subspace_dim):
    return {'random': 0, 'adversarial': subspace_dim, 'hybrid': subspace_dim // 2}[alg]


def s2s_matcher(backend):
    return {'cpu': pyppif, 'cuda': pyppifcuda}[backend].subspace_to_subspace_exhaustive_matcher


def p2s_matcher(backend):
    return {'cpu': pyppif, 'cuda': pyppifcuda}[backend].point_to_subspace_exhaustive_matcher


BACKENDS = ['cpu', pytest.param('cuda', marks=requires_cuda)]
ALGORITHMS = ['random', 'adversarial', 'hybrid']


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def database(rng):
    # 16 sub-databases of 100 words.
    return unit_rows(rng, 1600)


# ---------------------------------------------------------------------------
# Lifting.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('alg', ALGORITHMS)
@pytest.mark.parametrize('subspace_dim', [2, 4])
def test_lifting_produces_orthonormal_subspaces_through_the_descriptor(rng, database, alg, subspace_dim):
    descriptors = unit_rows(rng, 50)
    lifted = lift(alg, descriptors, subspace_dim, database, num_sub_databases=16, seed=1)

    assert len(lifted) == len(descriptors)
    for d, S in zip(descriptors, lifted):
        assert S.shape == (subspace_dim + 1, DIM)
        assert S.dtype == np.float32
        basis = S[:-1].astype(np.float64)
        # Both matchers assume orthonormal bases.
        np.testing.assert_allclose(basis @ basis.T, np.eye(subspace_dim), atol=1e-5)
        # The original descriptor lies on its subspace (paper, Sec. 3).
        assert ref_point_to_subspace(d, S) < 1e-6


@pytest.mark.parametrize('alg', ALGORITHMS)
@pytest.mark.parametrize('subspace_dim', [2, 4])
def test_adversarial_words_come_from_a_single_sub_database(rng, database, alg, subspace_dim):
    num_sub_databases = 16
    sub_database_size = len(database) // num_sub_databases
    descriptors = unit_rows(rng, 20)
    lifted = lift(alg, descriptors, subspace_dim, database, num_sub_databases=num_sub_databases, seed=3)

    sub_databases_used = set()
    for S in lifted:
        # Database words used as adversarial directions lie exactly on the subspace.
        on_subspace = [i for i, w in enumerate(database) if ref_point_to_subspace(w, S) < 1e-6]
        assert len(on_subspace) == num_adversarial_directions(alg, subspace_dim)
        sub_databases_used.update(i // sub_database_size for i in on_subspace)
    # One sub-database is picked per call (i.e. per image).
    assert len(sub_databases_used) <= 1


@pytest.mark.parametrize('alg', ALGORITHMS)
def test_lifting_is_deterministic_given_the_seed(rng, database, alg):
    descriptors = unit_rows(rng, 20)
    a = lift(alg, descriptors, 4, database, num_sub_databases=16, seed=5)
    b = lift(alg, descriptors, 4, database, num_sub_databases=16, seed=5)
    c = lift(alg, descriptors, 4, database, num_sub_databases=16, seed=6)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not all(np.array_equal(x, z) for x, z in zip(a, c))


# ---------------------------------------------------------------------------
# Matching.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('alg', ALGORITHMS)
@pytest.mark.parametrize('subspace_dim', [2, 4])
def test_subspace_to_subspace_matches_reference(rng, database, backend, alg, subspace_dim):
    descriptors1 = unit_rows(rng, 30)
    descriptors2 = np.concatenate([noisy_copies(rng, descriptors1[:10]), unit_rows(rng, 15)])
    lifted1 = lift(alg, descriptors1, subspace_dim, database, num_sub_databases=16, seed=1)
    lifted2 = lift(alg, descriptors2, subspace_dim, database, num_sub_databases=16, seed=2)

    distances = s2s_matcher(backend)(lifted1, lifted2)

    assert distances.shape == (30, 25)
    expected = np.array([[ref_subspace_to_subspace(A, B) for B in lifted2] for A in lifted1])
    np.testing.assert_allclose(distances, expected, atol=ATOL)
    # The lifted distance is upper bounded by the original one (paper, Sec. 3.1).
    raw = np.sum((descriptors1[:, None] - descriptors2[None]) ** 2, axis=-1)
    assert np.all(distances <= raw + ATOL)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('subspace_dim', [2, 4, 8, 16])
def test_point_to_subspace_matches_reference(rng, database, backend, subspace_dim):
    points = unit_rows(rng, 30)
    lifted = lift('hybrid', unit_rows(rng, 20), subspace_dim, database, num_sub_databases=16, seed=4)

    distances = p2s_matcher(backend)(points, lifted)

    assert distances.shape == (30, 20)
    expected = np.array([[ref_point_to_subspace(p, S) for S in lifted] for p in points])
    np.testing.assert_allclose(distances, expected, atol=ATOL)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('subspace_dim', [2, 4])
@pytest.mark.parametrize('shared_words', ['one', 'all'])
def test_subspaces_sharing_database_words(rng, backend, subspace_dim, shared_words):
    # Lifting two sets of features with the same database words makes their
    # subspaces intersect, so the true distance is 0. Sharing two or more words
    # also makes the subspaces share a direction, i.e. N = I - M M^T is singular.
    # A database of exactly subspace_dim words forces every feature to use all of them.
    words1 = unit_rows(rng, subspace_dim)
    if shared_words == 'one':
        words2 = np.concatenate([words1[:1], unit_rows(rng, subspace_dim - 1)])
    else:
        words2 = words1
    lifted1 = lift('adversarial', unit_rows(rng, 20), subspace_dim, words1, seed=1)
    lifted2 = lift('adversarial', unit_rows(rng, 20), subspace_dim, words2, seed=2)

    distances = s2s_matcher(backend)(lifted1, lifted2)

    assert not np.any(np.isnan(distances))
    np.testing.assert_allclose(distances, 0, atol=ATOL)


@requires_cuda
@pytest.mark.parametrize('subspace_dim', [2, 4])
def test_cpu_and_cuda_agree(rng, database, subspace_dim):
    lifted1 = lift('hybrid', unit_rows(rng, 40), subspace_dim, database, num_sub_databases=16, seed=1)
    lifted2 = lift('hybrid', unit_rows(rng, 30), subspace_dim, database, num_sub_databases=16, seed=2)
    np.testing.assert_allclose(
        pyppifcuda.subspace_to_subspace_exhaustive_matcher(lifted1, lifted2),
        pyppif.subspace_to_subspace_exhaustive_matcher(lifted1, lifted2),
        atol=ATOL)
    points = unit_rows(rng, 25)
    np.testing.assert_allclose(
        pyppifcuda.point_to_subspace_exhaustive_matcher(points, lifted2),
        pyppif.point_to_subspace_exhaustive_matcher(points, lifted2),
        atol=ATOL)


@pytest.mark.parametrize('backend', BACKENDS)
def test_empty_inputs(rng, database, backend):
    lifted = lift('random', unit_rows(rng, 5), 2, database)
    assert s2s_matcher(backend)(lifted, []).shape == (5, 0)
    assert s2s_matcher(backend)([], lifted).shape == (0, 5)
    assert p2s_matcher(backend)(unit_rows(rng, 3), []).shape == (3, 0)
