from simba.errors import DimensionError


def halve_matrix(mat):
    """
    Halve the dimensions of the given matrix, truncating where necessary.

    Throws `DimensionError` if matrix dimensions are not even.
    """
    if mat.shape[0] % 2 != 0 or mat.shape[1] % 2 != 0:
        raise DimensionError(f"Matrix dimensions not even: {mat.shape}")

    return mat.extract(range(mat.shape[0] // 2), range(mat.shape[1] // 2))
