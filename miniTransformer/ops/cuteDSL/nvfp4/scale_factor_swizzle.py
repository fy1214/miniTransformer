import cutlass
import cutlass.cute as cute

@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_ptr: cute.Pointer,
    sf_mma_ptr: cute.Pointer,
    mn: int,
    sf_k: int,
    l: int,
    mma_shape: tuple,
):
    mma_permute_order = (3, 4, 1, 5, 2, 0)
    permuted_shape = tuple(mma_shape[i] for i in mma_permute_order)
    cute_layout = cute.make_ordered_layout(permuted_shape, order=(2, 1, 4, 0, 3, 5))

    sf_ref_tensor = cute.make_tensor(
        sf_ref_ptr, cute.make_layout((mn, sf_k, l), stride=(sf_k, 1, mn * sf_k))
    )
    sf_mma_tensor = cute.make_tensor(sf_mma_ptr, cute_layout)

    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]
    pass