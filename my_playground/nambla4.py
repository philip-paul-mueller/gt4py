# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Implementation of the Nabla4 Stencil.
"""

import copy

from gt4py.next.common import NeighborTable
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.ffront.fbuiltins import Field
from gt4py.next.program_processors.runners import dace_fieldview as dace_backend
from gt4py.next.type_system import type_specifications as ts
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    KDim,
    Cell,
    Edge,
    IDim,
    JDim,
    MeshDescriptor,
    V2EDim,
    Vertex,
    simple_mesh,
    skip_value_mesh,
)
from typing import Sequence, Any
from functools import reduce
import numpy as np

import dace

wpfloat = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
SIZE_TYPE = ts.ScalarType(ts.ScalarKind.INT32)


def nabla4_np(
    nabv_norm: Field[[Edge, KDim], wpfloat],
    nabv_tang: Field[[Edge, KDim], wpfloat],
    z_nabla2_e: Field[[Edge, KDim], wpfloat],
    inv_vert_vert_length: Field[[Edge], wpfloat],
    inv_primal_edge_length: Field[[Edge], wpfloat],
    **kwargs,  # Allows to use the same call argument object as for the SDFG
) -> Field[[Edge, KDim], wpfloat]:
    N = nabv_norm - 2 * z_nabla2_e
    ell_v2 = inv_vert_vert_length**2
    N_ellv2 = N * ell_v2.reshape((-1, 1))

    T = nabv_tang - 2 * z_nabla2_e
    ell_e2 = inv_primal_edge_length**2
    T_elle2 = T * ell_e2.reshape((-1, 1))

    return 4 * (N_ellv2 + T_elle2)


def dace_strides(
    array: np.ndarray,
    name: None | str = None,
) -> tuple[int, ...] | dict[str, int]:
    if not hasattr(array, "strides"):
        return {}
    strides = array.strides
    if hasattr(array, "itemsize"):
        strides = tuple(stride // array.itemsize for stride in strides)
    if name is not None:
        strides = {f"__{name}_stride_{i}": stride for i, stride in enumerate(strides)}
    return strides


def dace_shape(
    array: np.ndarray,
    name: str,
) -> dict[str, int]:
    if not hasattr(array, "shape"):
        return {}
    return {f"__{name}_size_{i}": size for i, size in enumerate(array.shape)}


def make_syms(**kwargs: np.ndarray) -> dict[str, int]:
    SYMBS = {}
    for name, array in kwargs.items():
        SYMBS.update(**dace_shape(array, name))
        SYMBS.update(**dace_strides(array, name))
    return SYMBS


def build_nambla4_gtir():
    edge_k_domain = im.call("unstructured_domain")(
        im.call("named_range")(itir.AxisLiteral(value=Edge.value, kind=Edge.kind), 0, "num_edges"),
        im.call("named_range")(
            itir.AxisLiteral(value=KDim.value, kind=KDim.kind), 0, "num_k_levels"
        ),
    )

    num_edges = 27
    num_k_levels = 10

    EK_FTYPE = ts.FieldType(dims=[Edge, KDim], dtype=wpfloat)
    E_FTYPE = ts.FieldType(dims=[Edge], dtype=wpfloat)

    nabla4prog = itir.Program(
        id="nabla4_partial",
        function_definitions=[],
        params=[
            itir.Sym(id="nabv_norm", type=EK_FTYPE),
            itir.Sym(id="nabv_tang", type=EK_FTYPE),
            itir.Sym(id="z_nabla2_e", type=EK_FTYPE),
            itir.Sym(id="inv_vert_vert_length", type=E_FTYPE),
            itir.Sym(id="inv_primal_edge_length", type=E_FTYPE),
            itir.Sym(id="nab4", type=EK_FTYPE),
            itir.Sym(id="num_edges", type=SIZE_TYPE),
            itir.Sym(id="num_k_levels", type=SIZE_TYPE),
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.call(
                    im.call("as_fieldop")(
                        im.lambda_("NpT", "const_4")(
                            im.multiplies_(im.deref("NpT"), im.deref("const_4"))
                        ),
                        edge_k_domain,
                    )
                )(
                    # arg: `NpT`
                    im.call(
                        im.call("as_fieldop")(
                            im.lambda_("N_ell2", "T_ell2")(
                                im.plus(im.deref("N_ell2"), im.deref("T_ell2"))
                            ),
                            edge_k_domain,
                        )
                    )(
                        # arg: `N_ell2`
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("ell_v2", "N")(
                                    im.multiplies_(im.deref("N"), im.deref("ell_v2"))
                                ),
                                edge_k_domain,
                            )
                        )(
                            # arg: `ell_v2`
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("ell_v")(
                                        im.multiplies_(im.deref("ell_v"), im.deref("ell_v"))
                                    ),
                                    edge_k_domain,
                                )
                            )(
                                # arg: `ell_v`
                                "inv_vert_vert_length"
                            ),
                            # end arg: `ell_v2`
                            # arg: `N`
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("xn", "z_nabla2_e2")(
                                        im.minus(im.deref("xn"), im.deref("z_nabla2_e2"))
                                    ),
                                    edge_k_domain,
                                )
                            )(
                                # arg: `xn`
                                "nabv_norm",
                                # arg: `z_nabla2_e2`
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("z_nabla2_e", "const_2")(
                                            im.multiplies_(
                                                im.deref("z_nabla2_e"), im.deref("const_2")
                                            )
                                        ),
                                        edge_k_domain,
                                    )
                                )(
                                    # arg: `z_nabla2_e`
                                    "z_nabla2_e",
                                    # arg: `const_2`
                                    2.0,
                                ),
                                # end arg: `z_nabla2_e2`
                            ),
                            # end arg: `N`
                        ),
                        # end arg: `N_ell2`
                        # arg: `T_ell2`
                        im.call(
                            im.call("as_fieldop")(
                                im.lambda_("ell_e2", "T")(
                                    im.multiplies_(im.deref("T"), im.deref("ell_e2"))
                                ),
                                edge_k_domain,
                            )
                        )(
                            # arg: `ell_e2`
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("ell_e")(
                                        im.multiplies_(im.deref("ell_e"), im.deref("ell_e"))
                                    ),
                                    edge_k_domain,
                                )
                            )(
                                # arg: `ell_e`
                                "inv_primal_edge_length"
                            ),
                            # end arg: `ell_e2`
                            # arg: `T`
                            im.call(
                                im.call("as_fieldop")(
                                    im.lambda_("xt", "z_nabla2_e2")(
                                        im.minus(im.deref("xt"), im.deref("z_nabla2_e2"))
                                    ),
                                    edge_k_domain,
                                )
                            )(
                                # arg: `xt`
                                "nabv_tang",
                                # arg: `z_nabla2_e2`
                                im.call(
                                    im.call("as_fieldop")(
                                        im.lambda_("z_nabla2_e", "const_2")(
                                            im.multiplies_(
                                                im.deref("z_nabla2_e"), im.deref("const_2")
                                            )
                                        ),
                                        edge_k_domain,
                                    )
                                )(
                                    # arg: `z_nabla2_e`
                                    "z_nabla2_e",
                                    # arg: `const_2`
                                    2.0,
                                ),
                            ),
                            # end arg: `T`
                        ),
                        # end arg: `T_ell2`
                    ),
                    # end arg: `NpT`
                    # arg: `const_4`
                    4.0,
                ),
                domain=edge_k_domain,
                target=itir.SymRef(id="nab4"),
            )
        ],
    )

    offset_provider = {}

    nabv_norm = np.random.rand(num_edges, num_k_levels)
    nabv_tang = np.random.rand(num_edges, num_k_levels)
    z_nabla2_e = np.random.rand(num_edges, num_k_levels)
    inv_vert_vert_length = np.random.rand(num_edges)
    inv_primal_edge_length = np.random.rand(num_edges)
    nab4 = np.empty((num_edges, num_k_levels), dtype=nabv_norm.dtype)

    sdfg = dace_backend.build_sdfg_from_gtir(nabla4prog, offset_provider)

    call_args = dict(
        nabv_norm=nabv_norm,
        nabv_tang=nabv_tang,
        z_nabla2_e=z_nabla2_e,
        inv_vert_vert_length=inv_vert_vert_length,
        inv_primal_edge_length=inv_primal_edge_length,
        nab4=nab4,
        num_edges=num_edges,
        num_k_levels=num_k_levels,
    )
    SYMBS = make_syms(**call_args)

    sdfg(**call_args, **SYMBS)
    ref = nabla4_np(**call_args)

    assert np.allclose(ref, nab4)
    print(f"Test succeeded")


if "__main__" == __name__:
    build_nambla4_gtir()
