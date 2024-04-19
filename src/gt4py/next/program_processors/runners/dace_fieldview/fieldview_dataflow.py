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


from typing import Optional, Tuple, TypeAlias

import dace


class FieldviewRegion:
    """Defines the dataflow scope of a fieldview expression.

    This class defines a region of the dataflow which represents a fieldview expression.
    It usually consists of a map scope, with a set of input nodes that traverse the entry map;
    a set of transient data nodes (aka temporaries) where the output memlets traversing the
    exit map will write to; and the compute nodes (tasklets) implementing the expression
    within the map scope.
    More than one fieldfiew region can exist within a state. In this case, the temporaies
    which are written to by one fieldview region will be inputs to the next region. Also,
    the set of access nodes `node_mapping` is shared among all fieldview regions within a state.

    We use this class as return type when we visit a fieldview expression. It can be extended
    with all informatiion needed to construct the dataflow graph.
    """

    Connection: TypeAlias = Tuple[dace.nodes.Node, Optional[str]]

    sdfg: dace.SDFG
    state: dace.SDFGState
    node_mapping: dict[str, dace.nodes.AccessNode]

    # ordered list of input/output data nodes used by the field operator being built in this dataflow region
    input_connections: list[Connection]
    output_connections: list[Connection]

    input_nodes: list[str]
    output_nodes: list[str]

    def __init__(
        self,
        current_sdfg: dace.SDFG,
        current_state: dace.SDFGState,
    ):
        self.sdfg = current_sdfg
        self.state = current_state
        self.node_mapping = {}
        self.input_nodes = []
        self.output_nodes = []
        self.input_connections = []
        self.output_connections = []

    def _add_node(self, data: str) -> dace.nodes.AccessNode:
        assert data in self.sdfg.arrays
        if data in self.node_mapping:
            node = self.node_mapping[data]
        else:
            node = self.state.add_access(data)
            self.node_mapping[data] = node
        return node

    def add_input_node(self, data: str) -> dace.nodes.AccessNode:
        self.input_nodes.append(data)
        return self._add_node(data)

    def add_output_node(self, data: str) -> dace.nodes.AccessNode:
        self.output_nodes.append(data)
        return self._add_node(data)

    def clone(self) -> "FieldviewRegion":
        ctx = FieldviewRegion(self.sdfg, self.state)
        ctx.node_mapping = self.node_mapping
        return ctx

    def tasklet_name(self) -> str:
        return f"{self.state.label}_tasklet"

    def var_name(self) -> str:
        return f"{self.state.label}_var"
