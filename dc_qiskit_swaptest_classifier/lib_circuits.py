# -*- coding: utf-8 -*-

# Copyright 2019 Carsten Blank
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cmath
import logging
from typing import Dict, Optional, List

import math
import numpy as np
import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.extensions import TdgGate, TGate
from qiskit.extensions.standard.barrier import barrier
from qiskit.extensions.standard.ccx import ccx
from qiskit.extensions.standard.ch import ch
from qiskit.extensions.standard.crz import crz
from qiskit.extensions.standard.cswap import cswap
from qiskit.extensions.standard.cu1 import cu1
from qiskit.extensions.standard.cu3 import cu3
from qiskit.extensions.standard.cx import cx, CnotGate
from qiskit.extensions.standard.cz import cz
from qiskit.extensions.standard.h import h, HGate
from qiskit.extensions.standard.rx import rx
from qiskit.extensions.standard.ry import ry
from qiskit.extensions.standard.rz import rz
from qiskit.extensions.standard.s import s
from qiskit.extensions.standard.swap import swap, SwapGate
from qiskit.extensions.standard.x import x

LOG = logging.getLogger(__name__)


def create_swap_test_old_circuit(index_state, theta, use_barriers=False, readout_swap=None):
    q = qiskit.QuantumRegister(5, "q")
    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(q, c, name="improvement")

    # Index on q_0
    h(qc, q[0])
    if use_barriers: barrier(qc)

    # Conditionally exite x_1 on data q_2 (center!)
    h(qc, q[2])
    if use_barriers: barrier(qc)
    rz(qc, math.pi, q[2]).inverse()
    if use_barriers: barrier(qc)
    s(qc, q[2])
    if use_barriers: barrier(qc)
    cz(qc, q[0], q[2])
    if use_barriers: barrier(qc)

    # Label y_1
    cx(qc, q[0], q[1])
    if use_barriers: barrier(qc)

    # Ancilla Superposition
    h(qc, q[4])
    if use_barriers: barrier(qc)

    # Unknown data
    #     standard.rx(qc, theta - 0.2*math.pi, q[3])
    rx(qc, theta, q[3])
    if use_barriers: barrier(qc)

    # c-SWAP!!!
    cswap(qc, q[4], q[2], q[3])
    if use_barriers: barrier(qc)

    # Hadamard on ancilla q_4
    h(qc, q[4])

    # Measure on ancilla q_4 and label q_1
    if readout_swap is not None:
        barrier(qc)
        for i in range(q.size):
            j = readout_swap.get(i, i)
            if i != j:
                swap(qc, q[i], q[j])
    else:
        readout_swap = {}

    barrier(qc)
    m1 = readout_swap.get(4, 4)
    m2 = readout_swap.get(1, 1)
    qiskit.circuit.measure.measure(qc, q[m1], c[0])
    qiskit.circuit.measure.measure(qc, q[m2], c[1])

    return qc


def compute_rotation(index_state):

    if len(index_state) != 2:
        return None, None

    index_state = np.asarray(index_state)

    if abs(np.linalg.norm(index_state)) < 1e-6:
        return None, None

    index_state = index_state / np.linalg.norm(index_state)

    if abs(index_state[0] - index_state[1]) < 1e-6:
        return None, None

    a_1 = abs(index_state[0])
    w_1 = cmath.phase(index_state[0])
    a_2 = abs(index_state[1])
    w_2 = cmath.phase(index_state[1])

    alpha_z = w_2 - w_1
    alpha_y = 2 * np.arcsin(abs(a_2) / np.sqrt(a_2 ** 2 + a_1 ** 2))

    return alpha_y, alpha_z


def create_swap_test_circuit(index_state, theta, **kwargs):
    # type: (List[float], float, Optional[dict]) -> QuantumCircuit
    """

    :param index_state:
    :param theta:
    :param kwargs: use_barriers (bool) and readout_swap (Dict[int, int])
    :return:
    """
    use_barriers = kwargs.get('use_barriers', False)
    readout_swap = kwargs.get('readout_swap', None)

    q = qiskit.QuantumRegister(5, "q")
    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(q, c, name="improvement")

    # Index on q_0
    alpha_y, _ = compute_rotation(index_state)
    if alpha_y is None:
        h(qc, q[0])
    else:
        ry(qc, -alpha_y, q[0]).inverse()
    if use_barriers: barrier(qc)

    # Conditionally exite x_1 on data q_2 (center!)
    h(qc, q[2])
    if use_barriers: barrier(qc)
    rz(qc, math.pi, q[2]).inverse()
    if use_barriers: barrier(qc)
    s(qc, q[2])
    if use_barriers: barrier(qc)
    cz(qc, q[0], q[2])
    if use_barriers: barrier(qc)

    # Label y_1
    cx(qc, q[0], q[1])
    if use_barriers: barrier(qc)

    # Ancilla Superposition
    h(qc, q[4])
    if use_barriers: barrier(qc)

    # Unknown data
    rx(qc, theta, q[3])
    if use_barriers: barrier(qc)

    # c-SWAP!!!
    # standard.barrier(qc)
    cswap(qc, q[4], q[2], q[3])
    if use_barriers: barrier(qc)

    # Hadamard on ancilla q_4
    h(qc, q[4])
    if use_barriers: barrier(qc)

    # Measure on ancilla q_4 and label q_1
    if readout_swap is not None:
        barrier(qc)
        for i in range(q.size):
            j = readout_swap.get(i, i)
            if i != j:
                swap(qc, q[i], q[j])
    else:
        readout_swap = {}

    barrier(qc)
    m1 = readout_swap.get(4, 4)
    m2 = readout_swap.get(1, 1)
    qiskit.circuit.measure.measure(qc, q[m1], c[0])
    qiskit.circuit.measure.measure(qc, q[m2], c[1])

    return qc


class Ourense_ToffoliGate(Gate):
    """
    Toffoli gate for ourense: a-c-b: q_0 <- a, q_1 <- c, q_2 <- b.

    a,c and b,c must be connected form the beginning.

    The qubits b,c are swapped at the end.
    """

    def __init__(self):
        """Create new Toffoli gate."""
        super().__init__("o_ccx", 3, [])

    def _define(self):
        """
        gate ccx a,b,c
        {
            h c;
            cx b,c;
            tdg c;
            cx a,c;
            t c;
            cx b,c;
            tdg c;
            cx a,c;
            t b;
            t c;
            h c;
            swap b,c;
            # p(a) = a, p(b) = c, p(c) = b
            cx p(a),p(b);
            t p(a);
            tdg p(b);
            cx p(a),p(b);
        }
        """
        definition = []
        q = QuantumRegister(3, "q")
        a, b, c = q[0], q[1], q[2]
        pi = {a: a, b: c, c: b}
        rule = [
            (HGate(), [c], []),
            (CnotGate(), [b, c], []),
            (TdgGate(), [c], []),
            (CnotGate(), [a, c], []),
            (TGate(), [c], []),
            (CnotGate(), [b, c], []),
            (TdgGate(), [c], []),
            (CnotGate(), [a, c], []),
            (TGate(), [b], []),
            (TGate(), [c], []),
            (HGate(), [c], []),
            # Swap here: b / c
            (SwapGate(), [b, c], []),
            # Swapped!
            (CnotGate(), [pi[a], pi[b]], []),
            (TGate(), [pi[a]], []),
            (TdgGate(), [pi[b]], []),
            (CnotGate(), [pi[a], pi[b]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return Ourense_ToffoliGate()  # self-inverse

    def to_matrix(self):
        """Return a Numpy.array for the Toffoli gate."""
        return np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0]], dtype=complex)


class Ourense_Fredkin(Gate):
    """Fredkin gate. It swaps gates 2,3 (b,c) after it is done."""

    def __init__(self):
        """Create new Fredkin gate."""
        super().__init__("o_cswap", 3, [])

    def _define(self):
        """
        gate cswap a,b,c
        { cx c,b;
          ccx a,b,c;
          # now it is swapped: b,c
          cx pi(c),pi(b);
        }
        """
        definition = []
        q = QuantumRegister(3, "q")
        rule = [
            (CnotGate(), [q[2], q[1]], []),
            (Ourense_ToffoliGate(), [q[0], q[1], q[2]], []),
            (CnotGate(), [q[1], q[2]], [])
        ]
        for inst in rule:
            definition.append(inst)
        self.definition = definition

    def inverse(self):
        """Invert this gate."""
        return Ourense_Fredkin()  # self-inverse


def create_swap_test_circuit_ourense(index_state, theta, **kwargs):
    # type: (List[float], float, Optional[dict]) -> QuantumCircuit
    """

    :param index_state:
    :param theta:
    :param kwargs: use_barriers (bool) and readout_swap (Dict[int, int])
    :return:
    """
    use_barriers = kwargs.get('use_barriers', False)
    readout_swap = kwargs.get('readout_swap', None)

    q = qiskit.QuantumRegister(5, "q")
    qb_a, qb_d, qb_in, qb_m, qb_l = (q[0], q[1], q[2], q[3], q[4])
    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(q, c, name="swap_test_ourense")

    # Index on q_0
    alpha_y, _ = compute_rotation(index_state)
    if alpha_y is None:
        h(qc, qb_m)
    else:
        ry(qc, -alpha_y, qb_m).inverse()
    if use_barriers: barrier(qc)

    # Conditionally exite x_1 on data q_2 (center!)
    h(qc, qb_d)
    if use_barriers: barrier(qc)
    rz(qc, math.pi, qb_d).inverse()
    if use_barriers: barrier(qc)
    s(qc, qb_d)
    if use_barriers: barrier(qc)
    cz(qc, qb_m, qb_d)
    if use_barriers: barrier(qc)

    # Label y_1
    cx(qc, qb_m, qb_l)
    if use_barriers: barrier(qc)

    # Unknown data
    rx(qc, theta, qb_in)
    if use_barriers: barrier(qc)

    # Swap-Test itself
    # Hadamard on ancilla
    h(qc, qb_a)
    if use_barriers: barrier(qc)

    # c-SWAP!!!
    qc.append(Ourense_Fredkin(), [qb_a, qb_in, qb_d], [])
    if use_barriers: barrier(qc)

    # Hadamard on ancilla
    h(qc, qb_a)
    if use_barriers: barrier(qc)

    # Measure on ancilla and label
    if readout_swap is not None:
        barrier(qc)
        for i in range(q.size):
            j = readout_swap.get(i, i)
            if i != j:
                swap(qc, q[i], q[j])
    else:
        readout_swap = {}

    barrier(qc)
    readout_swap_qb = dict(map(lambda k,v: (q[k], q[v]), readout_swap))  # type: Dict[QuantumRegister, QuantumRegister]
    m1 = readout_swap_qb.get(qb_a, qb_a)
    m2 = readout_swap_qb.get(qb_l, qb_l)
    qiskit.circuit.measure.measure(qc, m1, c[0])
    qiskit.circuit.measure.measure(qc, m2, c[1])

    return qc


def create_hadamard_circuit_ourense(index_state, theta, **kwargs):
    # type: (List[float], float, Optional[dict]) -> QuantumCircuit
    """

    :param index_state:
    :param theta:
    :param kwargs: use_barriers (bool) and readout_swap (Dict[int, int])
    :return:
    """
    use_barriers = kwargs.get('use_barriers', False)
    readout_swap = kwargs.get('readout_swap', None)

    q = qiskit.QuantumRegister(4, "q")
    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(q, c, name="hadmard-classifier")

    q_m = q[2]
    q_a = q[0]
    q_d = q[1]
    q_l = q[3]

    # Index on q_0
    alpha_y, _ = compute_rotation(index_state)
    if alpha_y is None:
        h(qc, q_m)
    else:
        rx(qc, -alpha_y, q_m).inverse()
    if use_barriers: barrier(qc)

    # Ancilla Superposition
    h(qc, q_a)
    if use_barriers: barrier(qc)

    # Test Data
    cu3(qc, theta - 0.0 * math.pi, -math.pi / 2, math.pi / 2, q_a, q_d)
    if use_barriers: barrier(qc)

    # Training Data
    ## Conditionally excite x_1 on data q_2 (center!)
    x(qc, q_a)
    if use_barriers: barrier(qc)
    ch(qc, q_a, q_d)
    if use_barriers: barrier(qc)
    crz(qc, math.pi + 0.0 * math.pi, q_a, q_d).inverse()
    if use_barriers: barrier(qc)
    cu1(qc, math.pi / 2 - 0.0 * math.pi, q_a, q_d)
    if use_barriers: barrier(qc)
    ## 2-Controlled Z-Gate
    h(qc, q_d)
    if use_barriers: barrier(qc)
    ### Logical Swap on q_m & q_d
    # qc.append(Ourense_ToffoliGate(), [q_a, q_m, q_d], [])
    ccx(qc, q_a, q_m, q_d)
    if use_barriers: barrier(qc)
    # h(qc, q_m) # q_d -> q_m swapped
    h(qc, q_d) # q_d -> q_m swapped
    if use_barriers: barrier(qc)

    # Label y_1
    # cx(qc, q_d, q_l) # q_m -> q_d swapped
    cx(qc, q_m, q_l) # q_m -> q_d swapped
    if use_barriers: barrier(qc)

    # Hadamard on ancilla
    h(qc, q_a)
    if use_barriers: barrier(qc)

    # Measure on ancilla and label
    if readout_swap is not None:
        barrier(qc)
        for i in range(q.size):
            j = readout_swap.get(i, i)
            if i != j:
                swap(qc, q[i], q[j])
    else:
        readout_swap = {}

    barrier(qc)
    readout_swap_qb = dict(map(lambda k, v: (q[k], q[v]), readout_swap))  # type: Dict[QuantumRegister, QuantumRegister]
    m1 = readout_swap_qb.get(q_a, q_a)
    m2 = readout_swap_qb.get(q_l, q_l)
    qiskit.circuit.measure.measure(qc, m1, c[0])
    qiskit.circuit.measure.measure(qc, m2, c[1])

    return qc


def create_product_state_n_copies_circuit(index_state, theta, copies=1, use_barriers=False):
    a = qiskit.QuantumRegister(1, "a")
    index = qiskit.QuantumRegister(1, "m")
    d = qiskit.QuantumRegister(copies, "d")
    label = qiskit.QuantumRegister(1, "l")
    inp = qiskit.QuantumRegister(copies, "in")

    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(a, index, d, label, inp, c, name="improvement")

    # Index on q_0
    alpha_y, _ = compute_rotation(index_state)
    if alpha_y is None:
        h(qc, index)
    else:
        ry(qc, -alpha_y, index).inverse()
    if use_barriers: barrier(qc)

    # Unknown data
    for copy in range(copies):
        rx(qc, theta, inp[copy])
        if use_barriers: barrier(qc)

    # Conditionally exite x_1 on data q_2 (center!)
    for copy in range(copies):
        h(qc, d[copy])
        if use_barriers: barrier(qc)
        rz(qc, math.pi, d[copy]).inverse()
        if use_barriers: barrier(qc)
        s(qc, d[copy])
        if use_barriers: barrier(qc)
        cz(qc, index, d[copy])
        if use_barriers: barrier(qc)

    # Label y_1
    cx(qc, index, label)
    if use_barriers: barrier(qc)

    barrier(qc)
    # Ancilla Superposition
    h(qc, a)
    if use_barriers: barrier(qc)

    # c-SWAP!!!

    for copy in range(copies):
        cswap(qc, a[0], d[copy], inp[copy])
        if use_barriers: barrier(qc)

    # Hadamard on ancilla
    h(qc, a)
    if use_barriers: barrier(qc)

    # Measure on ancilla and label
    barrier(qc)
    qiskit.circuit.measure.measure(qc, a[0], c[0])
    qiskit.circuit.measure.measure(qc, label[0], c[1])

    return qc


def create_hadamard_circuit(index_state, theta, use_barriers=False, readout_swap=None):
    initial_layout = {'a': 1, 'm': 2, 'd': 0, 'l': 3, '': 4}

    q = qiskit.QuantumRegister(4, "q")
    c = qiskit.ClassicalRegister(2, "c")
    qc = qiskit.QuantumCircuit(q, c, name="hadmard-classifier")

    # Index on q_0
    alpha_y, _ = compute_rotation(index_state)
    if alpha_y is None:
        h(qc, q[initial_layout['m']])
    else:
        ry(qc, -alpha_y, q[initial_layout['m']]).inverse()
    if use_barriers: barrier(qc)

    # Ancilla Superposition
    h(qc, q[initial_layout['a']])
    if use_barriers: barrier(qc)

    # Unknown (test) data
    cu3(qc, theta - 0.0 * math.pi, -math.pi / 2, math.pi / 2, q[initial_layout['a']],
                 q[initial_layout['d']])
    if use_barriers: barrier(qc)

    # Conditionally exite x_1 on data q_2 (center!)
    x(qc, q[initial_layout['a']])
    if use_barriers: barrier(qc)
    ch(qc, q[initial_layout['a']], q[initial_layout['d']])
    if use_barriers: barrier(qc)
    crz(qc, math.pi + 0.0 * math.pi, q[initial_layout['a']], q[initial_layout['d']]).inverse()
    if use_barriers: barrier(qc)
    cu1(qc, math.pi / 2 - 0.0 * math.pi, q[initial_layout['a']], q[initial_layout['d']])
    if use_barriers: barrier(qc)

    h(qc, q[initial_layout['d']])
    if use_barriers: barrier(qc)
    ccx(qc, q[initial_layout['a']], q[initial_layout['m']], q[initial_layout['d']])
    if use_barriers: barrier(qc)
    h(qc, q[initial_layout['d']])
    if use_barriers: barrier(qc)
    x(qc, q[initial_layout['a']])
    if use_barriers: barrier(qc)

    # Label y_1
    cx(qc, q[initial_layout['m']], q[initial_layout['l']])
    if use_barriers: barrier(qc)

    # Hadamard on ancilla
    h(qc, q[initial_layout['a']])
    if use_barriers: barrier(qc)

    # Measure on ancilla q_4 and label q_1
    if readout_swap is not None:
        barrier(qc)
        for i in range(q.size):
            j = readout_swap.get(i, i)
            if i != j:
                swap(qc, q[i], q[j])
    else:
        readout_swap = {}

    barrier(qc)
    m1 = readout_swap.get(initial_layout['a'], initial_layout['a'])
    m2 = readout_swap.get(initial_layout['l'], initial_layout['l'])
    qiskit.circuit.measure.measure(qc, q[m1], c[0])
    qiskit.circuit.measure.measure(qc, q[m2], c[1])

    return qc
