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

import datetime
import logging
import os
from enum import Enum
from typing import Optional, Callable, List

import numpy as np
from distributed import Client, Future, TimeoutError
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.device import basic_device_noise_model
from qiskit.providers.models import BackendProperties

import lib_circuits as lib
from lib_experimental_utils import simulation, qasm_simulator, ibmqx4, create_direction_only_pass_manager, \
    create_experiment_qobj, get_gate_times, experiment, ibmqx2, ibmq_16_melbourne, create_optimize_only_pass_manager, \
    ibmq_ourense, ibmq_vigo

LOG = logging.getLogger(__name__)


def setup_logging():
    import lib_experimental_utils
    logging.basicConfig(format=logging.BASIC_FORMAT, level='WARN')
    logging.getLogger(lib_experimental_utils.__name__).setLevel('DEBUG')
    logging.getLogger(__name__).setLevel('DEBUG')


module_path = os.path.dirname(__file__)
client = Client(address='localhost:8786')  # type: Client


class BackendEnum(Enum):
    SIMULATOR = 0
    IBMQX2 = 1
    IBMQX4 = 2
    IBMQ_OURENSE = 3
    IBMQ_VIGO = 4


def update_files():
    client.upload_file("{}/lib_circuits.py".format(module_path))
    client.upload_file("{}/lib_experimental_utils.py".format(module_path))
    client.upload_file("{}/lib_experiment_setups.py".format(module_path))


def create_hadamard_simulation(backend_enum=BackendEnum.IBMQX4, instead_general_weights_use_hadamard=False, use_barriers=False, no_noise=False, use_dask=True):
    update_files()

    id = "sim_hadamard_{}".format(datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ"))

    circuit = lib.create_hadamard_circuit  # type: Callable[[List[float], float, Optional[dict]], QuantumCircuit]
    backend = ibmqx4  # type: Callable[[], BaseBackend]
    if backend_enum == BackendEnum.IBMQX2:
        circuit = lib.create_hadamard_circuit
        backend = ibmqx2
    elif backend_enum == BackendEnum.IBMQX4:
        circuit = lib.create_hadamard_circuit
        backend = ibmqx4
    elif backend_enum == BackendEnum.IBMQ_OURENSE:
        circuit = lib.create_hadamard_circuit_ourense
        backend = ibmq_ourense
    elif backend_enum == BackendEnum.IBMQ_VIGO:
        circuit = lib.create_hadamard_circuit_ourense
        backend = ibmq_vigo
    elif backend_enum == BackendEnum.SIMULATOR:
        circuit = lib.create_hadamard_circuit_ourense
        backend = qasm_simulator
    else:
        raise ValueError("Given backend not supported.")

    def calculation():
        setup_logging()

        if no_noise:
            noise_model = None
            device_properties = None  # type: Optional[BackendProperties]
            gate_times = []
        else:
            device_properties = backend().properties()  # type: Optional[BackendProperties]
            gate_times = get_gate_times(backend())

            noise_model = basic_device_noise_model(device_properties, gate_times=gate_times, temperature=0)  # type: Optional[NoiseModel]

        pass_manager = create_direction_only_pass_manager(backend())
        qobj, theta = create_experiment_qobj(
            factory=circuit,
            other_arguments={'use_barriers': use_barriers},
            weights=[] if instead_general_weights_use_hadamard else [1 / np.sqrt(2), 1 / np.sqrt(2)],
            theta_start=0.0,
            theta_end=2 * np.pi,
            theta_step=0.1,
            pm=pass_manager,
            device=qasm_simulator(),
            qobj_id=id,
            use_dask=use_dask)

        sim = simulation(id, qasm_simulator(), qobj=qobj, noise_model=noise_model)
        sim.set_theta(theta)
        sim.parameters['backend'] = backend().name()
        sim.parameters['use_barriers'] = use_barriers
        sim.parameters['circuit_factory'] = circuit.__name__
        sim.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard
        if noise_model is not None:
            sim.parameters['device_properties'] = device_properties.to_dict()
            # sim.parameters['gate_times'] = dict(gate_times)

        return sim

    if use_dask:
        future = client.submit(calculation)
        # noinspection PyTypeChecker
        client.publish_dataset(future, name=id)
        sim = None
    else:
        sim = calculation()

    return id if use_dask else sim


def create_hadamard_experiment_and_then_simulation(instead_ibmqx4_use_ibmqx2=False, instead_general_weights_use_hadamard=False, use_barriers=False, no_noise=False):
    update_files()

    id = "exp_sim_hadamard_{}".format(datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ"))

    backend = ibmqx2 if instead_ibmqx4_use_ibmqx2 else ibmqx4
    circuit = lib.create_hadamard_circuit
    pass_manager = create_direction_only_pass_manager(backend())

    def calculation():
        setup_logging()

        qobj, theta = create_experiment_qobj(
            factory=circuit,
            other_arguments={'use_barriers': use_barriers},
            weights=[] if instead_general_weights_use_hadamard else [1 / np.sqrt(2), 1 / np.sqrt(2)],
            theta_start=0.0,
            theta_end=2 * np.pi,
            theta_step=0.1,
            pm=pass_manager,
            device=qasm_simulator(),
            qobj_id=id,
            use_dask=True)

        # Experiment
        ex = experiment(id, backend(), qobj=qobj)
        ex.parameters['backend'] = backend().name()
        ex.parameters['use_barriers'] = use_barriers
        ex.parameters['circuit_factory'] = circuit.__name__
        ex.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard

        # Simulation
        if no_noise:
            noise_model = None
            device_properties = None  # type: Optional[BackendProperties]
            gate_times = []
        else:
            device_properties = backend().properties()  # type: Optional[BackendProperties]
            gate_times = get_gate_times(backend())
            noise_model = basic_device_noise_model(device_properties, gate_times=gate_times, temperature=0)  # type: Optional[NoiseModel]

        sim = simulation(id, qasm_simulator(), qobj=qobj, noise_model=noise_model)
        sim.set_theta(theta)
        sim.parameters['backend'] = backend().name()
        sim.parameters['use_barriers'] = use_barriers
        sim.parameters['circuit_factory'] = circuit.__name__
        sim.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard
        if noise_model is not None:
            sim.parameters['device_properties'] = device_properties.to_dict(),
        if noise_model is not None:
            sim.parameters['gate_times'] = gate_times

        LOG.info("Done with %s.", id)

        return ex, sim

    future = client.submit(calculation)
    # noinspection PyTypeChecker
    client.publish_dataset(future, name=id)
    return id


def create_regular_simulation(instead_ibmqx4_use_ibmqx2=False, instead_general_weights_use_hadamard=False, use_barriers=False, no_noise=False, readout_swap=None):
    update_files()

    id = "sim_regular_{}".format(datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ"))

    backend = ibmqx2 if instead_ibmqx4_use_ibmqx2 else ibmqx4
    circuit = lib.create_swap_test_old_circuit if instead_general_weights_use_hadamard else lib.create_swap_test_circuit

    def calculation():
        setup_logging()

        pass_manager = create_direction_only_pass_manager(backend())
        qobj, theta = create_experiment_qobj(
            factory=lib.create_swap_test_circuit,
            other_arguments={'use_barriers': True, 'readout_swap': readout_swap},
            weights=[1/np.sqrt(2), 1/np.sqrt(2)],
            theta_start=0.0,
            theta_end=2 * np.pi,
            theta_step=0.1,
            pm=pass_manager,
            device=qasm_simulator(),
            qobj_id=id,
            use_dask=True)

        # Simulation
        if no_noise:
            noise_model = None
            device_properties = None  # type: Optional[BackendProperties]
            gate_times = []
        else:
            device_properties = backend().properties()  # type: Optional[BackendProperties]
            gate_times = get_gate_times(backend())
            noise_model = basic_device_noise_model(device_properties, gate_times=gate_times,
                                                   temperature=0)  # type: Optional[NoiseModel]

        sim = simulation(id, qasm_simulator(), qobj=qobj, noise_model=noise_model)
        sim.set_theta(theta)
        sim.parameters['backend'] = backend().name()
        sim.parameters['use_barriers'] = use_barriers
        sim.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard
        sim.parameters['circuit_factory'] = circuit.__name__
        if noise_model is not None:
            sim.parameters['device_properties'] = device_properties.to_dict(),
        if noise_model is not None:
            sim.parameters['gate_times'] = gate_times

        return sim

    future = client.submit(calculation)
    # noinspection PyTypeChecker
    client.publish_dataset(future, name=id)
    return id


def create_regular_experiment_and_then_simulation(backend_enum=BackendEnum.IBMQX4,
                                                  instead_general_weights_use_hadamard=False, use_barriers=False,
                                                  no_noise=False, readout_swap=None, no_experiment=False,
                                                  dont_use_dask=False):
    # type: (BackendEnum, bool, bool, bool, bool, bool, bool) -> str
    update_files()

    id = "exp_sim_regular_{}".format(datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ"))

    def calculation():
        setup_logging()

        weights = [] if instead_general_weights_use_hadamard else [1 / np.sqrt(2), 1 / np.sqrt(2)]

        circuit = lib.create_swap_test_circuit  # type: Callable[[List[float], float, Optional[dict]], QuantumCircuit]
        backend = ibmqx4  # type: Callable[[], BaseBackend]
        if backend_enum == BackendEnum.IBMQX2:
            circuit = lib.create_swap_test_circuit
            backend = ibmqx2
        elif backend_enum == BackendEnum.IBMQX4:
            circuit = lib.create_swap_test_circuit
            backend = ibmqx4
        elif backend_enum == BackendEnum.IBMQ_OURENSE:
            circuit = lib.create_swap_test_circuit_ourense
            backend = ibmq_ourense
        elif backend_enum == BackendEnum.IBMQ_VIGO:
            circuit = lib.create_swap_test_circuit_ourense
            backend = ibmq_vigo
        else:
            raise ValueError("Given backend not supported.")

        pass_manager = create_direction_only_pass_manager(backend())
        qobj, theta = create_experiment_qobj(
            factory=circuit,
            other_arguments={'use_barriers': use_barriers, 'readout_swap': readout_swap},
            weights=weights,
            theta_start=0.0,
            theta_end=2 * np.pi,
            theta_step=0.1,
            pm=pass_manager,
            device=qasm_simulator(),
            qobj_id=id,
            use_dask=not dont_use_dask)

        if no_experiment:
            ex = None
        else:
            ex = experiment(id, backend(), qobj=qobj)
            ex.set_theta(theta)
            ex.parameters['backend'] = backend().name()
            ex.parameters['use_barriers'] = use_barriers
            ex.parameters['readout_swap'] = readout_swap
            ex.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard
            ex.parameters['circuit_factory'] = circuit.__name__

        if no_noise:
            noise_model = None
            device_properties = None  # type: Optional[BackendProperties]
            gate_times = []
        else:
            device_properties = backend().properties()  # type: Optional[BackendProperties]
            gate_times = get_gate_times(backend())
            noise_model = basic_device_noise_model(device_properties, gate_lengths=gate_times, temperature=0)  # type: Optional[NoiseModel]

        sim = simulation(id, qasm_simulator(), qobj=qobj, noise_model=noise_model)
        sim.parameters['backend'] = backend().name()
        sim.parameters['use_barriers'] = use_barriers
        sim.parameters['readout_swap'] = readout_swap
        sim.parameters['use_generic_weights_circuit'] = not instead_general_weights_use_hadamard
        sim.parameters['circuit_factory'] = circuit.__name__
        if noise_model is not None:
            sim.parameters['device_properties'] = device_properties.to_dict(),
        if noise_model is not None:
            sim.parameters['gate_times'] = gate_times

        sim.set_theta(theta)
        if no_experiment:
            ex = sim

        return ex, sim

    if dont_use_dask:
        calculation()
    else:
        experiment_future = client.submit(calculation)
        # noinspection PyTypeChecker
        client.publish_dataset(experiment_future, name=id)

    return id


def create_product_state_simulation(copies=1):
    update_files()

    id = "sim_product_state_{}_copies_{}".format(copies, datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ"))

    def calculation():
        setup_logging()

        # Try to use a noise model if we possible
        if copies == 1:
            device_properties = ibmqx4().properties()
            gate_times = get_gate_times(ibmqx4())
        elif copies <= 5:
            device_properties = ibmq_16_melbourne().properties()
            gate_times = get_gate_times(ibmq_16_melbourne())
        else:
            device_properties = None
            gate_times = None

        # If we don't have device properties we can't create a noise model
        if device_properties is None:
            noise_model = None  # type: Optional[NoiseModel]
        else:
            noise_model = basic_device_noise_model(device_properties, gate_times=gate_times,
                                                   temperature=0)  # type: Optional[NoiseModel]

        pass_manager = create_optimize_only_pass_manager(qasm_simulator())

        qobj, theta = create_experiment_qobj(
            factory=lib.create_product_state_n_copies_circuit,
            weights=[1 / np.sqrt(2), 1 / np.sqrt(2)],
            theta_start=0.0,
            theta_end=2 * np.pi,
            theta_step=0.1,
            pm=pass_manager,
            device=qasm_simulator(),
            qobj_id=id,
            use_dask=True,
            other_arguments={'copies': copies})
        sim = simulation(id, qasm_simulator(), qobj=qobj, noise_model=noise_model)
        sim.set_theta(theta)

        return sim

    future = client.submit(calculation)
    # noinspection PyTypeChecker
    client.publish_dataset(future, name=id)
    return id


def load_by_index(index=0, contains=None):
    datasets = list(reversed(sorted([d for d in client.list_datasets() if contains is None or contains in d])))

    if len(datasets) == 0:
        return None

    return load_by_id(datasets[index])


def load_by_id(id=None, delete=False):
    if id is None:
        datasets = list(reversed(sorted(client.list_datasets())))
        if len(datasets) == 0:
            return None
        else:
            id = datasets[0]

    future = client.get_dataset(id)  # type: Future

    try:
        result = future.result(timeout=2)

        if delete:
            client.unpublish_dataset(id)

        if isinstance(result, tuple):
            return list(result)

        if isinstance(result, list):
            return result

        return [result]

    except TimeoutError as tex:
        print(tex)
        return None


def get_ids():
    return list(reversed(sorted(client.list_datasets())))