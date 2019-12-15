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
import time
from typing import Union, Dict, Optional, List, Callable, Tuple, Any

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.compiler
from distributed import get_client, Client
from qiskit import QuantumCircuit
from qiskit.providers import BaseBackend, JobStatus, BaseJob, JobError
from qiskit.providers.aer import AerJob
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.ibmq import IBMQBackend, IBMQJob
from qiskit.providers.ibmq.accountprovider import AccountProvider
from qiskit.providers.models.backendproperties import BackendProperties, Gate
from qiskit.qobj import Qobj
from qiskit.result import Result
from qiskit.result.models import ExperimentResult
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.transpiler.passes import CXDirection, Unroller, Optimize1qGates
from scipy.optimize import minimize

LOG = logging.getLogger(__name__)

if qiskit.IBMQ.active_account() is None or len(qiskit.IBMQ.active_account()) == 0:
    qiskit.IBMQ.load_account()

active_account = qiskit.IBMQ.active_account()
if active_account is None:
    raise ValueError("No active accounts there!")
token = active_account["token"]
url = active_account["url"]


# Devices
def provider():
    # type: () -> AccountProvider
    return qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')


def ibmqx2():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmqx2')  # type: IBMQBackend


def ibmqx4():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmqx4')  # type: IBMQBackend


def ibmq_ourense():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmq_ourense')  # type: IBMQBackend


def ibmq_vigo():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmq_vigo')  # type: IBMQBackend


def ibmq_16_melbourne():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmq_16_melbourne')  # type: IBMQBackend


def qasm_simulator():
    # type: () -> AerBackend
    return qiskit.Aer.get_backend('qasm_simulator')  # type: AerBackend


def ibmq_simulator():
    # type: () -> IBMQBackend
    return provider().get_backend('ibmq_qasm_simulator')  # type: IBMQBackend


class RunningExperiment:
    def __init__(self, external_id, date, qobj, job, noise_model=None):
        # type: (str, datetime.datetime, Qobj, BaseJob, Optional[NoiseModel]) -> None
        self.external_id = external_id
        self.qobj = qobj
        self.date = date
        self.job = job
        self.noise_model = noise_model

    def is_done(self):
        status = self.job.status()
        if status is JobStatus.ERROR:
            raise Exception("Job Error occurred!")
        return status is JobStatus.DONE


def compare_plot(theta, classification, classification_label=None, compare_classification=None, compare_classification_label=None):
        plt.figure(figsize=(10, 7))
        theta = theta if len(theta) == len(classification) else range(len(classification))

        prefix_label = '{} '.format(classification_label) if classification_label is not None else ''
        plt.scatter(x=[xx for xx, p in zip(theta, classification) if p >= 0],
                    y=[p for p in classification if p >= 0],
                    label=prefix_label + '$\\tilde{y} = 0$',
                    c='red',
                    marker='^',
                    linewidths=1.0)
        plt.scatter(x=[xx for xx, p in zip(theta, classification) if p < 0],
                    y=[p for p in classification if p < 0],
                    label=prefix_label + '$\\tilde{y} = 1$',
                    c='white',
                    marker='^',
                    linewidths=1.0, edgecolors='red')
        y_lim_lower = min(classification) - 0.1
        y_lim_upper = max(classification) + 0.1

        if compare_classification is not None and len(compare_classification) == len(classification):
            prefix_label = '{} '.format(compare_classification_label) if compare_classification_label is not None else ''
            plt.scatter(x=[xx for xx, p in zip(theta, compare_classification) if p >= 0],
                        y=[p for p in compare_classification if p >= 0],
                        label=prefix_label + '$\\tilde{y} = 0$',
                        c='blue',
                        marker='s',
                        linewidths=1.0)
            plt.scatter(x=[xx for xx, p in zip(theta, compare_classification) if p < 0],
                        y=[p for p in compare_classification if p < 0],
                        label=prefix_label + '$\\tilde{y} = 1$',
                        c='white',
                        marker='s',
                        linewidths=1.0, edgecolors='blue')
            y_lim_lower = min(min(compare_classification) - 0.1, y_lim_lower)
            y_lim_upper = max(max(compare_classification) + 0.1, y_lim_upper)

        plt.legend(fontsize=17)
        plt.xlabel("$\\theta$ (rad.)", fontsize=22)
        plt.ylabel("$\\langle \\sigma_z^{(a)} \\sigma_z^{(l)} \\rangle$", fontsize=22)
        plt.tick_params(labelsize=22)

        plt.ylim((y_lim_lower, y_lim_upper))


class FinishedExperiment:

    def __init__(self, backend_version, date, qobj, job_id, backend_name, status, results, noise_model, external_id, theta, parameters):
        # type: (str, datetime.datetime, Qobj, str, str, JobStatus, List[ExperimentResult], Optional[NoiseModel], str, List[float], Dict[str, Any]) -> None
        self.parameters = parameters  # type: Dict[str, Any]
        self.theta = theta  # type: List[float]
        self.external_id = external_id  # type: str
        self.noise_model = noise_model  # type: Optional[NoiseModel]
        self.results = results  # type: List[ExperimentResult]
        self.status = status  # type: JobStatus
        self.backend_name = backend_name  # type: str
        self.job_id = job_id  # type: str
        self.qobj = qobj  # type: Qobj
        self.date = date  # type: datetime.datetime
        self.backend_version = backend_version  # type: str

    def to_result(self):
        return Result(backend_name=self.backend_name, backend_version=self.backend_version, qobj_id=self.qobj.qobj_id,
                        job_id=self.job_id, success=self.status is JobStatus.DONE, results=self.results)

    def get_classification(self):
        # type: () -> List[float]
        result = self.to_result()
        classification = [extract_classification(result.get_counts(i)) for i in range(len(result.results))]  # type: List[float]
        return classification

    def to_dict(self):

        return {
            'qobj': self.qobj.to_dict(),
            'date': self.date.isoformat(),
            'job_id': self.job_id,
            'backend_name': self.backend_name,
            'backend_version': self.backend_version,
            'job_status': self.status.name,
            'results': [e.to_dict() for e in self.results] if self.status is JobStatus.DONE else [],
            'classification': self.get_classification() if self.status is JobStatus.DONE else None,
            'noise_model': self.noise_model.as_dict() if self.noise_model is not None else None,
            'parameters': self.parameters,
            'external_id': self.external_id
        }

    def analyze(self):
        # type: () -> AnalyzeResult
        if 'weights' in self.parameters:
            weights = self.parameters['weights']
        else:
            weights = [0.5, 0.5]
        return analyze_solution(self, weights[0], weights[1])

    @staticmethod
    def from_running_experiment(running_experiment):
        # type: (RunningExperiment) -> FinishedExperiment
        while not running_experiment.is_done():
            time.sleep(1)
        external_id = running_experiment.external_id
        date = running_experiment.date
        qobj = running_experiment.qobj
        job_id = running_experiment.job.job_id()
        status = running_experiment.job.status()  # type: JobStatus
        results = running_experiment.job.result().results
        backend: BaseBackend = running_experiment.job.backend()
        backend_version = backend.configuration().backend_version
        backend_name = backend.name()
        noise_model = running_experiment.noise_model
        theta = []
        parameters = {}

        return FinishedExperiment(
            backend_name=backend_name,
            backend_version=backend_version,
            date=date,
            qobj=qobj,
            job_id=job_id,
            status=status,
            results=results,
            noise_model=noise_model,
            external_id=external_id,
            theta=theta,
            parameters=parameters
        )

    @staticmethod
    def from_dict(dict):
        return FinishedExperiment(
            backend_name=dict.get('backend_name', ''),
            backend_version=dict.get('backend_version', None),
            date=dateutil.parser.parse(dict['date']) if 'date' in dict else None,
            qobj=Qobj.from_dict(dict.get('qobj', {})),
            job_id=dict.get('job_id', ''),
            status=JobStatus[dict['job_status']] if 'job_status' in dict else JobStatus.INITIALIZING,
            results=[ExperimentResult.from_dict(d) for d in dict.get('results', [])],
            noise_model=NoiseModel.from_dict(dict['noise_model']) if 'noise_model' in dict and dict['noise_model'] is not None else None,
            external_id=dict.get('external_id', None),
            theta=dict.get('theta', []),
            parameters=dict.get('parameters', [])
        )

    @staticmethod
    def from_data(date=None, qobj=None, backend=None, job_id=None, noise_model=None, external_id=None, theta=None):
        # type: (str, dict, str, str, dict, str, list) -> FinishedExperiment
        """
        We expect a dict with a qobj, job_id, backend name and optionally a noise model.
        When we have a Aer backend the simulation is redone to have the results.
        If the backend is a IBMQ then it is retrieved from the API.

        Thus it can take some time until this call ends.
        :param date: a string
        :param qobj: a dictionary
        :param job_id: a string
        :param noise_model: a dictionary
        :return: the Finished Experiment
        """

        if theta is None:
            theta = []
        if 'ibmq' in backend and job_id is not None:
            backend_obj = provider().get_backend(backend)  # type: IBMQBackend
            job = backend_obj.retrieve_job(job_id)  # type: IBMQJob
            qobj = job.qobj().to_dict()
            qobj = Qobj.from_dict(qobj)
            date = job.creation_date()
        elif date is not None and qobj is not None and backend is not None:
            if isinstance(qobj, dict):
                qobj = Qobj.from_dict(qobj)
            backend_obj = qiskit.Aer.get_backend(backend)  # type: AerBackend
            job = backend_obj.run(qobj=qobj, noise_model=noise_model)  # type: AerJob
            job_id = job.job_id()
        else:
            raise ValueError("Either use a IBMQ backend with a job_id or provide a date, qobj, backend.")

        if noise_model is not None:
            noise_model = NoiseModel.from_dict(noise_model)
        if isinstance(date, str):
            date = dateutil.parser.parse(date)  # type: datetime.datetime

        external_id = 'job_{}'.format(date.strftime("%Y%m%dT%H%M%SZ")) if external_id is None else external_id
        running_experiment = RunningExperiment(date=date, qobj=qobj, noise_model=noise_model, job=job, external_id=external_id)

        while not running_experiment.is_done():
            time.sleep(10)
            LOG.info("Simulation job {} is not done yet.".format(job_id))

        fin_ex = FinishedExperiment.from_running_experiment(running_experiment)
        fin_ex.set_theta(theta)

        return fin_ex

    def show_plot(self, classification_label=None, compare_classification=None, compare_classification_label=None):
        compare_plot(self.theta, self.get_classification(), classification_label=classification_label,
                     compare_classification=compare_classification,
                     compare_classification_label=compare_classification_label)

    def set_theta(self, theta):
        # type: (List[float]) -> None
        self.theta = theta


class AnalyzeResult(object):

    def __init__(self, amplitude_factor, phase_shift, approximate_w_1, accuracy):
        # type: (float, float, float, float) -> None
        self.accuracy = accuracy
        self.approximate_w_1 = approximate_w_1
        self.phase_shift = phase_shift
        self.amplitude_factor = amplitude_factor

    def __str__(self):
        from sympy import nsimplify
        return "amplitude dampening: {:.4}, shift: {} pi, approx. w_1: {:.4}, accuracy: {:.4}".format(
            self.amplitude_factor,
            nsimplify(self.phase_shift / np.pi, tolerance=0.1),
            self.approximate_w_1,
        self.accuracy)

    def __repr__(self):
        return str(self)


ibmqx2_120_gf_time = {
    'CX0_1': 190,
    'CX0_2': 190,
    'CX1_2': 250,
    'CX3_2': 250,
    'CX3_4': 150,
    'CX4_2': 240,
}


ibmqx4_120_gf_time = {
    'CX1_0': 110,
    'CX2_0': 152,
    'CX2_1': 200,
    'CX3_2': 250,
    'CX3_4': 150,
    'CX4_2': 400,
}

ibmq_16_melbourne_110_gf_time = {
    'CX1_0': 239,
    'CX1_2': 174,
    'CX2_3': 261,
    'CX4_3': 266,
    'CX5_4': 300,
    'CX5_6': 300,
    'CX7_8': 348,
    'CX9_8': 348,
    'CX9_10': 300,
    'CX11_10': 261,
    'CX11_12': 217,
    'CX13_12': 300,
    'CX13_1': 652,
    'CX12_2': 1043,
    'CX11_3': 286,
    'CX4_10': 261,
    'CX5_9': 348,
    'CX6_8': 348
}


gf_times = {
    'ibmqx2': ibmqx2_120_gf_time,
    'ibmqx4': ibmqx4_120_gf_time,
    'ibmq_16_melbourne': ibmq_16_melbourne_110_gf_time
}


def get_gate_time(gate, qubit_data, gf_data):
    # type: (Gate, List[dict], Dict[str, int]) -> int
    if gate.gate == 'u1':
        return 0
    if gate.gate == 'u2':
        buffer = qubit_data[gate.qubits[0]].get('buffer', {}).get('value', None)
        gd_gatetime = qubit_data[gate.qubits[0]].get('gateTime', {}).get('value', None)
        return buffer + gd_gatetime
    if gate.gate == 'u3':
        buffer = qubit_data[gate.qubits[0]].get('buffer', {}).get('value', None)
        gd_gatetime = qubit_data[gate.qubits[0]].get('gateTime', {}).get('value', None)
        return 2*(buffer + gd_gatetime)
    if gate.gate == 'cx':
        gf_gatetime = gf_data[gate.name.upper()]
        buffer_0 = qubit_data[gate.qubits[0]].get('buffer', {}).get('value', None)
        buffer_1 = qubit_data[gate.qubits[1]].get('buffer', {}).get('value', None)
        gd_gatetime_0 = qubit_data[gate.qubits[0]].get('gateTime', {}).get('value', None)
        gd_gatetime_1 = qubit_data[gate.qubits[1]].get('gateTime', {}).get('value', None)
        return max(buffer_0 + gd_gatetime_0, buffer_1 + gd_gatetime_1) \
               + 2*(gf_gatetime + max(buffer_0, buffer_1)) \
               + buffer_0 + gd_gatetime_0
    return 0


def get_gate_times(backend):
    # type: (IBMQBackend) -> List[Tuple[str, List[int], int]]
    device_properties = backend.properties()
    gates = device_properties.gates #  type: List[Gate]
    gate_times = [(g.gate, g.qubits, [p.value for p in g.parameters if p.name == 'gate_length'][0]) for g in gates]

    return gate_times


def create_direction_only_pass_manager(device):
    # type: (BaseBackend) -> PassManager

    LOG.info("Creating direction-only PassManager for {}".format(device))

    cp = CouplingMap(couplinglist=device.configuration().coupling_map)
    basis = device.configuration().basis_gates

    pm = PassManager()
    pm.append(Unroller(basis=basis))
    # noinspection PyTypeChecker
    pm.append(CXDirection(coupling_map=cp))
    pm.append(Optimize1qGates())
    return pm


def create_optimize_only_pass_manager(device):
    # type: (BaseBackend) -> PassManager
    basis = device.configuration().basis_gates

    pm = PassManager()
    pm.append(Unroller(basis=basis))
    pm.append(Optimize1qGates())
    return pm


def retrieve_compiled_circuit(weights, theta, factory, pm, device, other_arguments):
    # type: (List[float], float, Callable[[List[float], float, dict], QuantumCircuit], PassManager, BaseBackend, dict) -> QuantumCircuit
    return qiskit.compiler.transpile(factory(weights, theta, **other_arguments if other_arguments is not None else {}), pass_manager=pm, backend=device)


def create_experiment_qobj(factory, weights, theta_start, theta_end, theta_step, pm, device, qobj_id=None, use_dask=False, other_arguments=None):
    # type: (Callable[[List[float], float, dict], QuantumCircuit], List[float], float, float, float, PassManager, BaseBackend, Optional[str], bool, Optional[dict]) -> Tuple[Qobj, List[float]]

    LOG.info("Creating Qobj with {}".format(
        {
            'factory': factory,
            'weights': weights,
            'theta_start': theta_start,
            'theta_end': theta_end,
            'theta_step': theta_step,
            'pm': str(pm),
            'device': str(device),
            'qobj_id': qobj_id,
            'use_dask': use_dask,
            'other_arguments': other_arguments
        }
    ))

    r = np.arange(theta_start, theta_end, theta_step)
    if use_dask:
        client = get_client()  # type: Client
        futures = [client.submit(retrieve_compiled_circuit, weights, theta, factory, pm, device, other_arguments) for theta in r]
        circuits = client.gather(futures)
    else:
        circuits = [retrieve_compiled_circuit(weights, theta, factory, pm, device, other_arguments) for theta in r]
    LOG.debug(len(r) * 5)
    # noinspection PyTypeChecker
    qobj = qiskit.compiler.assemble(circuits, backend_name=qasm_simulator().name(), shots=8192, max_credits=len(r) * 5, qobj_id=qobj_id)
    return qobj, list(r.tolist())


def create_experiment_list_qobj(factory, weights, theta_start, theta_end, theta_step, pm, device, qobj_id=None, use_dask=False, other_arguments=None):
    # type: (Callable[[List[float], float], QuantumCircuit], List[float], float, float, float, PassManager, BaseBackend, Optional[str], bool, Optional[dict]) -> List[Qobj]
    r = np.arange(theta_start, theta_end, theta_step)
    if use_dask:
        client = get_client()  # type: Client
        futures = [client.submit(retrieve_compiled_circuit, theta, factory, pm, device, other_arguments) for theta in r]
        circuits = client.gather(futures)
    else:
        circuits = [retrieve_compiled_circuit(weights, theta, factory, pm, device, other_arguments) for theta in r]
    LOG.debug(len(r) * 5)
    if use_dask:
        client = get_client()  # type: Client
        qobjs = client.gather([client.submit(qiskit.compiler.assemble, c, backend_name=qasm_simulator().name(), shots=8192, max_credits=len(r) * 5, qobj_id=qobj_id)
                 for c in circuits])
    else:
        # noinspection PyTypeChecker
        qobjs = [qiskit.compiler.assemble(c, backend_name=qasm_simulator().name(), shots=8192, max_credits=len(r) * 5, qobj_id=qobj_id) for c in circuits]
    return qobjs


def extract_classification(counts):
    # type: (Dict[str, int]) -> float
    shots = sum(counts.values())
    return (counts.get('00', 0) - counts.get('01', 0) - counts.get('10', 0) + counts.get('11', 0)) / float(shots)


ibmqx_4_properties_instance = BackendProperties.from_dict({'backend_name': 'ibmqx4',
 'backend_version': '1.0.0',
 'gates': [{'gate': 'u1',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0}],
   'qubits': [0]},
  {'gate': 'u2',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0009443532335046134}],
   'qubits': [0]},
  {'gate': 'u3',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0018887064670092268}],
   'qubits': [0]},
  {'gate': 'u1',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0}],
   'qubits': [1]},
  {'gate': 'u2',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0012019552727863259}],
   'qubits': [1]},
  {'gate': 'u3',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0024039105455726517}],
   'qubits': [1]},
  {'gate': 'u1',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0}],
   'qubits': [2]},
  {'gate': 'u2',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0012019552727863259}],
   'qubits': [2]},
  {'gate': 'u3',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0024039105455726517}],
   'qubits': [2]},
  {'gate': 'u1',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0}],
   'qubits': [3]},
  {'gate': 'u2',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0013737021608475342}],
   'qubits': [3]},
  {'gate': 'u3',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0027474043216950683}],
   'qubits': [3]},
  {'gate': 'u1',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.0}],
   'qubits': [4]},
  {'gate': 'u2',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.001803112096824766}],
   'qubits': [4]},
  {'gate': 'u3',
   'parameters': [{'date': '2019-05-08T09:57:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.003606224193649532}],
   'qubits': [4]},
  {'gate': 'cx',
   'name': 'CX1_0',
   'parameters': [{'date': '2019-05-08T01:27:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.024311890455604945}],
   'qubits': [1, 0]},
  {'gate': 'cx',
   'name': 'CX2_0',
   'parameters': [{'date': '2019-05-08T01:32:39+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.023484363587478657}],
   'qubits': [2, 0]},
  {'gate': 'cx',
   'name': 'CX2_1',
   'parameters': [{'date': '2019-05-08T01:38:20+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.04885221406150694}],
   'qubits': [2, 1]},
  {'gate': 'cx',
   'name': 'CX3_2',
   'parameters': [{'date': '2019-05-08T01:44:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.06682678733530181}],
   'qubits': [3, 2]},
  {'gate': 'cx',
   'name': 'CX3_4',
   'parameters': [{'date': '2019-05-08T01:50:07+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.05217118636435464}],
   'qubits': [3, 4]},
  {'gate': 'cx',
   'name': 'CX4_2',
   'parameters': [{'date': '2019-05-08T01:56:04+00:00',
     'name': 'gate_error',
     'unit': '',
     'value': 0.06446497941268642}],
   'qubits': [4, 2]}],
 'general': [],
 'last_update_date': '2019-05-08T01:56:04+00:00',
 'qconsole': False,
 'qubits': [[{'date': '2019-05-08T01:16:56+00:00',
    'name': 'T1',
    'unit': 'µs',
    'value': 43.21767480545737},
   {'date': '2019-05-08T01:17:40+00:00',
    'name': 'T2',
    'unit': 'µs',
    'value': 19.77368032971812},
   {'date': '2019-05-08T01:56:04+00:00',
    'name': 'frequency',
    'unit': 'GHz',
    'value': 5.246576101635769},
   {'date': '2019-05-08T01:16:37+00:00',
    'name': 'readout_error',
    'unit': '',
    'value': 0.08650000000000002}],
  [{'date': '2019-05-08T01:16:56+00:00',
    'name': 'T1',
    'unit': 'µs',
    'value': 43.87997000828745},
   {'date': '2019-05-08T01:18:27+00:00',
    'name': 'T2',
    'unit': 'µs',
    'value': 11.390521028550571},
   {'date': '2019-05-08T01:56:04+00:00',
    'name': 'frequency',
    'unit': 'GHz',
    'value': 5.298309751315148},
   {'date': '2019-05-08T01:16:37+00:00',
    'name': 'readout_error',
    'unit': '',
    'value': 0.07999999999999996}],
  [{'date': '2019-05-07T09:14:18+00:00',
    'name': 'T1',
    'unit': 'µs',
    'value': 48.97128225850014},
   {'date': '2019-05-08T01:19:07+00:00',
    'name': 'T2',
    'unit': 'µs',
    'value': 31.06845465651204},
   {'date': '2019-05-08T01:56:04+00:00',
    'name': 'frequency',
    'unit': 'GHz',
    'value': 5.3383288291854765},
   {'date': '2019-05-08T01:16:37+00:00',
    'name': 'readout_error',
    'unit': '',
    'value': 0.038250000000000006}],
  [{'date': '2019-05-08T01:16:56+00:00',
    'name': 'T1',
    'unit': 'µs',
    'value': 38.30486582843196},
   {'date': '2019-05-08T01:18:27+00:00',
    'name': 'T2',
    'unit': 'µs',
    'value': 32.35546811356613},
   {'date': '2019-05-08T01:56:04+00:00',
    'name': 'frequency',
    'unit': 'GHz',
    'value': 5.426109336844823},
   {'date': '2019-05-08T01:16:37+00:00',
    'name': 'readout_error',
    'unit': '',
    'value': 0.35675}],
  [{'date': '2019-05-08T01:16:56+00:00',
    'name': 'T1',
    'unit': 'µs',
    'value': 36.02606265575505},
   {'date': '2019-05-07T09:15:02+00:00',
    'name': 'T2',
    'unit': 'µs',
    'value': 4.461644223370699},
   {'date': '2019-05-08T01:56:04+00:00',
    'name': 'frequency',
    'unit': 'GHz',
    'value': 5.174501299220437},
   {'date': '2019-05-08T01:16:37+00:00',
    'name': 'readout_error',
    'unit': '',
    'value': 0.2715000000000001}]]})


def properties_dict(T1=45, T2=20, gate_error=0.001, cx_error=0.05, readout_error=0.1, gate_time_single=60, gate_time_cx=240):
    # type: (int, int, float, float, float, Optional[int], Optional[int]) -> Dict[str, Union[str, list]]
    properties = {
        'backend_name': 'ibmqx4',
        'backend_version': '1.0.0',
        'gates': [],
        'general': [],
        'last_update_date': '2019-04-26T09:53:20+00:00',
        'qubits': []
    }

    coupling_map = [[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]]
    frequencies = [5.246495339914302, 5.298304321478395, 5.33833077451951, 5.426128480932129, 5.174479097262167]

    # CX gates
    for cp in coupling_map:
        cx_gate = {'gate': 'cx',
                   'name': 'CX{}_{}'.format(cp[0], cp[1]),
                   'parameters': [{'date': '2019-04-26T09:53:20+00:00',
                                   'name': 'gate_error',
                                   'unit': '',
                                   'value': cx_error}],
                   'qubits': cp}

        if gate_time_cx is not None:
            cx_gate["parameters"].append({'date': '2019-04-27T01:56:34+00:00',
                                          'name': 'gate_time',
                                          'unit': 'ns',
                                          'value': gate_time_cx})
        properties["gates"].append(cx_gate)

    # U1, U2, U3 and qubits
    for qubit, freq in zip(range(5), frequencies):

        for u in range(1, 4):
            gate = {'gate': 'u{}'.format(u),
                    'parameters': [{'date': '2019-04-27T01:56:34+00:00',
                                    'name': 'gate_error',
                                    'unit': '',
                                    'value': (u - 1) * gate_error}],
                    'qubits': [qubit]}
            if gate_time_single is not None:
                gate['parameters'].append(
                    {'date': '2019-04-27T01:56:34+00:00',
                     'name': 'gate_time',
                     'unit': 'ns',
                     'value': (u - 1) * gate_time_single}
                )
            properties["gates"].append(gate)

        qubit = [{'date': '2019-04-26T09:14:38+00:00',
                  'name': 'T1',
                  'unit': 'µs',
                  'value': T1},
                 {'date': '2019-04-26T09:15:23+00:00',
                  'name': 'T2',
                  'unit': 'µs',
                  'value': T2},
                 {'date': '2019-04-26T09:53:20+00:00',
                  'name': 'frequency',
                  'unit': 'GHz',
                  'value': freq},
                 {'date': '2019-04-26T09:14:20+00:00',
                  'name': 'readout_error',
                  'unit': '',
                  'value': readout_error}]
        properties['qubits'].append(qubit)

    return properties


def plot(theta, sim_classification, classification):
    # type: (List[float], List[float], List[float]) -> plt

    plt.figure(figsize=(10,7))

    plt.scatter(x=[xx for xx, p in zip(theta, sim_classification) if p >= 0], y=[p for p in sim_classification if p >= 0],
                label='simulation w/ noise $\\tilde{y} = 0$',
                c='red',
                marker='^',
                linewidths=1.0)
    plt.scatter(x=[xx for xx, p in zip(theta, sim_classification) if p < 0], y=[p for p in sim_classification if p < 0],
                label='simulation w/ noise $\\tilde{y} = 1$',
                c='white',
                marker='^',
                linewidths=1.0, edgecolors='red')

    plt.scatter(x=[xx for xx, p in zip(theta, classification) if p >= 0], y=[p for p in classification if p >= 0],
                label='real $\\tilde{y} = 0$',
                c='blue',
                marker='o',
                linewidths=1.0)
    plt.scatter(x=[xx for xx, p in zip(theta, classification) if p < 0], y=[p for p in classification if p < 0],
                label='real $\\tilde{y} = 1$',
                c='white',
                marker='o',
                linewidths=1.0, edgecolors='blue')

    plt.legend(fontsize=17)
    plt.xlabel("$\\theta$ (rad.)", fontsize=22)
    plt.ylabel("$\\langle \\sigma_z \\sigma_z \\rangle$", fontsize=22)
    plt.tick_params(labelsize=22)
    plt.ylim((-0.3,0.3))
    return plt
    # plt.savefig("{}.pdf".format(output_pdf_file))


def save(directory, experiment=None, simulation=None):
    # type: (str, Optional[FinishedExperiment], Optional[FinishedExperiment]) -> None

    if experiment is None and simulation is None:
        return

    id = experiment.external_id if experiment is not None else simulation.external_id

    LOG.info("Saving to %s/%s.py", directory, id)

    with open(os.path.join(directory, id + ".py"), 'w') as file:
        content = {}

        if experiment is not None:
            content['experiment'] = experiment.to_dict()

        if simulation is not None:
            content['simulation'] = simulation.to_dict()

        # When printed out naively a lot of indentation is missing
        # it is nicer to have this.
        import pprint
        import io
        output = io.StringIO()
        pprint.pprint(content, stream=output)

        # As we have numpy arrays in the data, we need to import the function _code:`asarray` with the name `array`
        # It is a bit of a hack but makes it smooth afterwards.
        file.writelines(['from numpy import asarray as array\n', 'result = '])
        file.write(output.getvalue())

# Experiments


def experiment(external_id, device_backend, qobj):
    # type: (str, IBMQBackend, Qobj) -> FinishedExperiment
    # noinspection PyTypeChecker
    status = device_backend.status()
    LOG.info("Executing experiment with id {} on backend {} (status: {})".format(external_id, device_backend, status.status_msg))

    if not status.operational:
        raise ValueError("Backend is not operational.")

    experiment_job = device_backend.run(qobj)
    job_id = experiment_job.job_id()
    ex = RunningExperiment(external_id, datetime.datetime.now(), qobj, experiment_job)

    is_done = False
    while not is_done:
        try:
            is_done = ex.is_done()
            LOG.info("Job {} is not done yet: pos. {}.".format(job_id, experiment_job.queue_position()))
        except JobError as error:
            LOG.info("RETRYING... an error occurred while checking for job status: {}".format(error))
            time.sleep(30)
        time.sleep(30)

    LOG.info("Job {} is done.".format(job_id))
    return FinishedExperiment.from_running_experiment(ex)


# Simulations

def simulation(external_id, simulation_backend, qobj, noise_model=None):
    # type: (str, AerBackend, Qobj, Optional[NoiseModel]) -> FinishedExperiment
    LOG.info("Executing simulation with id {} on backend {} with noise {}".format(
        external_id, simulation_backend, noise_model is not None))

    simulation_job = simulation_backend.run(qobj, noise_model=noise_model)
    job_id = simulation_job.job_id()
    sim = RunningExperiment(external_id, datetime.datetime.now(), qobj, simulation_job, noise_model=noise_model)

    while not sim.is_done():
        time.sleep(10)
        LOG.info("Simulation job {} is not done yet.".format(job_id))

    LOG.info("Simulation job {} is done.".format(job_id))
    return FinishedExperiment.from_running_experiment(sim)


def theory_expectation(w_1, w_2):
    def inner(x):
        return w_1 * np.sin(x/2 + np.pi/4)**2 - w_2 * np.cos(x/2 + np.pi/4)**2
    return inner


def plot2(experiment, simulation, w_1, w_2, plot_factor_theory=1.0):
    # type: (FinishedExperiment, FinishedExperiment, float, float, float) -> plt

    experiment.show_plot(compare_classification=simulation.get_classification(),
                         classification_label='exp.',
                         compare_classification_label='sim.')
    finer_theta = np.arange(0, 2 * np.pi, 1e-3)
    t_cl = [theory_expectation(w_1, w_2)(t) for t in finer_theta]
    plt.plot([xx for xx, p in zip(finer_theta, t_cl) if p > 1e-6],
             [plot_factor_theory * p for p in t_cl if p > 1e-6],
             c="black",
             label=('${} \\cdot$'.format(plot_factor_theory) if plot_factor_theory < 1.0 else '') + 'theory $\\tilde{y} = 0$',
             linestyle='-')
    plt.plot([xx for xx, p in zip(finer_theta, t_cl) if p < -1e-6],
             [plot_factor_theory * p for p in t_cl if p < -1e-6],
             c="black",
             label=('${} \\cdot$'.format(plot_factor_theory) if plot_factor_theory < 1.0 else '') + 'theory $\\tilde{y} = 1$',
             linestyle=':')
    plt.legend(fontsize=18, ncol=3, columnspacing=1, mode='expend', bbox_to_anchor=(1.0, 1.21), frameon=False)
    plt.ylim((-(plot_factor_theory + 0.1) / 2, (plot_factor_theory + 0.1) / 2))
    return plt


def analyze_solution(finished_experiment, w_1, w_2):
    # type: (FinishedExperiment, float, float) -> AnalyzeResult

    def mse(classification, theta):
        classification = np.asarray(classification)

        def inner(x):
            a, vartheta, w_1 = x
            reference = np.asarray([
                a * theory_expectation(w_1=w_1, w_2=1 - w_1)(t - vartheta) for t in theta
            ])
            return np.sqrt(sum(np.power(classification - reference, 2)))

        return inner

    fun = mse(finished_experiment.get_classification(), finished_experiment.theta)
    x_0 = np.asarray([1.0, 0, 0])
    result = minimize(fun, x_0)

    [a, vartheta, approx_w_1] = result.x

    theory_classification = [theory_expectation(w_1, w_2)(t) for t in finished_experiment.theta]
    accuracy = sum([1 if np.sign(ce * ct) > 0 else 0
                    for ce, ct in zip(finished_experiment.get_classification(), theory_classification)])
    accuracy = accuracy / len(finished_experiment.theta)

    return AnalyzeResult(a, vartheta, approx_w_1, accuracy)
