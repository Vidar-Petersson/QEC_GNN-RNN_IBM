# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" The following code has been taken and modified from the qiskit-qec library https://github.com/qiskit-community/qiskit-qec """

from typing import List, Optional

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

class RepetitionCodeCircuit:
    """RepetitionCodeCircuit class."""

    def __init__(
        self,
        d: int,
        T: int,
        xbasis: bool = False,
        resets: bool = False,
        delay: Optional[int] = None,
        barriers: bool = False,
        noise_angle: float = 0,
    ):
        """
        Creates the circuits corresponding to a logical 0 and 1 encoded
        using a repetition code.

        Implementation of a distance d repetition code, implemented over
        T syndrome measurement rounds.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement.
            xbasis (bool): Whether to use the X basis to use for encoding (Z basis used by default).
            resets (bool): Whether to include a reset gate after mid-circuit measurements.
            delay (float): Time (in dt) to delay after mid-circuit measurements (and delay).
            barriers (bool): Whether to include barriers between different sections of the code.


        Additional information:
            No measurements are added to the circuit if `T=0`. Otherwise
            `T` rounds are added, followed by measurement of the code
            qubits (corresponding to a logical measurement and final
            syndrome measurement round).
        """

        super().__init__()

        self.n = d
        self.d = d
        self.T = 0

        self.code_qubit = QuantumRegister(d, "code_qubit")
        self.link_qubit = QuantumRegister((d - 1), "link_qubit")
        self.qubit_registers = {"code_qubit", "link_qubit"}

        self.link_bits = []
        self.code_bit = ClassicalRegister(d, "code_bit")

        self.circuit = {}
        for log in ["0", "1"]:
            self.circuit[log] = QuantumCircuit(self.link_qubit, self.code_qubit, name=log)

        self._xbasis = xbasis
        self._resets = resets
        self._barriers = barriers
        self._noise_angle = noise_angle

        self._preparation()

        delay = delay or 0

        for _ in range(T - 1):
            self.syndrome_measurement(delay=delay)
            if noise_angle > 0.0:
                self.inject_error(alpha=noise_angle)

        if T != 0:
            self.syndrome_measurement(final=True)
            self.readout()

        self.gauge_ops = [[j, j + 1] for j in range(self.d - 1)]
        self.measured_logical = [[0], [self.d - 1]]
        self.basis = "x"

        self.resets = resets
        self.delay = delay
        self.base = "0"

    def get_circuit_list(self) -> List[QuantumCircuit]:
        """Returns circuit list.

        circuit_list: self.circuit as a list, with
        circuit_list[0] = circuit['0']
        circuit_list[1] = circuit['1']
        """
        circuit_list = [self.circuit[log] for log in ["0", "1"]]
        return circuit_list

    def x(self, logs=("0", "1"), barrier=False):
        """Applies a logical x to the circuits for the given logical values.

        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the start.
        """
        barrier = barrier or self._barriers
        for log in logs:
            if barrier and (log == "1" or self._xbasis):
                self.circuit[log].barrier()
            if self._xbasis:
                self.circuit[log].z(self.code_qubit)
            else:
                self.circuit[log].x(self.code_qubit)

    def _preparation(self):
        """Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        """
        for log in ["0", "1"]:
            if self._xbasis:
                self.circuit[log].h(self.code_qubit)
        self.x(["1"])

    def syndrome_measurement(self, final: bool = False, barrier: bool = False, delay: int = 0):
        """Application of a syndrome measurement round.

        Args:
            final (bool): Whether to disregard the reset (if applicable) due to this
            being the final syndrome measurement round.
            barrier (bool): Boolean denoting whether to include a barrier at the start.
            delay (float): Time (in dt) to delay after mid-circuit measurements (and delay).
        """
        barrier = barrier or self._barriers

        self.link_bits.append(ClassicalRegister((self.d - 1), "round_" + str(self.T) + "_link_bit"))

        for log in ["0", "1"]:
            self.circuit[log].add_register(self.link_bits[-1])

            # entangling gates
            if barrier:
                self.circuit[log].barrier()
            if self._xbasis:
                self.circuit[log].h(self.link_qubit)
            for j in range(self.d - 1):
                if self._xbasis:
                    self.circuit[log].cx(self.link_qubit[j], self.code_qubit[j])
                else:
                    self.circuit[log].cx(self.code_qubit[j], self.link_qubit[j])
            for j in range(self.d - 1):
                if self._xbasis:
                    self.circuit[log].cx(self.link_qubit[j], self.code_qubit[j + 1])
                else:
                    self.circuit[log].cx(self.code_qubit[j + 1], self.link_qubit[j])
            if self._xbasis:
                self.circuit[log].h(self.link_qubit)

            # measurement
            if barrier:
                self.circuit[log].barrier()
            for j in range(self.d - 1):
                self.circuit[log].measure(self.link_qubit[j], self.link_bits[self.T][j])

            # resets
            if self._resets and not final:
                if barrier:
                    self.circuit[log].barrier()
                for j in range(self.d - 1):
                    self.circuit[log].reset(self.link_qubit[j])

            # delay
            if delay > 0 and not final:
                if barrier:
                    self.circuit[log].barrier()
                for j in range(self.d - 1):
                    self.circuit[log].delay(delay, self.link_qubit[j])

        self.T += 1

    def inject_error(self, alpha=0):
        """
        Injects a coherent rotation error about a fixed axis (Z-axis) on each code qubit,
        matching the protocol from the reference paper (Fig. 2a).

        :param alpha: Rotation angle in radians to apply (deterministic).
        """
        # Apply RZ(alpha) on each code qubit for both logical circuits
        for log in ["0", "1"]:
            for q in range(self.d):
                self.circuit[log].rz(alpha, self.code_qubit[q])

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """
        for log in ["0", "1"]:
            if self._xbasis:
                self.circuit[log].h(self.code_qubit)
            self.circuit[log].add_register(self.code_bit)
            self.circuit[log].measure(self.code_qubit, self.code_bit)