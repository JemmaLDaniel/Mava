# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

# type: ignore

"""Jax systems parameter server."""


from typing import Any, Dict, List, Sequence, Union

import jax.numpy as jnp

from mava.core_jax import SystemParameterServer

# from mava.callbacks import Callback
# from mava.callbacks import CallbackHookMixin
from mava.utils.training_utils import non_blocking_sleep


class ParameterServer(SystemParameterServer):
    def __init__(
        self,
        components: List[Any],
    ) -> None:
        """Initialise the parameter server."""
        self.callbacks = components

        self.on_parameter_server_init_start()

        self.on_parameter_server_init()

        self.on_parameter_server_init_checkpointing()

        self.on_parameter_server_checkpointer()

        self.on_parameter_server_init_end()

    def get_parameters(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, jnp.ndarray]]:
        """Get parameters from the parameter server.

        Args:
            names : Names of the parameters to get
        Returns:
            The parameters that were requested
        """
        self._names = names

        self.on_parameter_server_get_parameters_start()

        if isinstance(names, str):
            self.system_parameters = self.parameters[names]
        else:
            self.system_parameters = {}
            for var_key in names:
                self._var_key = var_key
                # TODO (dries): Do we really have to convert the parameters to
                # numpy each time. Can we not keep the parameters in numpy form
                # without the checkpointer complaining?
                self.on_parameter_server_get_parameters()

        self.on_parameter_server_get_parameters_end()

        return self.system_parameters

    def set_parameters(
        self, names: Sequence[str], vars: Dict[str, jnp.ndarray]
    ) -> None:
        """Set parameters in the parameter server.

        Args:
            names : Names of the parameters to set
            vars : The values to set the parameters to
        """

        if isinstance(names, str):
            self._names = [names]
            self._vars = {names: vars}
        else:
            self._names = names
            self._vars = vars

        self.on_parameter_server_set_parameters_start()

        for var_key in names:
            self._var_key = var_key
            assert var_key in self.parameters
            if isinstance(self.parameters[var_key], tuple):
                # Loop through tuple
                for var_i in range(len(self.parameters[var_key])):
                    self._var_i = var_i

                    self.on_parameter_server_set_parameters_if_tuple()
            else:
                self.on_parameter_server_set_parameters_if_dict()

        self.on_parameter_server_set_parameters_end()

    def add_to_parameters(
        self, names: Sequence[str], vars: Dict[str, jnp.ndarray]
    ) -> None:
        """Add to the parameters in the parameter server.

        Args:
            names : Names of the parameters to add to
            vars : The values to add to the parameters to
        """
        if isinstance(names, str):
            self._names = [names]
            self._vars = {names: vars}
        else:
            self._names = names
            self._vars = vars

        self.on_parameter_server_add_to_parameters_start()

        for var_key in names:
            assert var_key in self.parameters
            self._var_key = var_key

            self.on_parameter_server_add_to_parameters()

        self.on_parameter_server_add_to_parameters_end()

    def run(self) -> None:
        """Run the parameter server. This function allows for checkpointing and other \
            centralised computations to be performed by the parameter server."""

        self.on_parameter_server_run_start()

        # Checkpoints every 5 minutes
        while True:
            # Wait 10 seconds before checking again
            non_blocking_sleep(10)

            self.on_parameter_server_run_loop_start()

            self.on_parameter_server_run_loop_checkpoint()

            self.on_parameter_server_run_loop()

            self.on_parameter_server_run_loop_termination()

            self.on_parameter_server_run_loop_end()
