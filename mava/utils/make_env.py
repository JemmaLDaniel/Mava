# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Dict, Tuple, Type

import jaxmarl
import jumanji
import matrax
from gigastep import ScenarioBuilder
from jaxmarl.environments.smax import Scenario
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.environments.routing.cleaner.generator import (
    RandomGenerator as CleanerRandomGenerator,
)
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorRandomGenerator,
)
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from omegaconf import DictConfig
import jax.numpy as jnp

from mava.types import MarlEnv
from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    CleanerWrapper,
    ConnectorWrapper,
    GigastepWrapper,
    LbfWrapper,
    MabraxWrapper,
    MatraxWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
    SmaxWrapper,
)
from mava.wrappers.jaxmarl import JaxMarlWrapper

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "MaConnector-v2": {
        "generator": ConnectorRandomGenerator,
        "wrapper": ConnectorWrapper,
    },
    "Cleaner-v0": {"generator": CleanerRandomGenerator, "wrapper": CleanerWrapper},
}

# Registry mapping environment names directly to the corresponding wrapper classes.
_matrax_registry = {"Matrax": MatraxWrapper}
_jaxmarl_registry: Dict[str, Type[JaxMarlWrapper]] = {"Smax": SmaxWrapper, "MaBrax": MabraxWrapper}
_gigastep_registry = {"Gigastep": GigastepWrapper}


def add_extra_wrappers(
    train_env: MarlEnv, eval_env: MarlEnv, config: DictConfig
) -> Tuple[MarlEnv, MarlEnv]:
    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    # Config generator and select the wrapper.
    generator = _jumanji_registry[env_name]["generator"]
    generator = generator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[env_name]["wrapper"]

    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(env_name, generator=generator, **env_config)
    eval_env = jumanji.make(env_name, generator=generator, **env_config)
    train_env = wrapper(train_env, add_global_state=add_global_state)
    eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env

MAP_NAME_TO_SCENARIO = {
    # name: (unit_types, n_allies, n_enemies, SMACv2 position generation, SMACv2 unit generation)
    "3m": Scenario(jnp.zeros((6,), dtype=jnp.uint8), 3, 3, False, False),
    "2s3z": Scenario(
        jnp.array([2, 2, 3, 3, 3] * 2, dtype=jnp.uint8), 5, 5, False, False
    ),
    "25m": Scenario(jnp.zeros((50,), dtype=jnp.uint8), 25, 25, False, False),
    "3s5z": Scenario(
        jnp.array(
            [
                2,
                2,
                2,
                3,
                3,
                3,
                3,
                3,
            ]
            * 2,
            dtype=jnp.uint8,
        ),
        8,
        8,
        False,
        False,
    ),
    "8m": Scenario(jnp.zeros((16,), dtype=jnp.uint8), 8, 8, False, False),
    "5m_vs_6m": Scenario(jnp.zeros((11,), dtype=jnp.uint8), 5, 6, False, False),
    "10m_vs_11m": Scenario(jnp.zeros((21,), dtype=jnp.uint8), 10, 11, False, False),
    "27m_vs_30m": Scenario(jnp.zeros((57,), dtype=jnp.uint8), 27, 30, False, False),
    "3s5z_vs_3s6z": Scenario(
        jnp.concatenate(
            [
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8),
                jnp.array([2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
            ]
        ),
        8,
        9,
        False,
        False,
    ),
    "3s_vs_5z": Scenario(
        jnp.array([2, 2, 2, 3, 3, 3, 3, 3], dtype=jnp.uint8), 3, 5, False, False
    ),
    "6h_vs_8z": Scenario(
        jnp.array([5, 5, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3], dtype=jnp.uint8),
        6,
        8,
        False,
        False,
    ),
    "smacv2_5_units": Scenario(jnp.zeros((10,), dtype=jnp.uint8), 5, 5, True, True),
    "smacv2_10_units": Scenario(jnp.zeros((20,), dtype=jnp.uint8), 10, 10, True, True),
    "smacv2_20_units": Scenario(jnp.zeros((40,), dtype=jnp.uint8), 20, 20, True, True),
    "smacv2_40_units": Scenario(jnp.zeros((80,), dtype=jnp.uint8), 40, 40, True, True),
    "smacv2_80_units": Scenario(jnp.zeros((160,), dtype=jnp.uint8), 80, 80, True, True),
    "smacv2_160_units": Scenario(jnp.zeros((320,), dtype=jnp.uint8), 160, 160, True, True),
    "smacv2_320_units": Scenario(jnp.zeros((640,), dtype=jnp.uint8), 320, 320, True, True),
    "smacv2_640_units": Scenario(jnp.zeros((1280,), dtype=jnp.uint8), 640, 640, True, True),
    "smacv2_1280_units": Scenario(jnp.zeros((2560,), dtype=jnp.uint8), 1280, 1280, True, True),
}


def map_name_to_scenario(map_name):
    """maps from smac map names to a scenario array"""
    return MAP_NAME_TO_SCENARIO[map_name]


def make_jaxmarl_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a JAXMARL environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A JAXMARL environment.

    """
    kwargs = dict(config.env.kwargs)
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Create jaxmarl envs.
    train_env = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(env_name, **kwargs),
        add_global_state,
    )
    eval_env = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(env_name, **kwargs),
        add_global_state,
    )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)

    return train_env, eval_env


def make_matrax_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """
    Creates Matrax environments for training and evaluation.

    Args:
    ----
        env_name: The name of the environment to create.
        config: The configuration of the environment.
        add_global_state: Whether to add the global state to the observation.

    Returns:
    -------
        A tuple containing a train and evaluation Matrax environment.

    """
    # Select the Matrax wrapper.
    wrapper = _matrax_registry[env_name]

    # Create envs.
    task_name = config["env"]["scenario"]["task_name"]
    train_env = matrax.make(task_name, **config.env.kwargs)
    eval_env = matrax.make(task_name, **config.env.kwargs)
    train_env = wrapper(train_env, add_global_state)
    eval_env = wrapper(eval_env, add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_gigastep_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a Gigastep environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
    -------
        A tuple of the environments.

    """
    wrapper = _gigastep_registry[env_name]

    kwargs = config.env.kwargs
    scenario = ScenarioBuilder.from_config(config.env.scenario.task_config)

    train_env = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)
    eval_env = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create environments for training and evaluation.

    Args:
    ----
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    env_name = config.env.scenario.name

    if env_name in _jumanji_registry:
        return make_jumanji_env(env_name, config, add_global_state)
    elif env_name in jaxmarl.registered_envs:
        return make_jaxmarl_env(env_name, config, add_global_state)
    elif env_name in _matrax_registry:
        return make_matrax_env(env_name, config, add_global_state)
    elif env_name in _gigastep_registry:
        return make_gigastep_env(env_name, config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
