from __future__ import annotations

import pickle
import traceback
from copy import deepcopy
from functools import partial
from pprint import pprint

import numpy as np
import utility
from controller import Controller
from conveyor.system import System
from p_tqdm import p_umap


def saveSystem(system: System, name: str):
    with open(name, "wb") as outp:
        pickle.dump(system, outp, pickle.HIGHEST_PROTOCOL)


def runExperiment(type_controller: str, permutation_arms: list, length_buffer: float, **kwargs):
    n_trays = kwargs.get("n_trays", 500)
    n_veggies = kwargs.get("n_veggies", 1000)

    spacing_trays = kwargs.get("spacing_trays", 5)
    max_veggies = kwargs.get("max_veggies", 3)
    width_belt = kwargs.get("width_belt", 20)
    speed_belt = kwargs.get("speed_belt", 3)
    cutoff_probability = kwargs.get("cutoff_probability", None)

    t_min = kwargs.get("t_min", 0)
    t_max = kwargs.get("t_max", 1000)
    t_step_arm_eval = kwargs.get("t_step_arm_eval", 10)
    t_step_tray_eval = kwargs.get("t_step_tray_eval", 0.5)
    speed = kwargs.get("speed_arm", 10)
    force_count = kwargs.get("force_count", n_veggies)
    prio_last = kwargs.get("prio_last", ("None", 1))

    t_step = min(t_step_arm_eval, t_step_tray_eval)

    pos_mount = []
    reach_bb = []
    speed_arm = []
    last_reach_start = 0
    for reach in permutation_arms:
        pos_mount.append((last_reach_start + reach / 2, 0))
        reach_bb.append((width_belt + 3, reach / 2, width_belt + 3, reach / 2))
        speed_arm.append(speed)

        last_reach_start += reach

    for i in range(kwargs.get("n_experiments", 100)):
        try:
            system = System(length_buffer=length_buffer, width_belt=width_belt, speed_belt=speed_belt, cutoff_probability=cutoff_probability, seed=None)
            rng_origin = deepcopy(system.rng)

            system.genArms(pos_mount=pos_mount, reach_bb=reach_bb, speed=speed_arm)
            system.genTrays(count=n_trays, spacing=spacing_trays, max_veggies=max_veggies)
            system.genVeggies(count=n_veggies, t_min=t_min, t_max=t_max)

            system.t_start = t_min
            system.t_stop = t_max
            system.t_step = t_step

            controller = Controller(system=system)

            dt_tray_eval = 0
            dt_arm_eval = 0
            forced = False
            for t in np.arange(t_min, t_max, t_step):
                system_state = system.getSystemState(t)
                if dt_tray_eval >= t_step_tray_eval:
                    forced = controller.schedulePossibleTrays(system_state=system_state, force_count=force_count, cutoff_prob=cutoff_probability)
                    dt_tray_eval = 0

                if dt_arm_eval >= t_step_arm_eval or forced:
                    controller.makeDecision(system_state=system_state, type_controller=type_controller, prio_last=prio_last)
                    dt_arm_eval = 0
                    forced = False

                dt_tray_eval += t_step
                dt_arm_eval += t_step

            name = f"./saved_systems/{type_controller}_{length_buffer}_{permutation_arms}_{i}.pkl"
            saveSystem(system=system, name=name)
        except Exception as e:
            name = f"./FAILED/{type_controller}_{length_buffer}_{permutation_arms}_{i}.txt"
            with open(name, "w") as fp:
                fp.write(str(e) + "\n")
                fp.write(traceback.format_exc())
            name = f"./FAILED/{type_controller}_{length_buffer}_{permutation_arms}_{i}.rng"
            with open(name, "wb") as outp:
                pickle.dump(rng_origin, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    n_arms = 4
    min_reach = 12.5
    buffer_lengths = [0, 10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200]
    controllers = ["prob", "random"]

    controller_list = []
    buffer_list = []
    config_list = []

    for element in utility.genCombinations(controllers, n_arms, min_reach, buffer_lengths):
        controller_list.append(element[0])
        buffer_list.append(element[1])
        config_list.append(element[2])

    kwargs = dict(n_experiments=100, cutoff_probability=0.1, force_count=4)

    pprint(kwargs)

    p_umap(partial(runExperiment, **kwargs), controller_list, config_list, buffer_list)
