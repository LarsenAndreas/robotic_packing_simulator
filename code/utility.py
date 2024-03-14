from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conveyor.arm import Arm
    from conveyor.system import System

import json
import os
import pickle
from itertools import product
from math import factorial

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from p_tqdm import p_umap
from tqdm import tqdm


def getRect(arm: Arm, clr="black", ls="--"):
    x_rect = arm.pos_mount[0] - arm.reach_bb[3]
    y_rect = arm.pos_mount[1] - arm.reach_bb[2]
    w_rect = arm.reach_bb[1] + arm.reach_bb[3]
    h_rect = arm.reach_bb[0] + arm.reach_bb[2]

    return Rectangle(xy=(x_rect, y_rect), width=w_rect, height=h_rect, fill=False, ls=ls, ec=clr)


def calcSystemPerfBasic(system: System, t: float, **kwargs) -> dict:
    n_dec = kwargs.get("n_dec", 4)

    runout = {0: 0}
    runout_data = []
    kind_runout = {v: 0 for v in system.kind_veggies}
    weight_runout = 0
    for veggie in system.veggies:
        if not veggie.t_picked:
            t_falloff = veggie.t_spawn + ((system.length_belt + system.length_buffer) / system.speed_belt)
            if t_falloff <= t:
                runout[round(t_falloff, n_dec)] = runout.get(round(t_falloff, n_dec), 0) + abs(veggie.weight)
                runout_data.append(abs(veggie.weight))
                kind_runout[veggie.kind] += 1
                weight_runout += veggie.weight

    overfill = {0: 0}
    overfill_data = []
    weight_overfill = 0
    weight_throughput = 0
    weight_discarded = 0
    count_discarded = 0
    count_throughput = 0
    for tray in system.trays:
        state_tray = tray.getState(t, time_dependent_specs=True)
        if len(state_tray["remain_content"]) == 0:
            overfill[round(state_tray["t_packed"], n_dec)] = overfill.get(round(state_tray["t_packed"], n_dec), 0) + abs(state_tray["t_packed"])
            overfill_data.append(abs(state_tray["remain_weight"]))
            weight_overfill += abs(state_tray["remain_weight"])
            weight_throughput += tray.spec_weight + abs(state_tray["remain_weight"])
            count_throughput += 1

        if tray.discarded is not False and t >= tray.discarded:
            weight_discarded += tray.spec_weight - state_tray["remain_weight"]
            count_discarded += 1

    t_spawn_veggie_first = min(system.veggies, key=lambda x: x.t_spawn).t_spawn
    downtime_arms = {str(arm): max(0, t - t_spawn_veggie_first - ((system.length_buffer + arm.pos_mount[0] - arm.reach_bb[3]) / system.speed_belt)) for arm in system.arms}
    for arm in system.arms:
        for task in arm.tasks:
            if task["time_allocated"][1] < t:
                downtime_arms[str(arm)] -= task["time_allocated"][1] - task["time_allocated"][0]
            elif task["time_allocated"][0] <= t <= task["time_allocated"][1]:
                downtime_arms[str(arm)] -= t - task["time_allocated"][0]

    return {
        "runout": runout,
        "overfill": overfill,
        "runout_data": runout_data,
        "overfill_data": overfill_data,
        "weight_runout": weight_runout,
        "weight_overfill": weight_overfill,
        "weight_discarded": weight_discarded,
        "weight_throughput": weight_throughput,
        "count_discarded": count_discarded,
        "count_throughput": count_throughput,
        "kind_runout": kind_runout,
        "downtime_arms": downtime_arms,
    }


def calcSystemPerf(system: System, t: float) -> dict:
    # DATA
    time = np.arange(system.t_start, t, system.t_step)
    n_dec = str(system.t_step)[::-1].find(".")

    runout = {0: 0}
    kind_runout = []
    weight_runout = []
    count_veggie_runout = 0
    for veggie in system.veggies:
        state_veggie = veggie.getState(t)
        if state_veggie["pos"] is None:
            t_falloff = veggie.t_spawn + (system.length_belt + system.length_buffer) / system.speed_belt
            runout[round(t_falloff, n_dec)] = runout.get(round(t_falloff, n_dec), 0) + abs(veggie.weight)
            kind_runout.append(veggie.kind)
            weight_runout.append(veggie.weight)
            count_veggie_runout += 1

    time_pruned = [round(t, n_dec) for t in time if not runout.get(round(t, n_dec))]
    idx_sort = np.argsort(list(runout.keys()) + time_pruned)
    time_runout = np.array(list(runout.keys()) + time_pruned)[idx_sort]
    weight_time_runout = np.array(list(runout.values()) + [0 for _ in range(len(time_pruned))])[idx_sort]
    if len(weight_time_runout) > 1:
        gradient_weight_time_runout = np.gradient(np.cumsum(weight_time_runout), system.t_step)
    else:
        gradient_weight_time_runout = weight_time_runout

    weight_discarded = []
    overfill = {0: 0}
    weight_overfill = []
    count_tray_filled = 0
    count_tray_discarded = 0
    for tray in system.trays:
        state_tray = tray.getState(t, time_dependent_specs=True)
        if len(state_tray["remain_content"]) == 0:
            overfill[round(state_tray["t_packed"], n_dec)] = overfill.get(round(state_tray["t_packed"], n_dec), 0) + abs(state_tray["t_packed"])
            weight_overfill.append(abs(state_tray["remain_weight"]))
            count_tray_filled += 1

        if tray.discarded is not None and t >= tray.discarded:
            weight_discarded.append(tray.spec_weight)
            count_tray_discarded += 1

    time_pruned = [round(t, n_dec) for t in time if not overfill.get(round(t, n_dec))]
    idx_sort = np.argsort(list(overfill.keys()) + time_pruned)
    time_overfill = np.array(list(overfill.keys()) + time_pruned)[idx_sort]
    weight_time_overfill = np.array(list(overfill.values()) + [0 for _ in range(len(time_pruned))])[idx_sort]
    if len(weight_time_overfill) > 1:
        gradient_weight_time_overfill = np.gradient(np.cumsum(weight_time_overfill), system.t_step)
    else:
        gradient_weight_time_overfill = weight_time_overfill

    t_spawn_veggie_first = min(system.veggies, key=lambda x: x.t_spawn).t_spawn
    x_downtime_arms = [f"Arm {i+1}" for i, *_ in enumerate(system.arms)]
    y_downtime_arms = [max(0, t - t_spawn_veggie_first - (system.length_buffer + a.pos_mount[0] - a.reach_bb[3]) / system.speed_belt) for a in system.arms]
    for i, arm in enumerate(system.arms):
        for task in arm.tasks:
            if task["time_allocated"][1] < t:
                y_downtime_arms[i] -= task["time_allocated"][1] - task["time_allocated"][0]
            elif task["time_allocated"][0] <= t <= task["time_allocated"][1]:
                y_downtime_arms[i] -= t - task["time_allocated"][0]
    if t > 0:
        y_downtime_arms = np.round(np.array(y_downtime_arms) / t * 100, 2)

    return {
        "kind_runout": kind_runout,
        "count_veggie_runout": count_veggie_runout,
        "time_runout": time_runout,
        "weight_runout": weight_runout,
        "weight_time_runout": weight_time_runout,
        "gradient_weight_time_runout": gradient_weight_time_runout,
        "weight_discarded": weight_discarded,
        "count_tray_filled": count_tray_filled,
        "count_tray_discarded": count_tray_discarded,
        "time_overfill": time_overfill,
        "weight_overfill": weight_overfill,
        "weight_time_overfill": weight_time_overfill,
        "gradient_weight_time_overfill": gradient_weight_time_overfill,
        "x_downtime_arms": x_downtime_arms,
        "y_downtime_arms": y_downtime_arms,
    }


def genCombiReach(min_reach, n_arms):
    min_reach = min_reach
    n_arms = n_arms

    pool = [min_reach]
    i = 0
    while pool[i] < (100 - min_reach * (n_arms - 1)):
        pool.append(pool[i] + min_reach)
        i += 1
    n = len(pool)

    satisfies = []

    for p in product(pool, repeat=n_arms):
        if sum(p) == 100:
            satisfies.append(p)

    return satisfies


def genCombinations(controllers: list[str, str], n_arms: int, min_reach: float, buffer_lengths: list[float, float]):
    reach_combinations = genCombiReach(min_reach=min_reach, n_arms=n_arms)
    test_combinations = product(controllers, buffer_lengths, reach_combinations)
    return test_combinations


def saveSystem(system: System, name: str):
    with open(name, "wb") as outp:
        pickle.dump(system, outp, pickle.HIGHEST_PROTOCOL)


def aggregate(folder: str) -> dict:
    data = {}
    for file in tqdm(os.listdir(folder)):
        if file == "data.json":
            continue
        with open(f"{folder}/{file}", "rb") as inp:
            system = pickle.load(inp)

        perf = calcSystemPerfBasic(system, t=1000, n_dec=1)

        name = file[: file.rindex("_")]
        if not data.get(name):
            data[name] = perf
        else:
            for key in perf.keys():
                if key in ("runout", "overfill", "kind_runout", "downtime_arms"):
                    for i in perf[key].keys():
                        data[name][key][i] = data[name][key].get(i, 0) + perf[key][i]
                elif key in (
                    "weight_runout",
                    "weight_overfill",
                    "weight_throughput",
                    "weight_discarded",
                    "count_discarded",
                    "count_throughput",
                    "runout_data",
                    "overfill_data",
                ):
                    data[name][key] += perf[key]
                # elif key in ("runout_data", "overfill_data"):
                #     data[name][key] = perf[key]
                else:
                    raise Exception(f"{key=} not recognized!")

    print("Saving data...")
    with open("data_test.json", "w") as fp:
        json.dump(data, fp, indent=4)
    print("Saved!")
    return data


def makeCSV(files: list[str,], t: float) -> pd.DataFrame:
    df = {
        "controller": [],
        "buffer": [],
        "tray_completed": [],
        "avg_tray_completed_weight": [],
        "tray_discarded": [],
        "avg_tray_discarded_weight": [],
        "veggie_discarded": [],
        "avg_veggie_discarded_weight": [],
        "arm1_reach": [],
        "arm2_reach": [],
        "arm3_reach": [],
        "arm4_reach": [],
        "arm1_downtime": [],
        "arm2_downtime": [],
        "arm3_downtime": [],
        "arm4_downtime": [],
    }

    for file in files:
        if file == "data.json":
            continue
        with open(file, "rb") as inp:
            system = pickle.load(inp)

        # CONTROLLER
        if "true_random" in file.lower():
            controller = "truerandom"
        elif "random_mod" in file.lower():
            controller = "random_mod"
        elif "prob_mod" in file.lower():
            controller = "prob_mod"
        elif "random" in file.lower():
            controller = "random"
        elif "prob" in file.lower():
            controller = "prob"
        else:
            raise Exception("No Controller found!")

        df["controller"].append(controller)

        # BUFFER/ARM CONFIG
        *_, buffer, config, _ = file.split("_")

        df["buffer"].append(buffer)

        config = config[1:-1]
        arm1, arm2, arm3, arm4 = map(str.strip, config.split(","))
        df["arm1_reach"].append(arm1)
        df["arm2_reach"].append(arm2)
        df["arm3_reach"].append(arm3)
        df["arm4_reach"].append(arm4)

        # DOWNTIME
        t_spawn_veggie_first = min(system.veggies, key=lambda x: x.t_spawn).t_spawn
        y_downtime_arms = [max(0, t - t_spawn_veggie_first - (system.length_buffer + a.pos_mount[0] - a.reach_bb[3]) / system.speed_belt) for a in system.arms]
        for i, arm in enumerate(system.arms):
            for task in arm.tasks:
                if task["time_allocated"][1] < t:
                    y_downtime_arms[i] -= task["time_allocated"][1] - task["time_allocated"][0]
                elif task["time_allocated"][0] <= t <= task["time_allocated"][1]:
                    y_downtime_arms[i] -= t - task["time_allocated"][0]

        df["arm1_downtime"].append(y_downtime_arms[0] / t * 100)
        df["arm2_downtime"].append(y_downtime_arms[1] / t * 100)
        df["arm3_downtime"].append(y_downtime_arms[2] / t * 100)
        df["arm4_downtime"].append(y_downtime_arms[3] / t * 100)

        # RUNOUT
        veggie_discarded = 0
        veggie_discarded_weight = 0
        for veggie in system.veggies:
            pos = veggie._getPos(t)
            if pos is not None and pos[0] > system.belt_end:
                veggie_discarded += 1
                veggie_discarded_weight += veggie.weight

        df["veggie_discarded"].append(veggie_discarded)
        df["avg_veggie_discarded_weight"].append(veggie_discarded_weight / veggie_discarded if veggie_discarded > 0 else 0)

        # GIVE-AWAY
        tray_completed = 0
        tray_discarded = 0
        tray_completed_weight = 0
        tray_discarded_weight = 0
        for tray in system.trays:
            if tray.discarded and t >= tray.discarded:
                tray_discarded += 1
                tray_discarded_weight += tray.spec_weight
            else:
                state_tray = tray.getState(t, time_dependent_specs=True)
                if len(state_tray["remain_content"]) == 0:
                    tray_completed += 1
                    tray_completed_weight += abs(state_tray["remain_weight"])

        df["tray_completed"].append(tray_completed)
        df["avg_tray_completed_weight"].append(tray_completed_weight / tray_completed if tray_completed > 0 else 0)
        df["tray_discarded"].append(tray_discarded)
        df["avg_tray_discarded_weight"].append(tray_discarded_weight / tray_discarded if tray_discarded_weight > 0 else 0)

    return pd.DataFrame(df)


if __name__ == "__main__":

    files = [f"saved_systems_5/{file}" for file in os.listdir("saved_systems_5")]
    chunks = [files[i : i + 100] for i in range(0, len(files), 100)]
    df_list = p_umap(partial(makeCSV, t=1000), chunks)

    df = pd.concat(df_list)
    df.to_csv("raw_data.csv", index=False)
