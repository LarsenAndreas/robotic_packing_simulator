import os
import shutil
from functools import partial
from math import ceil

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from controller import Controller
from conveyor.system import System
from conveyor.veggie import Veggie
from custom_markers.markers import marker_robot
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from p_tqdm import p_umap
from tqdm import tqdm
from utility import getRect

sns.set_theme()


def plotSystem(t_list: float, system: System):
    for t in t_list:
        plt.figure(figsize=(16, 9))
        ax_viz = plt.subplot(111)

        state_system = system.getSystemState(t, time_dependent_specs=True)

        # TRAYS
        plot_trays = {"x": [], "y": [], "text": [], "clr": []}
        for i, state_trays in enumerate(state_system["trays"].values()):
            x, y = state_trays["pos"]
            remain_weight = state_trays["remain_weight"]
            remain_content = state_trays["remain_content"]
            plot_trays["x"].append(x)
            plot_trays["y"].append(y)
            if len(remain_content) == 0:
                plot_trays["text"].append(f"DONE!\n{remain_weight:.1f}g")
                plot_trays["clr"].append("green")
            else:
                remain_prime = np.prod([Veggie.getProperties(kind=kind)["prime"] for kind in remain_content])
                plot_trays["text"].append(f"{remain_prime}\n{remain_weight:.1f}g")
                plot_trays["clr"].append("red")

        ax_viz.scatter(
            x=plot_trays["x"] + plot_trays["x"],
            y=plot_trays["y"] + plot_trays["y"],
            marker="s",
            s=300,
            c=plot_trays["clr"] + plot_trays["clr"],
        )
        for x, y, text in zip(plot_trays["x"], plot_trays["y"], plot_trays["text"]):
            ax_viz.text(x, y + np.sign(y), text, ha="center", va="top" if y < 0 else "bottom", fontsize=8, c="black")

        # ARMS
        plot_arms = {"x": [], "y": [], "item": [], "bb": [], "targets": []}
        for arm_id, (arm, state_arm) in enumerate(state_system["arms"].items()):
            x, y = state_arm["pos"]
            plot_arms["x"].append(x)
            plot_arms["y"].append(y)
            plot_arms["item"].append(state_arm.get("target_veggie", arm_id))
            plot_arms["bb"].append(getRect(arm))

            if state_arm.get("pos_stop"):
                plot_arms["targets"].append((state_arm["pos"], state_arm["pos_stop"]))

        ax_viz.scatter(plot_arms["x"], plot_arms["y"], marker=marker_robot, s=300, c="black")
        for x, y, text, bb in zip(plot_arms["x"], plot_arms["y"], plot_arms["item"], plot_arms["bb"]):
            ax_viz.text(x + 1, y, text)
            ax_viz.add_patch(bb)

        ax_viz.add_collection(LineCollection(plot_arms["targets"], linestyles=":"))

        # VEGGIES
        plot_veggies = {kind: {"x": [], "y": [], "marker": Veggie.getProperties(kind, rng=None)["marker"]} for kind in system.kind_veggies}
        for state_veggie in state_system["veggies"].values():
            x, y = state_veggie["pos"]
            plot_veggies[state_veggie["kind"]]["x"].append(x)
            plot_veggies[state_veggie["kind"]]["y"].append(y)

        for _, data in plot_veggies.items():
            ax_viz.scatter(data["x"], data["y"], marker=data["marker"][0], s=300, c=data["marker"][1])

        # BELT
        buffer_start = system.belt_start - system.length_buffer
        ax_viz.hlines(y=(system.width_belt / 2, -system.width_belt / 2), xmin=buffer_start, xmax=system.belt_end)
        ax_viz.fill_between([buffer_start, system.belt_start], system.width_belt / 2, -system.width_belt / 2, color="C2", alpha=0.3)
        ax_viz.text(buffer_start + system.length_buffer / 2, 0, "Buffer", fontsize=16, ha="center")
        ax_viz.set_xlim(left=buffer_start - 5, right=system.belt_end + 5)

        ax_viz.spines[["right", "top", "left", "bottom"]].set_visible(False)

        ax_viz.set_title(f"{t=:.3f}")
        plt.savefig(f"./frames/frame_{t:.4f}.jpg")
        plt.close()


def makeVideo(
    type_controller: str,
    length_buffer: float,
    width_belt: float,
    speed_belt: float,
    speed_arms: float,
    permutation_arms: list[float],
    cutoff_probability: float,
    n_trays: int,
    n_veggies: int,
    spacing_trays: float,
    max_veggies: int,
    t_min: float,
    t_max: float,
    t_step_arm_eval: float = 10,
    t_step_tray_eval: float = 0.5,
    t_step_viz: float = 0.2,
    force_count: float = 0.6,
    seed=None,
    output: str = "animation.mp4",
    fps: int = 20,
):
    system = System(length_buffer=length_buffer, width_belt=width_belt, speed_belt=speed_belt, cutoff_probability=cutoff_probability, seed=seed)
    pos_mount = []
    reach_bb = []
    last_reach_start = 0
    for reach in permutation_arms:
        pos_mount.append((last_reach_start + reach / 2, 0))
        reach_bb.append((width_belt + 3, reach / 2, width_belt + 3, reach / 2))

        last_reach_start += reach
    system.genArms(pos_mount=pos_mount, reach_bb=reach_bb, speed=[speed_arms for _ in permutation_arms])
    system.genTrays(count=n_trays, spacing=spacing_trays, max_veggies=max_veggies)
    system.genVeggies(count=n_veggies, t_min=t_min, t_max=t_max)

    controller = Controller(system=system)

    dt_tray_eval = 0
    dt_arm_eval = 0
    forced = False

    t_step = min(t_step_arm_eval, t_step_tray_eval)
    for t in tqdm(np.arange(t_min, t_max, t_step), desc="Running Controller"):
        system_state = system.getSystemState(t)
        if dt_tray_eval >= t_step_tray_eval:
            forced = controller.schedulePossibleTrays(system_state=system_state, force_count=force_count, cutoff_prob=cutoff_probability)
            dt_tray_eval = 0

        if dt_arm_eval >= t_step_arm_eval or forced:
            controller.makeDecision(system_state=system_state, type_controller=type_controller, prio_last=("none", 1.1))
            dt_arm_eval = 0
            forced = False

        dt_tray_eval += t_step
        dt_arm_eval += t_step

    folder = "frames"
    shutil.rmtree(folder)
    os.mkdir(folder)
    chunks = 12
    time_steps = np.arange(t_min, t_max, t_step_viz)
    partitions = int(ceil(len(time_steps) / chunks))
    time_step_chunks = [time_steps[i * partitions : (i + 1) * partitions] for i in range(chunks)]
    p_umap(partial(plotSystem, system=system), time_step_chunks, desc="Creating Frames")
    image_folder = "./frames"
    video_name = output
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")], key=lambda x: float(x[6:-4]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(filename=video_name, frameSize=(width, height), fps=fps, fourcc=cv2.VideoWriter_fourcc(*"mp4v"), isColor=True)
    for image in tqdm(images, desc="Stiching Frames"):
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
    print(f'Video saved as "{output}"')


def plotData(file: str, section: str, **kwargs: dict):
    df = pd.read_csv(file)

    # Make the x-axis consistent
    df = df.sort_values(["buffer", "controller"])
    df["config"] = df["arm1_reach"].astype(str) + "-" + df["arm2_reach"].astype(str) + "-" + df["arm3_reach"].astype(str) + "-" + df["arm4_reach"].astype(str)
    formatter = {"prob": "Probability", "random": "Random"}

    ## BUFFER ##
    if section == "buffer":
        plot_kwargs = dict(gap=0.1, fliersize=0)
        sns.set_theme(font_scale=2)

        fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        g = sns.boxplot(data=df, x="buffer", y="avg_tray_completed_weight", hue="controller", ax=axs, **plot_kwargs)
        g.set(xlabel="Buffer Size", ylabel="Average Overfill")

        # sns.move_legend(axs, loc="upper left")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        sns.move_legend(axs, loc="upper right")
        plt.tight_layout()
        plt.savefig(f"figures/buffer_size-vs-overfill.png")

        fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        sns.boxplot(data=df, x="buffer", y="tray_discarded", hue="controller", ax=axs[1], legend=False, **plot_kwargs).set(xlabel="Buffer Size", ylabel="Trays Discarded")
        g = sns.boxplot(data=df, x="buffer", y="tray_completed", hue="controller", ax=axs[0], **plot_kwargs)
        g.set(xlabel="Buffer Size", ylabel="Trays Completed")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        plt.tight_layout()
        plt.savefig(f"figures/buffer_size-vs-trays_discarded_completed.png")

        fig, axs = plt.subplots(4, 1, figsize=(16, 9), sharex=True)
        plot_kwargs = dict(gap=0.1, fliersize=0)
        sns.boxplot(data=df, x="buffer", y="arm1_downtime", hue="controller", ax=axs[0], legend=False, **plot_kwargs).set(xlabel="Buffer Size", ylabel="Arm1 Downtime")
        sns.boxplot(data=df, x="buffer", y="arm2_downtime", hue="controller", ax=axs[1], legend=False, **plot_kwargs).set(xlabel="Buffer Size", ylabel="Arm2 Downtime")
        sns.boxplot(data=df, x="buffer", y="arm3_downtime", hue="controller", ax=axs[2], legend=False, **plot_kwargs).set(xlabel="Buffer Size", ylabel="Arm3 Downtime")
        g = sns.boxplot(data=df, x="buffer", y="arm4_downtime", hue="controller", ax=axs[3], legend=True, **plot_kwargs)
        g.set(xlabel="Buffer Size", ylabel="Arm4 Downtime")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        sns.move_legend(axs[3], loc="lower right")
        plt.tight_layout()
        plt.savefig(f"figures/buffer_size-vs-arm_downtime.png")

        fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        g = sns.boxplot(data=df, x="buffer", y="veggie_discarded", hue="controller", ax=axs, legend=True, **plot_kwargs)
        g.set(xlabel="Buffer Size", ylabel="Veggies Discarded")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        plt.tight_layout()
        plt.savefig(f"figures/buffer_size-vs-veggies_discarded.png")
    ## ###### ##

    ## CONFIG ##
    if section == "config":
        fig, axs = plt.subplots(1, 1, figsize=(16, 9), sharex=True)
        g = sns.boxplot(data=df, x="config", y="avg_tray_completed_weight", hue="controller", ax=axs)
        g.set(xlabel="Configuration", ylabel="Average Overfill")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        sns.move_legend(axs, loc="upper left")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"figures/config-vs-overfill.png")

        fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        plot_kwargs = dict(gap=0.1, fliersize=0)
        sns.boxplot(data=df, x="config", y="tray_discarded", hue="controller", ax=axs[1], legend=False, **plot_kwargs).set(xlabel="Configuration", ylabel="Discarded Trays")
        g = sns.boxplot(data=df, x="config", y="tray_completed", hue="controller", ax=axs[0], **plot_kwargs)
        g.set(xlabel="Configuration", ylabel="Completed Trays")
        g.legend_.set_title("Controller")
        for t in g.legend_.texts:
            t.set_text(formatter[t._text])
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"figures/config-vs-trays_discarded_completed.png")
    ## ###### ##

    ## MISC ##
    if section == "misc":
        controller = kwargs.get("controller", "prob")
        df = df.loc[df["controller"] == controller]

        palette = sns.color_palette()
        if controller == "prob":
            palette = [palette[0]]
        elif controller == "random":
            palette = [palette[1]]

        g = sns.relplot(
            data=df.groupby(["controller", "config", "buffer"]).mean(),
            x="buffer",
            y="avg_tray_completed_weight",
            col="config",
            col_wrap=5,
            hue="controller",
            palette=palette,
            legend=False,
            height=2,
            aspect=1.5,
        )
        # g.fig.(formatter[controller])
        g.set_axis_labels("Buffer Length", "Average Overfill")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            ax.set_title(ax.get_title(), fontsize=16)
        # plt.tight_layout()
        plt.savefig(f"figures/{controller}_config-vs-overfill.png")

    ## #### ##

    plt.show()


if __name__ == "__main__":
    plotData(file="raw_data.csv", section="misc", controller="prob")

    system_specs = {
        "type_controller": "prob",
        "n_trays": 20,
        "n_veggies": 200,
        "permutation_arms": [50.0, 25.0, 12.5, 25.0],
        "spacing_trays": 5,
        "max_veggies": 3,
        "length_buffer": 30,
        "width_belt": 20,
        "speed_belt": 3,
        "speed_arms": 10,
        "cutoff_probability": 0.05,
        "t_step_arm_eval": 10,
        "t_step_tray_eval": 0.5,
        "force_count": 4,
        "seed": 42,
    }
    makeVideo(output="animation.mp4", fps=20, t_min=0, t_max=100, t_step_viz=0.2, **system_specs)
