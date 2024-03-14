# robotic_packing_simulator

Code accompanying the paper: "The Value of Information in Automated Packing Optimization with a Robotic Arm Sorting System Utilizing Multiple Conveyors"

To run experiments edit `/code/experiments.py` and run the script.

The 3D plots referenced in the paper can be found under `/figures`. They can be recreated using `/code/3dplot.py` after running the experiments.

To create visualizations such as plots or animations, edit `/code/plotting.py` and run the script. Note that the scripts expects each vegetable to have an associated SVG. This is also true for the robotic arms. SVGs can be downloaded from [SVG Repo](https://www.svgrepo.com/), but are not provided here. The SVGs should be placed in `/code/custom_markers/icons/`.
