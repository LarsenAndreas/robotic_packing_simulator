import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

clr = sns.color_palette("deep").as_hex()

data = {
    # "arm1": dict(b0=0.4280, b1=-0.0011, b2=-0.0010, b3=-3.802e-6),  # Original
    "arm1": dict(b0=0.4738, b1=-0.0035, b2=-0.0016, b3=3.362e-6, b4=3.729e-5, b5=-3.802e-6),  # Extended
    "arm2": dict(b0=0.4183, b1=-0.0007, b2=-0.0012, b3=4.095e-6),
    "arm3": dict(b0=0.3791, b1=0.0008, b2=-0.0011, b3=1.908e-6),
    "arm4": dict(b0=0.3735, b1=0.0011, b2=-0.0010, b3=-2.201e-6),
}

fig = go.Figure()
controller = "prob"
arm = "arm1"
df = pd.read_csv("raw_data.csv")
df = df.loc[df["controller"] == controller]
count = df.groupby(["controller", "buffer", "arm1_reach"]).count().reset_index()
df = df.groupby(["controller", "buffer", "arm1_reach"]).mean().reset_index()
fig.add_scatter3d(
    x=df["buffer"],
    y=df[f"{arm}_reach"],
    z=df["avg_tray_completed_weight"],
    name=f"Data {arm}",
    mode="markers",
    marker=dict(color=clr[3], size=10),
    text=[f"c: {c}" for c in count["avg_tray_completed_weight"]],
)

buffer, reach = np.meshgrid(np.arange(0, 200), np.arange(0, 65))
coeff = data[arm]
# surface = coeff["b0"] + coeff["b1"] * reach + coeff["b2"] * buffer + coeff["b3"] * buffer * reach
surface = coeff["b0"] + coeff["b1"] * reach + coeff["b2"] * buffer + coeff["b3"] * buffer**2 + coeff["b4"] * reach**2 + coeff["b5"] * reach * buffer
fig.add_surface(
    x=buffer.flatten(),
    y=buffer.flatten(),
    z=surface,
    name=f"Model {arm}",
    showscale=False,
    opacity=0.6,
    colorscale=[[0, clr[0]], [1, clr[0]]],
)

fig.update_layout(
    scene=dict(
        xaxis_title=dict(text="Buffer", font=dict(size=40)),
        yaxis_title=dict(text="Reach", font=dict(size=40)),
        zaxis_title=dict(text="Overfill", font=dict(size=40)),
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),
        ),
    ),
)

fig.write_html(f"figures/scatter_3d-{arm}-{controller}-coeff={len(coeff)}.html")
fig.show()
