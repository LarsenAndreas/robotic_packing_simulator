from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from matplotlib.transforms import Affine2D
import os

dirname = os.path.dirname(__file__)

icon_robot_path, icon_robot_attributes = svg2paths(os.path.join("code/custom_markers/icons/robot.svg"))
marker_robot = parse_path(icon_robot_attributes[0]["d"])
marker_robot.vertices -= marker_robot.vertices.mean(axis=0)
marker_robot = marker_robot.transformed(Affine2D().rotate_deg(180))

icon_cucumber_path, icon_cucumber_attributes = svg2paths(os.path.join("code/custom_markers/icons/cucumber.svg"))
marker_cucumber = parse_path(icon_cucumber_attributes[0]["d"])
marker_cucumber.vertices -= marker_cucumber.vertices.mean(axis=0)
marker_cucumber = marker_cucumber.transformed(Affine2D().rotate_deg(180))

icon_eggplant_path, icon_eggplant_attributes = svg2paths(os.path.join("code/custom_markers/icons/eggplant.svg"))
marker_eggplant = parse_path(icon_eggplant_attributes[0]["d"])
marker_eggplant.vertices -= marker_eggplant.vertices.mean(axis=0)
marker_eggplant = marker_eggplant.transformed(Affine2D().rotate_deg(180))

icon_tomato_path, icon_tomato_attributes = svg2paths(os.path.join("code/custom_markers/icons/tomato.svg"))
marker_tomato = parse_path(icon_tomato_attributes[0]["d"])
marker_tomato.vertices -= marker_tomato.vertices.mean(axis=0)
marker_tomato = marker_tomato.transformed(Affine2D().rotate_deg(180))
