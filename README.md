# Mobile Robot SLAM and Exploration

Project for the mobile robotics course (ROB201) at ENSTA Paris. A simulated robot with a lidar builds a map of an unknown environment, localizes itself in it, explores it on its own and returns to its starting point.

Built on the course template [ensta-rob201](https://github.com/emmanuel-battesti/ensta-rob201) and the [Place-Bot](https://github.com/emmanuel-battesti/place-bot) simulator. See `INSTALL.md` for setup.

## What I implemented (`tp_rob201/`)

| File | Content |
|---|---|
| `control.py` | Reactive obstacle avoidance and potential-field navigation toward a goal |
| `occupancy_grid.py` | Log-odds occupancy grid: lidar ray tracing, world/map conversions, display, save and load |
| `tiny_slam.py` | Scan-matching localization (pose score optimization) and map update |
| `planner.py` | A* path planning on the occupancy grid |
| `frontier_exploration.py` | Frontier detection and selection by distance and expected information gain |
| `my_robot_slam.py` | State machine: exploration, return to start and final navigation |

## Run

```bash
pip install -r requirements.txt
python tp_rob201/main.py
```
