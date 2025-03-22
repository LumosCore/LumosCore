
![image](rapidNetSim.png)

# Introduction
This project is used for [LumosCore](https://arxiv.org/abs/2411.01503)'s simulation and 
its future work. LumosCore reconfigures the network topology when task starts, and dose not customize the routing strategy.


# Prerequisites
This project contains two python packages, rapidnetsim and figret. We recommend you to install the packages in `requirements.txt`
and then run `pip install .` in the root directory of this project. `cmake` and cpp build tools are also required.

# Get Started

## Simple Test
```
cd large_exp_4096GPU
bash start.sh
```
The finish time of each task is shown in large_exp_4096GPU/beta_2000/***/task_time.log

# Feature
Simulate real time through global static simulator and event base class.

Task generator generate numerous jobs.

Global topology including link capacity.

Jobs can share links.

Update link occupancy at every task event.

Network refresh after every event is done.

Different routing schemes are supported.

Initial configuration templating.

Large-scale verification.

Multi-stage controller.

Ring adn Butterfly (HDRM) strategy.

Multiple tasks can be excuted in turn.

The real start time of every task is determined by TaskStartEvent and TASK_WAITING_LIST.

Automatic Logger.

Measurement event measure link sharing.

Collision weight mechanism.

Support single GPU occupation.
