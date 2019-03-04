# PablosPuzzlingPendulum

This repository is a final "dump" of an university project (bachelor thesis-level), which aimed at implementing methods to control a robot arm perform a pendulum swing up. As it was a very deadline-driven project, the code commits might not be as clean as desired. The final PDF describing in the project can be found here:

todo: insert link :)


# Structure of project and repo

The aim of the project was to try to make a robot arm (see picture) swing up and balance a pendulum.



The repository contains several folders with a specific functionality. 

* `communication` implements the low-level data communication to send commands and receive observations from sensors from an arduino to a pc over USB. 
* `micro` contains the code running on an arduino, which is required for repaying movement commands to a servo motor and relaying sensor data for motor joint positions and pendulum positions.
* `mm` stands for mathematical modeling. This folder contains code for simulating the pendulum environment.
* 'ourgym' implements an openAI gym interface for both the simulation and the "real-world" control of the robot. This makes it easier to test reinforcement learning algorithms on the robot.
* `rl` implements several reinforcement learning algorithms, including "normal" tabular SARSA as well as DQN and actor-critic. 
* `simulation`should be part of the `ourgym` folder.
* `vision` has some code experimenting with using a camera to predict the pendulum position. Ultimately not used.

In the root folder there are some scripts to automate experimenting with all the algorithms. These were used in a google-cloud environment to run simulations.
