# ************** STUDENTS EDIT THIS FILE **************

from SteeringBehaviors import Wander
import SimulationEnvironment as sim

import numpy as np


def collect_training_data(total_actions, total_batches):
    # set-up environment
    sim_env = sim.SimulationEnvironment()

    # robot control
    action_repeat = 100
    steering_behavior = Wander(action_repeat)

    num_params = 7
    # STUDENTS: network_params will be used to store your training data
    # a single sample will be comprised of: sensor_readings, action, collision

    for stkv in range(total_batches):
        network_params = []
        for action_i in range(total_actions):
            progress = 100*float(action_i)/total_actions
            print(f'Collecting Training Data {progress}%   ', end="\r", flush=True)

            # steering_force is used for robot control only
            action, steering_force = steering_behavior.get_action(
                    action_i,
                    sim_env.robot.body.angle
                )

            for action_timestep in range(action_repeat):
                if action_timestep == 0:
                    _, collision, sensor_readings = sim_env.step(steering_force)
                else:
                    _, collision, _ = sim_env.step(steering_force)

                if collision:
                    steering_behavior.reset_action()
                    # STUDENTS NOTE: this statement only EDITS collision of PREVIOUS action
                    # if current action is very new.
                    if action_timestep < action_repeat * .3:   # in case prior action caused collision
                        network_params[-1][-1] = collision  # share collision result with prior action
            network_params.append(list(sensor_readings) + [action] + [collision])
        np.savetxt(f"./saved/training_set_part_{stkv+1}.csv", np.array(network_params), delimiter=",")

if __name__ == '__main__':
    total_actions = 10000
    total_batches = 100
    collect_training_data(total_actions, total_batches)
