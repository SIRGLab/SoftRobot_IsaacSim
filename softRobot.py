import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint  # Ensure you have torchdiffeq installed


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Run batched soft robots in Isaac Sim.")
    parser.add_argument(
        "--env_num",
        "--num_envs",
        dest="env_num",
        type=positive_int,
        default=1,
        help="Number of soft-robot environments to create (default: 1).",
    )
    parser.add_argument(
        "--env_spacing",
        type=float,
        default=0.4,
        help="Spacing in metres between environment origins (default: 0.4).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Isaac Sim without opening a window.",
    )
    args, _ = parser.parse_known_args()
    return args


class sfr(nn.Module):
    def __init__(self) -> None:
        super(sfr, self).__init__()
        self.l0 = 100e-3  # initial length of robot
        self.d = 7.5e-3  # cables offset
        self.ds = 0.005  # ode step time
        
        r0 = torch.zeros(3, 1).to(device)
        R0 = torch.eye(3).reshape(9, 1).to(device)
        y0 = torch.cat((r0, R0, torch.zeros([2, 1],device=device)), dim=0)
        
        self.y0 = y0.squeeze()

    def updateAction(self, actions):
        # Assuming actions is of shape (batch_size, 3)
        l = self.l0 + actions[:, 0]  # batch_size
        ux = actions[:, 2] / -(l * self.d)  # batch_size
        uy = actions[:, 1] / (l * self.d)  # batch_size
        return l, ux, uy

    def odeFunction(self, s, y):
        batch_size = y.shape[0]
        dydt = torch.zeros((batch_size, 14)).to(device)
        
        e3 = torch.tensor([0.0, 0.0, 1.0],device=device).reshape(1, 3, 1).repeat(batch_size, 1, 1)
        ux = y[:, 12]  # batch_size
        uy = y[:, 13]  # batch_size
        
        # Compute u_hat for each batch element
        u_hat = torch.zeros((batch_size, 3, 3),device=device)
        u_hat[:, 0, 2] = uy
        u_hat[:, 1, 2] = -ux
        u_hat[:, 2, 0] = -uy
        u_hat[:, 2, 1] = ux

        r = y[:, 0:3].reshape(batch_size, 3, 1)
        R = y[:, 3:12].reshape(batch_size, 3, 3)
        
        dR = torch.matmul(R, u_hat)  # batch_size x 3 x 3
        dr = torch.matmul(R, e3).squeeze(-1)  # batch_size x 3

        # Reshape and assign to dydt
        dydt[:, 0:3] = dr
        dydt[:, 3:12] = dR.reshape(batch_size, 9)
        return dydt

    def odeStepFull(self, actions):
        batch_size = actions.size(0)
        
        # Create a batch of initial conditions
        y0_batch = self.y0.unsqueeze(0).repeat(batch_size, 1).to(device)  # (batch_size, 14)
        l, ux, uy = self.updateAction(actions)
        y0_batch[:, 12] = ux
        y0_batch[:, 13] = uy
        
        sol = None
        number_of_segment = 3  
        for n in range(number_of_segment):
            
            # Determine the maximum length in the batch to ensure consistent integration steps
            max_length = torch.max(l).detach().cpu().numpy()
            t_eval = torch.arange(0.0, max_length + self.ds, self.ds).to(device)
        
            # Solve ODE for all batch elements simultaneously
            sol_batch = odeint(self.odeFunction, y0_batch, t_eval)  # (timesteps, batch_size, 14)

            # Mask out solutions for each trajectory after their respective lengths
            lengths = (l / self.ds).long()
            
            sol_masked = sol_batch.to(device)  # (timesteps, batch_size, 14)
        
            for i in range(batch_size):
                sol_masked[lengths[i]:, i ] = sol_masked[lengths[i], i]  # Masking with last one after trajectory ends
        
            if sol is None:
                sol = sol_masked
            else:                
                sol = torch.cat((sol, sol_masked), dim=0)
                    
            y0_batch = sol_masked[-1]  # (batch_size, 14)
            if n < number_of_segment-1:
                l, ux, uy = self.updateAction(actions[:, (n+1)*3:(n+2)*3])
                y0_batch[:, 12] = ux
                y0_batch[:, 13] = uy
                
        return sol  # (timesteps, batch_size, 14)


    def downsample_simple(self,arr, m):
        n = len(arr)
        indices = np.linspace(0, n - 1, m, dtype=int)  # Linearly spaced indices
        return arr[indices]


class Simulation:
    def __init__(self, num_sphere=30, env_num=1, env_spacing=0.4, headless=False) -> None:
        from isaacsim import SimulationApp

        self.num_sphere = num_sphere
        self.env_num = env_num
        self.env_spacing = env_spacing
        self.sphere_names = []

        self.simulation_app = SimulationApp({"headless": headless})

        from isaacsim.core.api import World
        self.my_world = World(stage_units_in_meters=1.0)

    def _environment_origins(self):
        """Lay environments out on a centred square grid."""
        columns = math.ceil(math.sqrt(self.env_num))
        rows = math.ceil(self.env_num / columns)
        x_center = (columns - 1) / 2.0
        y_center = (rows - 1) / 2.0

        return np.array(
            [
                [
                    (env_id % columns - x_center) * self.env_spacing,
                    (env_id // columns - y_center) * self.env_spacing,
                    0.0,
                ]
                for env_id in range(self.env_num)
            ],
            dtype=np.float32,
        )

    def create_robot(self):
        from isaacsim.core.api.objects import VisualSphere

        self.env_origins = self._environment_origins()
        for env_id, origin in enumerate(self.env_origins):
            environment_spheres = []
            for sphere_id in range(self.num_sphere):
                name = f"visual_sphere_{env_id}_{sphere_id}"
                self.my_world.scene.add(
                    VisualSphere(
                        prim_path=f"/World/envs/env_{env_id}/sphere_{sphere_id}",
                        name=name,
                        position=origin + np.array([0.0, 0.0, 0.5]),
                        radius=0.01 if sphere_id != self.num_sphere - 1 else 0.02,
                        color=(
                            np.array([255, 0, 255])
                            if sphere_id != self.num_sphere - 1
                            else np.array([0, 255, 0])
                        ),
                    )
                )
                environment_spheres.append(name)
            self.sphere_names.append(environment_spheres)

    def reset(self):
        self.my_world.scene.add_default_ground_plane()
        self.my_world.reset()
        self.t  = self.my_world.current_time


def main():
    args = parse_args()
    robot = sfr().to(device)
    sim = Simulation(
        num_sphere=30,
        env_num=args.env_num,
        env_spacing=args.env_spacing,
        headless=args.headless,
    )

    try:
        sim.create_robot()
        sim.reset()
        phase = torch.arange(args.env_num, device=device) * 0.2
        actions = torch.zeros((args.env_num, 9), device=device)

        while sim.simulation_app.is_running():
            if sim.my_world.is_playing():
                t = sim.my_world.current_time
                w = 2 * np.pi

                # One row per environment. A small phase offset makes it easy to
                # see that every environment is being updated independently.
                sine_wave = torch.sin(w * t + phase)
                actions[:, 1] = 0.005 * sine_wave
                actions[:, 4] = 0.005 * sine_wave
                actions[:, 7] = 0.005 * sine_wave

                sol = robot.odeStepFull(actions)
                sol = robot.downsample_simple(sol, sim.num_sphere).detach().cpu().numpy()

                for env_id, origin in enumerate(sim.env_origins):
                    for sphere_id, sphere_name in enumerate(sim.sphere_names[env_id]):
                        sphere = sim.my_world.scene.get_object(sphere_name)
                        position = sol[sphere_id, env_id, :3] + origin
                        sphere.set_world_pose(position=position)

                sim.my_world.step(render=not args.headless)
    finally:
        sim.simulation_app.close()


if __name__ == "__main__":
    main()
