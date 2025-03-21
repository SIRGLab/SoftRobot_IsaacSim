
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


from torchdiffeq import odeint  # Ensure you have torchdiffeq installed
import numpy as np

from isaacsim import SimulationApp


device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

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
    def __init__(self,numb_sphere = 30) -> None:
        self.num_sphere = numb_sphere

        self.simulation_app = SimulationApp({"headless": False})

        from isaacsim.core.api import World
        self.my_world = World(stage_units_in_meters=1.0)
    
        self.num_sphere = 30
    def create_robot(self):
        from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, VisualSphere

        for i in range(self.num_sphere):
            self.my_world.scene.add(
                VisualSphere(
                    prim_path="/sphere"+str(i),
                    name="visual_sphere"+str(i),
                    position=np.array([0, 0, 0.5]),
                    radius=0.01 if i != self.num_sphere-1 else 0.02,
                    color=np.array([255, 0, 255]) if i != self.num_sphere-1 else np.array([0, 255, 0]),
                )
            )

    def reset(self):
        self.my_world.scene.add_default_ground_plane()
        self.my_world.reset()
        self.t  = self.my_world.current_time


robot = sfr().to(device)
sim = Simulation(numb_sphere=30)
sim.create_robot()
sim.reset()

t  = sim.my_world.current_time
while sim.simulation_app.is_running():   
    if sim.my_world.is_playing():    
        w  = 2*np.pi
        t +=  sim.my_world.current_time - t
        actions = torch.tensor([[0.0, 0.005*np.sin(w*t), 0.0, 
                                0.0, 0.005*np.sin(w*t), 0.0,
                                0.0, 0.005*np.sin(w*t), 0.0]], device=device).reshape(1, 9)
        robot.updateAction(actions)
        sol = robot.odeStepFull(actions)
        sol = robot.downsample_simple(sol, sim.num_sphere).detach().cpu().numpy()
        
        for i in range(sim.num_sphere):
            sphere = sim.my_world.scene.get_object("visual_sphere"+str(i))
            new_position = sol[i, 0, :3]
            sphere.set_world_pose(position=new_position)
            
        sim.my_world.step(render=True)

sim.simulation_app.close()