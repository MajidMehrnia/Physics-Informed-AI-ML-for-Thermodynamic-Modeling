# ==============================================================================
# Step 3: PDE and Network Definition
#
# This section establishes the physical laws and the neural network
# architecture for the simulation.
#
# - Physical Properties: Constants for fluid (water) and solid (aluminum)
#   properties are defined, like viscosity and thermal conductivity.
# - PDEs: The governing partial differential equations are instantiated.
#   - NavierStokes: For incompressible fluid flow.
#   - AdvectionDiffusion: For heat transfer in the moving fluid.
#   - Diffusion: For heat conduction in the solid.
# - Neural Networks: Two FullyConnectedArch networks are created.
#   - flow_net: Maps (x, y, z) to flow variables (u, v, w, p).
#   - temp_net: Maps (x, y, z) to temperature fields (T_f, T_s).
# - Nodes: The PDEs and networks are combined into a list of computational
#   nodes to build the model graph.
# ==============================================================================

from sympy import Symbol, Key

from physicsnemo.hydra import instantiate_arch
from physicsnemo.architecture import FullyConnectedArch

from physicsnemo.eq.pdes.navier_stokes import NavierStokes
from physicsnemo.eq.pdes.advection_diffusion import AdvectionDiffusion
from physicsnemo.eq.pdes.diffusion import Diffusion

# --- 3.1. Define PDEs ---
# Physical properties (e.g., for water and aluminum)
# Fluid (Water)
nu_f = 1.0e-6  # Kinematic viscosity
D_f = 1.4e-7   # Thermal diffusivity
k_f = 0.6      # Thermal conductivity

# Solid (Aluminum)
D_s = 9.7e-5
k_s = 237

# Instantiate equations
flow_eq = NavierStokes(nu=nu_f, rho=1.0, dim=3, time=False)
temp_eq_f = AdvectionDiffusion(T="T_f", rho=1.0, D=D_f, dim=3, time=False, stream_fun_name=flow_eq.stream_fun_name)
temp_eq_s = Diffusion(T="T_s", D=D_s, dim=3, time=False)

# --- 3.2. Define Neural Networks ---
# One network for velocity/pressure, one for temperature
flow_net = FullyConnectedArch(
    input_keys=[Key("x"), Key("y"), Key("z")],
    output_keys=[Key("u"), Key("v"), Key("w"), Key("p")]
)
temp_net = FullyConnectedArch(
    input_keys=[Key("x"), Key("y"), Key("z")],
    output_keys=[Key("T_f"), Key("T_s")]
)

# Combine all components into a list of nodes
nodes = (
    flow_eq.make_nodes()
    + temp_eq_f.make_nodes()
    + temp_eq_s.make_nodes()
    + [flow_net.make_node(name="flow_network")]
    + [temp_net.make_node(name="temp_network")]
)
