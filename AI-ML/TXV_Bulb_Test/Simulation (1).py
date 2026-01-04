# ==============================================================================
# Step 4 & 5: Domain, Constraints, and Solver
#
# This final script sets up the simulation, applies all physical constraints
# (boundary conditions), and runs the solver.
#
# - Domain Creation: The CustomCoil geometry is instantiated.
# - Boundary Conditions (PointwiseBoundaryConstraint):
#   - `inlet`: Sets constant velocity and temperature for incoming fluid.
#   - `outlet`: Sets zero-pressure at the outlet.
#   - `no_slip`: Sets zero velocity at fluid-solid interface walls.
#   - `interface`: Ensures temperature and heat flux continuity between
#     fluid and solid.
#   - `outer_wall`: Applies a constant temperature to the outer wall as a
#     heat source.
# - Interior Constraints (PointwiseInteriorConstraint):
#   - Enforces the governing PDEs within their respective domains.
# - Solver: Configured with training parameters (e.g., max_steps).
# - Execution: `solver.solve()` starts the training process, where the
#   neural networks are trained to satisfy all defined constraints.
# ==============================================================================

from sympy import Symbol
from physicsnemo.hydra import ModulusConfig
from physicsnemo.solver import Solver
from physicsnemo.domain import Domain
from physicsnemo.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)

# Import from other files
from `1_geometry` import CustomCoil, GEOMETRY_PARAMS
from `2_physics_and_networks` import nodes

# ==============================================================================
# Step 4: Define Domain and Add Constraints
# ==============================================================================

# --- 4.1. Create Domain ---
domain = Domain()
geo = CustomCoil(GEOMETRY_PARAMS)

# --- 4.2. Boundary Conditions ---
# Inlet: Constant velocity and temperature
inlet_constraint = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo.inlet,
    outvar={"u": 1.0, "v": 0, "w": 0, "T_f": 0.0}, # Fluid enters at u=1, T=0
    batch_size=128
)
domain.add_constraint(inlet_constraint, "inlet")

# Outlet: Zero pressure
outlet_constraint = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo.outlet,
    outvar={"p": 0},
    batch_size=128
)
domain.add_constraint(outlet_constraint, "outlet")

# No-Slip: Zero velocity on fluid-solid interface walls
no_slip_constraint = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo.interface,
    outvar={"u": 0, "v": 0, "w": 0},
    batch_size=128
)
domain.add_constraint(no_slip_constraint, "no_slip")

# Interface: Continuity of temperature and heat flux
interface_constraint = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo.interface,
    outvar={"T_f": Symbol("T_s"), "normal_gradient_T_f": (k_s / k_f) * Symbol("normal_gradient_T_s")},
    batch_size=128
)
domain.add_constraint(interface_constraint, "interface")

# Outer Wall Heat Source: Constant temperature on the outer solid wall
outer_wall_constraint = PointwiseBoundaryConstraint(
    nodes=nodes,
    geometry=geo.outer_wall,
    outvar={"T_s": 1.0}, # Outer wall temperature is 1.0
    batch_size=128
)
domain.add_constraint(outer_wall_constraint, "outer_wall")

# --- 4.3. Interior Constraints ---
# Fluid Domain
interior_f_constraint = PointwiseInteriorConstraint(
    nodes=nodes,
    geometry=geo.fluid_geo,
    outvar={"navier_stokes": 0, "advection_diffusion": 0},
    batch_size=512,
)
domain.add_constraint(interior_f_constraint, "interior_fluid")

# Solid Domain
interior_s_constraint = PointwiseInteriorConstraint(
    nodes=nodes,
    geometry=geo.solid_geo,
    outvar={"diffusion": 0},
    batch_size=512,
)
domain.add_constraint(interior_s_constraint, "interior_solid")

# ==============================================================================
# Step 5: Define Solver and Run Simulation
# ==============================================================================

# --- 5.1. Create Solver ---
# Use a default config file, you can modify this
cfg = ModulusConfig.from_dict({
    "training": {"max_steps": 200000},
    "optimizer": {"lr": 0.001},
})
solver = Solver(cfg, domain)

# --- 5.2. Start Solving ---
solver.solve()

print("Simulation finished!")
