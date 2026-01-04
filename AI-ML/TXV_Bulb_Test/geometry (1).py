# ==============================================================================
# Step 1: Library Imports
# ==============================================================================
import sympy
from physicsnemo.geometry.sympy_utils import Extrude

# ==============================================================================
# Step 2: Geometry Parameters and Class Definition
#
# This block defines the geometry of the coil. It uses the `sympy` library
# to create 2D shapes for the fluid and solid domains and then extrudes them
# into 3D.
#
# - GEOMETRY_PARAMS: A dictionary holding all the key dimensions of the coil.
# - CustomCoil class:
#   - The __init__ method constructs the 2D cross-section by combining and
#     subtracting basic shapes (Polygon, Circle).
#   - It first defines the path for the fluid (fluid_geometry_2d).
#   - It then creates the solid pipe structure (solid_geometry_2d) by
#     buffering the fluid path and subtracting the fluid area itself.
#   - Finally, it uses the `Extrude` utility to give depth to the 2D shapes,
#     creating the 3D geometries (fluid_geo, solid_geo) and defines the
#     surfaces for the inlet, outlet, interface, and outer walls.
# ==============================================================================

# --- 2.1. Main Geometry Parameters ---
GEOMETRY_PARAMS = {
    "pipe_length_main": 4.2,      # Scaled down for simulation stability
    "pipe_length_mid": 3.0,
    "wall_thickness": 0.02,
    "extrusion_depth": 0.5,       # Depth of the coil in the third dimension (z)
    "inlet_y_start": -0.5,
    "inlet_height": 0.05,
    "annulus_radius_inner": 0.45,
    "annulus_radius_outer": 0.5,
    "middle_pipe_y_start": 0.45,
    "middle_pipe_height": 0.05,
    "top_pipe_y_start": 0.70,
    "top_pipe_height": 0.05,
    "connector_y_start": 0.5,
    "connector_height": 0.2,
}

# --- 2.2. Coil Geometry Construction Class ---
class CustomCoil:
    def __init__(self, params):
        self.params = params
        t = params["wall_thickness"]
        len_main = params["pipe_length_main"]
        len_mid = params["pipe_length_mid"]
        depth = params["extrusion_depth"]
        
        # --- 2D Fluid Domain Construction ---
        inlet_y0, inlet_h = params["inlet_y_start"], params["inlet_height"]
        inlet_channel = sympy.Polygon(sympy.Point(-len_main, inlet_y0), sympy.Point(0, inlet_y0), sympy.Point(0, inlet_y0 + inlet_h), sympy.Point(-len_main, inlet_y0 + inlet_h))

        annulus_r_in, annulus_r_out = params["annulus_radius_inner"], params["annulus_radius_outer"]
        annulus_channel = sympy.Circle(sympy.Point(0, 0), annulus_r_out) - sympy.Circle(sympy.Point(0, 0), annulus_r_in)

        mid_y0, mid_h = params["middle_pipe_y_start"], params["middle_pipe_height"]
        middle_channel = sympy.Polygon(sympy.Point(-len_main, mid_y0), sympy.Point(0, mid_y0), sympy.Point(0, mid_y0 + mid_h), sympy.Point(-len_main, mid_y0 + mid_h))

        top_y0, top_h = params["top_pipe_y_start"], params["top_pipe_height"]
        top_outlet_channel = sympy.Polygon(sympy.Point(-len_main, top_y0), sympy.Point(0, top_y0), sympy.Point(0, top_y0 + top_h), sympy.Point(-len_main, top_y0 + top_h))

        conn_y0, conn_h = params["connector_y_start"], params["connector_height"]
        left_connector = sympy.Polygon(sympy.Point(-len_main, conn_y0), sympy.Point(-len_mid, conn_y0), sympy.Point(-len_mid, conn_y0 + conn_h), sympy.Point(-len_main, conn_y0 + conn_h))

        self.fluid_geometry_2d = (inlet_channel | annulus_channel | middle_channel | top_outlet_channel | left_connector)

        # --- 2D Solid Domain Construction ---
        total_solid_blob = self.fluid_geometry_2d.buffer(t)
        central_void = sympy.Circle(sympy.Point(0, 0), annulus_r_in)
        self.solid_geometry_2d = total_solid_blob - self.fluid_geometry_2d - central_void

        # --- Extrude Geometries to 3D ---
        self.fluid_geo = Extrude(self.fluid_geometry_2d, (-depth/2, depth/2))
        self.solid_geo = Extrude(self.solid_geometry_2d, (-depth/2, depth/2))

        # --- Define 3D Boundaries ---
        self.inlet = Extrude(sympy.Line(sympy.Point(-len_main, inlet_y0), sympy.Point(-len_main, inlet_y0 + inlet_h)), (-depth/2, depth/2))
        self.outlet = Extrude(sympy.Line(sympy.Point(-len_main, top_y0), sympy.Point(-len_main, top_y0 + top_h)), (-depth/2, depth/2))
        self.interface = self.solid_geo.boundary & self.fluid_geo.boundary
        self.outer_wall = self.solid_geo.boundary - self.interface

