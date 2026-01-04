import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# ==============================================================================
# STEP 1: 3D GEOMETRIC PARAMETER DEFINITION (FIXED DESIGN INPUTS)
# ==============================================================================
# Overall condenser envelope dimensions
width = 302.0                # Condenser width in X-direction [mm]
height = 75.0                # Condenser height in Y-direction [mm]
extrusion_depth = 3.5        # Depth / thickness in Z-direction [mm] - 1 fin

# Pin / tube geometry
pin_diameter = 10.0          # Pin outer diameter [mm]
pin_radius = pin_diameter / 2.0

# Tube bank configuration
center_spacing = 25.0        # Center-to-center pitch in X-direction [mm]
num_pins_per_row = 12        # Number of pins per horizontal row

# ==============================================================================
# STEP 2: ROW-WISE PIN CENTERLINE LOCATIONS (STAGGERED BANK LAYOUT)
# ==============================================================================
# Lower bank – Row 1
y_pos_row_1 = 10.0
x_pos_row_1 = 21.0

# Lower bank – Row 2 (staggered offset)
y_pos_row_2 = 29.0
x_pos_row_2 = 6.0

# Upper bank – Row 3 (vertical repetition of Row 1)
y_pos_row_3 = y_pos_row_1 + 38.0
x_pos_row_3 = x_pos_row_1

# Upper bank – Row 4 (vertical repetition of Row 2)
y_pos_row_4 = y_pos_row_2 + 38.0
x_pos_row_4 = x_pos_row_2

# ==============================================================================
# STEP 3: 3D SCENE INITIALIZATION
# ==============================================================================
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111, projection='3d')

# ==============================================================================
# STEP 4: PIN / TUBE SOLID REPRESENTATION (CYLINDRICAL EXTRUSION)
# ==============================================================================
print("Generating 48 cylindrical pin elements for 3D visualization...")

def plot_cylinder(ax, center_x, center_y, radius, height):
    """
    Generates a solid cylindrical surface representing a condenser pin / tube
    extruded in the Z-direction.
    """
    z = np.linspace(0, height, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y

    ax.plot_surface(
        x_grid,
        y_grid,
        z_grid,
        color='gray',
        alpha=0.7,
        linewidth=0
    )

# Consolidated row definition for loop-based generation
row_params = [
    (x_pos_row_1, y_pos_row_1),
    (x_pos_row_2, y_pos_row_2),
    (x_pos_row



# ==============================================================================
# STEP 1: 2D GEOMETRIC DEFINITION OF THE CONDENSER LAYOUT
# ==============================================================================

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# Main condenser envelope (2D projection)
condenser_width = 302.0      # Overall condenser width [mm]
condenser_height = 75.0      # Overall condenser height [mm]
origin = (0.0, 0.0)          # Bottom-left reference coordinate

# Tube / port geometry parameters
tube_outer_diameter = 10.0   # Tube outer diameter [mm]
tube_radius = tube_outer_diameter / 2.0

# Tube pitch and population
horizontal_pitch = 25.0      # Center-to-center spacing in X-direction [mm]
tubes_per_row = 12           # Number of tubes per horizontal row

# ==============================================================================
# STEP 2: ROW-WISE TUBE CENTERLINE COORDINATES
# ==============================================================================
# Row 1: Primary reference row
row1_y = 10.0                # Vertical offset from bottom plate [mm]
row1_x_start = 21.0          # First tube center X-position [mm]

# Row 2: Staggered row (offset pattern)
row2_y = 29.0
row2_x_start = 6.0

# Row 3: Vertical repetition of Row 1 (two-pass / multi-bank configuration)
row3_y = row1_y + 38.0       # Vertical tube pitch between banks [mm]
row3_x_start = row1_x_start  # Same horizontal alignment as Row 1

# Row 4: Vertical repetition of Row 2
row4_y = row2_y + 38.0
row4_x_start = row2_x_start  # Same horizontal alignment as Row 2

# ==============================================================================
# STEP 3: FIGURE INITIALIZATION AND BASE GEOMETRY
# ==============================================================================
fig, ax = plt.subplots(figsize=(15, 5))

# Draw condenser outer boundary
condenser_outline = Rectangle(
    xy=origin,
    width=condenser_width,
    height=condenser_height,
    facecolor='lightblue',
    edgecolor='black',
    linewidth=2,
    zorder=1
)
ax.add_patch(condenser_outline)

# ==============================================================================
# STEP 4: TUBE CENTERLINE GENERATION AND RENDERING
# ==============================================================================
# Row 1 tubes
print(f"Generating tube Row 1 at Y = {row1_y} mm")
for i in range(tubes_per_row):
    cx = row1_x_start + i * horizontal_pitch
    tube = Circle(
        (cx, row1_y),
        radius=tube_radius,
        facecolor='gray',
        edgecolor='black',
        zorder=2
    )
    ax.add_patch(tube)

# Row 2 tubes
print(f"Generating tube Row 2 at Y = {row2_y} mm")
for i in range(tubes_per_row):
    cx = row2_x_start + i * horizontal_pitch
    tube = Circle(
        (cx, row2_y),
        radius=tube_radius,
        facecolor='gray',
        edgecolor='black',
        zorder=2
    )
    ax.add_patch(tube)

# Row 3 tubes (upper bank continuation)
print(f"Generating tube Row 3 at Y = {row3_y} mm")
for i in range(tubes_per_row):
    cx = row3_x_start + i * horizontal_pitch
    tube = Circle(
        (cx, row3_y),
        radius=tube_radius,
        facecolor='gray',
        edgecolor='black',
        zorder=2
    )
    ax.add_patch(tube)

# Row 4 tubes
print(f"Generating tube Row 4 at Y = {row4_y} mm")
for i in range(tubes_per_row):
    cx = row4_x_start + i * horizontal_pitch
    tube = Circle(
        (cx, row4_y),
        radius=tube_radius,
        facecolor='gray',
        edgecolor='black',
        zorder=2
    )
    ax.add_patch(tube)

# ==============================================================================
# STEP 5: PLOT CONFIGURATION AND FINAL VISUALIZATION
# ==============================================================================
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-10, condenser_width + 10)
ax.set_ylim(-10, condenser_height + 10)

ax.set_title(
    "2D Geometric Layout of EV Condenser Tube Banks (4-Row Configuration)",
    fontsize=14
)
ax.set_xlabel("X Direction [mm]")
ax.set_ylabel("Y Direction [mm]")
ax.grid(True, linestyle='--')

print("Geometry generation completed successfully. Displaying condenser layout...")
plt.show()
