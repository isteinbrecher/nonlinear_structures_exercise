# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: queens
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Beam finite element simulations with 4C
#
# In this exercise, we will be creating, running and analyzing a beam finite element model, using the open source finite element software [4C](https://www.4c-multiphysics.org). 4C contains a variety of different finite element implementations for all kind of physical fields. In this exercise, we will exclusively be using reduced dimensional, geometrically exact beam elements, based on these two works:
# - Jelenić, G., and Crisfield, M. A., 1999, “Geometrically Exact 3D Beam Theory: Implementation of a Strain-Invariant Finite Element for Statics and Dynamics,” Computer Methods in Applied Mechanics and Engineering, 171(1), pp. 141–171.
# - Meier, C., 2016, “Geometrically Exact Finite Element Formulations for Slender Beams and Their Contact Interaction,” Dissertation, Technische Universität München.
#
# For pre-processing, we will use the open source beam finite element generator [BeamMe](https://github.com/beamme-py/beamme). Post-processing is done directly within this notebook.
#
# Note: The beam elements in 4C are developed for 3D analysis. However, in this exercise, we will only be considering plane examples.

# %%
# We have to import the required packages, classes and functions.
import matplotlib.pyplot as plt
import numpy as np
from beamme.core.mesh import Mesh
from beamme.four_c.element_beam import Beam3rLine2Line2
from beamme.four_c.material import MaterialReissner

from utils.lecture_utils import (
    create_beam_mesh_line_2d,
    create_boundary_condition_2d,
    get_force_displacement_data,
    plot_beam_2d,
    run_four_c,
)

# %% [markdown]
# ## Cantilever beam
#
# In this first exercise, we will be creating the following cantilever beam model:
#
# <img src="doc/cantilever.png" alt="Cantilever with force" width="400">
#
# The beam has a length of $l = 0.3\text{m}$, a circular cross-section with width the radius $r = 0.01\text{m}$, a Young's modulus of $E = 2.1\cdot 10^{11}\text{N}/\text{mm}^2$. The left end of the beam is clamped, while a load of $F = 10000\text{N}$ is applied at the right end in negative $y$-direction.
#
# We use two-noded linear beam finite elements based on the geometrically exact beam theory. The cantilever is discretized using 3 elements.
#
# The following code block creates the beam finite element model using BeamMe and runs the simulation in 4C. At the end, the deformed shape of the beam and the cross-section force resultants are plotted. The simulation files are stored in a folder named `cantilever_1_0`.
#

# %%
# In the beginning, we create the mesh container which will hold all elements,
# materials, and boundary conditions.
mesh = Mesh()

# Here we define the material properties for the beam elements.
# We use a Reissner beam material with specified radius, Young's modulus,
material = MaterialReissner(radius=0.01, youngs_modulus=2.1e11)

# Next, we create a straight beam connecting two points in space.
# For spatial discretization, we use 3 linear two-noded beam elements.
beam_set = create_beam_mesh_line_2d(
    mesh, Beam3rLine2Line2, material, [0, 0], [0.3, 0], n_el=3
)

# To fix the cantilver beam, we apply Dirichlet boundary conditions to all
# positions and rotations at one node.
create_boundary_condition_2d(
    mesh, beam_set["start"], bc_type="dirichlet", directions=["x", "y", "theta"]
)

# At the other end of the beam, we apply a Neumann boundary condition
# representing a downward force in y-direction.
create_boundary_condition_2d(
    mesh, beam_set["end"], bc_type="neumann", directions=["y"], values=[-10000.0]
)

# Now we can run the simulation. We have to give a unique name, in this case "cantilever_1_0",
# to store the results in a separate folder.
run_four_c(
    mesh=mesh,
    simulation_name="cantilever_1_0",
    n_steps=1,
    tol=1e-10,
)

# Display the solution (referenced by the unique name "cantilever_1_0").
plot_beam_2d("cantilever_1_0")

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 2.1:</strong>
#
#   We can see the deformed shape of the cantilever beam after applying the downward force. The cross-section force resultants are also plotted along the beam length. According to the linear Bernoulli-Euler beam theory, the bending moment should be linear, with the maximum value at the fixed end and zero at the free end. The shear force should be constant and the axial force should be zero. 
#   
#   1. The simulation results do not match these expectations. Can you explain why?
#   1. What could be done to get the expected results? Adapt the code below accordingly and rerun the simulation.
# </div>

# %%
mesh = Mesh()
material = MaterialReissner(radius=0.01, youngs_modulus=2.1e11)
beam_set = create_beam_mesh_line_2d(
    mesh,
    Beam3rLine2Line2,
    material,
    [0, 0],
    [0.3, 0],
    n_el=100,  # We increase the number of elements here
)
create_boundary_condition_2d(
    mesh, beam_set["start"], bc_type="dirichlet", directions=["x", "y", "theta"]
)
create_boundary_condition_2d(
    mesh,
    beam_set["end"],
    bc_type="neumann",
    directions=["y"],
    values=[-1.0],  # We reduce the force here
)

run_four_c(
    mesh=mesh,
    simulation_name="cantilever_1_1",
    n_steps=1,
    tol=1e-10,
)

plot_beam_2d("cantilever_1_1")

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 2.2:</strong>
#
#   - We not want to change the load at the free end to an external moment with magnitude $M = 2000\text{Nm}$. Adapt the following code to achieve this (20 elements should be used for the discretization).
#
#     *Hint:* Look at the function `create_boundary_condition_2d` and its arguments. The `directions` argument specifies which components of the boundary condition are applied, while the `values` argument specifies the corresponding magnitudes.
#
#   - Increase the magnitude of the applied moment such that the beam exactly deforms into a half circle. 
#
#     *Hint:* The moment of inertia for a circular cross-section is given as $I = \frac{\pi r^4}{4}$. The required moment to deform the beam into a half circle can then be calculated as $M = \frac{\pi}{2} \frac{EI}{l}$.
#
#     *Hint:* This is highly non-linear problem. You might need to increase the number of load steps `n_steps` to get a converged solution. 
# </div>

# %%
youngs_modulus = 2.1e11
radius = 0.01
moment_of_inertia = np.pi * radius**4 / 4
length = 0.3

moment_magnitude = np.pi * youngs_modulus * moment_of_inertia / length

mesh = Mesh()
material = MaterialReissner(radius=radius, youngs_modulus=youngs_modulus)
beam_set = create_beam_mesh_line_2d(
    mesh, Beam3rLine2Line2, material, [0, 0], [length, 0], n_el=20
)
create_boundary_condition_2d(
    mesh, beam_set["start"], bc_type="dirichlet", directions=["x", "y", "theta"]
)
create_boundary_condition_2d(
    mesh,
    beam_set["end"],
    bc_type="neumann",
    directions=["theta"],
    values=[moment_magnitude],
)

run_four_c(
    mesh=mesh,
    simulation_name="cantilever_1_2",
    n_steps=10,
    tol=1e-10,
)

plot_beam_2d("cantilever_1_2")

# %% [markdown]
# ## Euler buckling column
#
# In the first exercise, we will be creating the following cantilever beam model:
#
# <img src="doc/cantilever.png" alt="Cantilever with force" width="400">
#
# The beam has a length of $l = 0.3\text{m}$, a circular cross-section with width the radius $r = 0.01\text{m}$, a Young's modulus of $E = 2.1\cdot 10^{11}\text{N}/\text{mm}^2$. The left end of the beam is clamped, while a load of $F = 10000\text{N}$ is applied at the right end in negative $y$-direction.
#
# We use two-noded linear beam finite elements based on the geometrically exact beam theory. The cantilever is discretized using 3 elements.
#
# The following code block creates the beam finite element model using BeamMe and runs the simulation in 4C. At the end, the deformed shape of the beam and the cross-section force resultants are plotted. The simulation files are stored in a folder named `cantilever_1_0`.

# %%
youngs_modulus = 2.1e11
radius = 0.01
moment_of_inertia = np.pi * radius**4 / 4
length = 0.3
critical_load = np.pi**2 * youngs_modulus * moment_of_inertia / (length**2)
print(critical_load)

# Create the mesh
mesh = Mesh()
material = MaterialReissner(radius=radius, youngs_modulus=youngs_modulus)
beam_set = create_beam_mesh_line_2d(
    mesh, Beam3rLine2Line2, material, [0, 0], [length, 0], n_el=20
)

# Apply Dirichlet BCs at both ends
create_boundary_condition_2d(
    mesh, beam_set["start"], bc_type="dirichlet", directions=["x", "y"]
)
create_boundary_condition_2d(
    mesh, beam_set["end"], bc_type="dirichlet", directions=["y"]
)

# Apply imperfection moments at both ends
imperfection_moment = 10
create_boundary_condition_2d(
    mesh,
    beam_set["start"],
    bc_type="neumann",
    directions=["theta"],
    values=[imperfection_moment],
    linear_increase=False,
)
create_boundary_condition_2d(
    mesh,
    beam_set["end"],
    bc_type="neumann",
    directions=["theta"],
    values=[-imperfection_moment],
    linear_increase=False,
)

# Apply an axial force at the right end
force = 500000.0
create_boundary_condition_2d(
    mesh,
    beam_set["end"],
    bc_type="neumann",
    directions=["x"],
    values=[-force],
    linear_increase=True,
)

run_four_c(
    mesh=mesh,
    simulation_name="cantilever_1_3",
    n_steps=20,
    tol=1e-10,
)

plot_beam_2d("cantilever_1_3")

# Plot the force displacement relationship
force, displacement = get_force_displacement_data("cantilever_1_3")
plt.plot(displacement[:, 1], force)
plt.axhline(y=critical_load, color="r", linestyle="--", label="Critical Load")
plt.xlabel("v")
plt.ylabel("F")
plt.title("Force-displacement plot")
plt.grid(True)
plt.show()
