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
# $
# % Define TeX macros for this document
# \def\vv#1{\boldsymbol{#1}}
# \def\mm#1{\boldsymbol{#1}}
# \def\R#1{\mathbb{R}^{#1}}
# \def\SO{SO(3)}
# \def\triad{\mm{\Lambda}}
# $

# %% [markdown]
# # Large rotations
#
# We will use the large rotation framework implemented in the beam finite element input generator [**BeamMe**](https://github.com/beamme-py/beamme).
# BeamMe provides a `Rotation` class that encapsulates various representations of rotations (rotation matrices, rotation vectors, quaternions, etc.) and methods for converting between them, composing rotations, and applying rotations to vectors.
#
# Before solving the following exercises, have a look at the large rotation example notebook in the BeamMe repository: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/beamme-py/beamme/main?labpath=examples%2Fexample_1_finite_rotations.ipynb)

# %% [markdown]
# ## Exercises
#
# We need to import the relevant python packages and objects for the exercises:

# %%
import numpy as np
from beamme.core.rotation import Rotation

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.1:</strong>
#
#   Create a rotation object representing a rotation of 90 degrees about the $z$-axis and print the quaternion, rotation matrix, and rotation vector representations of this rotation.
# </div>

# %%
# === SOLUTION START: Ex 1.1 ===
rotation_z90 = Rotation([0.0, 0.0, 1.0], np.pi / 2)

print("Quaternion:\n", rotation_z90.get_quaternion())
print("Rotation matrix:\n", rotation_z90.get_rotation_matrix())
print("Rotation vector:\n", rotation_z90.get_rotation_vector())
# === SOLUTION END: Ex 1.1 ===

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.2:</strong>
#
#   Use the rotation from Exercise 1.1 to rotate the vector $\vv{a} = [1, 0, 0]^T$. Print the rotated vector.
# </div>

# %%
a = [1.0, 0.0, 0.0]

# === SOLUTION START: Ex 1.2 ===
rotated_a = rotation_z90 * a
print("Rotated vector:\n", rotated_a)
# === SOLUTION END: Ex 1.2 ===

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.3:</strong>
#
#   Create two rotation objects: $\triad_1$ representing a rotation of $45^\circ$ degrees about the $x$-axis and $\triad_2$ representing a rotation of $30^\circ$ about the $y$-axis. Compose these two rotations, by first applying $\triad_1$ and then $\triad_2$. Extract and print the rotation vector corresponding to the composed rotation.
#
#   Show that a reordering of the how the rotations are applied leads to a different result, i.e., first applying $\triad_2$ and then $\triad_1$.
# </div>

# %%
# === SOLUTION START: Ex 1.3 ===
lambda_1 = Rotation([1.0, 0.0, 0.0], np.pi / 4)  # 45° about x
lambda_2 = Rotation([0.0, 1.0, 0.0], np.pi / 6)  # 30° about y

# Apply lambda_1 first, then lambda_2:
lambda_21 = lambda_2 * lambda_1
print("Composed (apply 1 then 2) rotation vector:\n", lambda_21.get_rotation_vector())

# Reordered: apply lambda_2 first, then lambda_1:
lambda_12 = lambda_1 * lambda_2
print("Reordered (apply 2 then 1) rotation vector:\n", lambda_12.get_rotation_vector())
# Comment: The two rotation vectors are different, demonstrating the non-commutative nature of finite rotations.
# === SOLUTION END: Ex 1.3 ===

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.4:</strong>
#
#   In this exercise, we investigate the inverse of a finite rotation and its properties.
#
#   Create a rotation object $\triad$ representing a rotation of $60^\circ$ about the axis
#   $[1, 1, 0]^T$.
#
#   1. Compute the inverse rotation $\triad^{-1}$ and print the rotation vectors of $\triad$ and $\triad^{-1}$. Comment on the relation between them.
#   2. Apply $\triad$ to a vector $\vv{a} = [1, 0.2, -0.1]^T$ and then apply the inverse
#       rotation $\triad^{-1}$ to the result. Verify that the original vector is recovered.
#   3. Verify that the composition $\triad^{-1}\triad$ (and $\triad\triad^{-1}$) corresponds
#       to the identity rotation by checking its rotation matrix representation.
#
#   *Hint:* You may use the `inv()` method of the `Rotation` class to compute the inverse rotation.
# </div>

# %%
angle = np.pi / 3
axis = [1.0, 1.0, 0.0]
a = [1.0, 0.2, -0.1]

# === SOLUTION START: Ex 1.4 ===
rotation_axis60 = Rotation(axis, angle)
rotation_axis60_inv = rotation_axis60.inv()

print("Rotation vector (lambda):\n", rotation_axis60.get_rotation_vector())
print("Rotation vector (lambda^{-1}):\n", rotation_axis60_inv.get_rotation_vector())
# Comment: the inverse corresponds to the opposite rotation, thus the rotation vector
# has the same magnitude but opposite direction.

rotated_a = rotation_axis60 * a
restored_a = rotation_axis60_inv * rotated_a
print("Original a:\n", a)
print("Restored a (lambda^{-1} * lambda * a):\n", restored_a)
# Comment: the original vector is recovered.

composition_left = rotation_axis60_inv * rotation_axis60
composition_right = rotation_axis60 * rotation_axis60_inv
print("lambda^{-1} * lambda", composition_left)
print("lambda * lambda^{-1}", composition_right)
# === SOLUTION END: Ex 1.4 ===

# %% [markdown]
# <div class="alert alert-info" role="alert">
#   <strong>Exercise 1.5:</strong>
#
#   In the lecture notes, we discussed the difference between additive and multiplicative rotation vector increments. Combining (2.14) and (2.15) gives the following relations:
#   $$
#   \triad(\vv{\psi}_2) =
#   \triad(\vv{\psi}_1 + \Delta \vv{\psi}) =
#   \triad(\Delta \vv{\theta})  \triad(\vv{\psi}_1) =
#   \triad(\vv{\psi}_1) \triad(\Delta \vv{\Theta}).
#   $$
#   Here, $\Delta \vv{\psi}$ is the additive rotation vector increment, while $\Delta \vv{\theta}$ and $\Delta \vv{\Theta}$ are the multiplicative rotation vector increments applied from the left and right, respectively.
#
#   The initial rotation vector $\vv{\psi}_1 = [0.1, 0.2, 0.3]^T$ and additive increment $\Delta \vv{\psi} = [0.01, -0.02, 0.03]^T$ are given.
#
#   1. Compute $\vv{\psi}_2$
#   2. Compute the multiplicative increments $\Delta \vv{\theta}$ and $\Delta \vv{\Theta}$.
#   3. Verify the relation (2.16), i.e., $\Delta \vv{\theta} = \triad(\vv{\psi}_1) \Delta \vv{\Theta}$.
#   4. Check if the relation (2.22), $\delta \vv{\psi} = \mm{T}(\vv{\psi}) \delta \vv{\theta}$ holds for the given values. Comment on the result.
#
#   *Hint:* You may use the `get_transformation_matrix()` method of the `Rotation` class to compute the transformation matrix $\mm{T}(\vv{\psi})$.
# </div>

# %%
psi_1 = np.array([0.1, 0.2, 0.3])
delta_psi = np.array([0.01, -0.02, 0.03])

# === SOLUTION START: Ex 1.5 ===
psi_2 = psi_1 + delta_psi
print("psi_2:\n", psi_2)

rotation_1 = Rotation.from_rotation_vector(psi_1)
rotation_2 = Rotation.from_rotation_vector(psi_2)

# Left multiplicative increment: rot_2 = rot(dtheta_left) * rot_1
left_relative = rotation_2 * rotation_1.inv()
delta_theta_left = left_relative.get_rotation_vector()
print("Delta theta (left):\n", delta_theta_left)

# Right multiplicative increment: rot_2 = rot_1 * rot(dTheta_right)
right_relative = rotation_1.inv() * rotation_2
delta_theta_right = right_relative.get_rotation_vector()
print("Delta Theta (right):\n", delta_theta_right)

# Verify (2.16): Delta theta = triad(psi_1) * Delta Theta  (depending on conventions)
delta_theta_left_from_right = rotation_1 * delta_theta_right
print("Delta theta from (2.16) using right increment:\n", delta_theta_left_from_right)

# Check transformation relation
T = rotation_1.get_transformation_matrix()
delta_psi_via_T = T @ delta_theta_left
print("Delta psi via T(psi_1) * Delta theta:\n", delta_psi_via_T)

err = np.linalg.norm(delta_psi_via_T - delta_psi)
print("||Delta psi via T - Delta psi||:\n", err)
# Comment: The reason for the missmatch is that relation (2.22) is only valid for infinitesimal increments.
# === SOLUTION END: Ex 1.5 ===
