"""This file contains functionality for the lecture lab."""

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import yaml
from beamme.core.boundary_condition import BoundaryCondition
from beamme.core.conf import bme
from beamme.core.function import Function
from beamme.four_c.header_functions import set_header_static, set_runtime_output
from beamme.four_c.input_file import InputFile
from beamme.four_c.model_importer import import_four_c_model
from beamme.four_c.run_four_c import clean_simulation_directory
from beamme.mesh_creation_functions.beam_line import create_beam_mesh_line
from beamme.utils.environment import is_mybinder
from IPython.display import display
from ipywidgets import Button, HBox, Label, Output, VBox


def create_beam_mesh_line_2d(
    mesh, beam_class, material, start_point, end_point, **kwargs
):
    """Create a 2D line beam mesh."""
    start_point_3d = np.array([start_point[0], start_point[1], 0.0])
    end_point_3d = np.array([end_point[0], end_point[1], 0.0])
    beam_set = create_beam_mesh_line(
        mesh,
        beam_class,
        material,
        start_point_3d,
        end_point_3d,
        set_nodal_arc_length=True,
        **kwargs,
    )
    mesh.add(
        BoundaryCondition(
            beam_set["line"],
            {
                "NUMDOF": 6,
                "ONOFF": [0, 0, 1, 1, 1, 0],
                "VAL": [0, 0, 0, 0, 0, 0],
                "FUNCT": [0, 0, 0, 0, 0, 0],
            },
            bc_type=bme.bc.dirichlet,
        )
    )
    return beam_set


def create_boundary_condition_2d(
    mesh,
    geometry_set,
    *,
    directions=None,
    bc_type=None,
    values=None,
    linear_increase=True,
):
    """Create a boundary condition for plane beams."""

    if values is not None and not len(directions) == len(values):
        raise ValueError("directions and values must have the same length")

    if values is None:
        values = [0.0] * len(directions)

    value_bc = [0, 0, 0, 0, 0, 0]
    on_off = [0, 0, 0, 0, 0, 0]
    if "x" in directions:
        on_off[0] = 1
        value_bc[0] = values[directions.index("x")]
    if "y" in directions:
        on_off[1] = 1
        value_bc[1] = values[directions.index("y")]
    if "theta" in directions:
        on_off[5] = 1
        value_bc[5] = values[directions.index("theta")]

    if bc_type == "dirichlet":
        bc_type_value = bme.bc.dirichlet
        function_string = (
            "SPACE_TIME" if geometry_set.geometry_type == bme.geo.point else "TIME"
        )
    elif bc_type == "neumann":
        bc_type_value = bme.bc.neumann
        function_string = "TIME"
    else:
        raise ValueError("bc_type must be either 'dirichlet' or 'neumann'")

    if not linear_increase:
        function_bc = [0, 0, 0, 0, 0, 0]
    else:
        function_load = Function([{f"SYMBOLIC_FUNCTION_OF_{function_string}": "t"}])
        mesh.add(function_load)
        function_bc = [function_load] * 6

    mesh.add(
        BoundaryCondition(
            geometry_set,
            {"NUMDOF": 6, "ONOFF": on_off, "VAL": value_bc, "FUNCT": function_bc},
            bc_type=bc_type_value,
        )
    )


def run_four_c(
    *,
    mesh=None,
    simulation_name=None,
    total_time=1.0,
    n_steps=1,
    tol=1e-10,
    display_log=True,
):
    """Run a 4C simulation with given parameters."""

    # Setup the input file with mesh and parameters.
    input_file = InputFile()
    input_file.add(mesh)
    set_header_static(
        input_file,
        total_time=total_time,
        n_steps=n_steps,
        max_iter=50,
        tol_residuum=1.0,
        tol_increment=tol,
        create_nox_file=False,
        predictor="TangDis",
    )
    set_runtime_output(
        input_file,
        output_solid=False,
        output_stress_strain=False,
        btsvmt_output=False,
        btss_output=False,
        output_triad=True,
        every_iteration=False,
        absolute_beam_positions=True,
        element_owner=True,
        element_gid=True,
        element_mat_id=True,
        output_energy=False,
        output_strains=True,
    )
    input_file["IO/RUNTIME VTK OUTPUT/BEAMS"]["MATERIAL_FORCES_GAUSSPOINT"] = True
    input_file["IO/RUNTIME VTK OUTPUT/BEAMS"]["NUMBER_SUBSEGMENTS"] = 1
    input_file["IO/MONITOR STRUCTURE DBC"] = {
        "INTERVAL_STEPS": 1,
        "WRITE_CONDITION_INFORMATION": True,
        "FILE_TYPE": "yaml",
    }

    # Dump the file to disc.
    simulation_directory = Path.cwd() / "simulations" / simulation_name
    input_file_path = simulation_directory / f"{simulation_name}.4C.yaml"
    clean_simulation_directory(simulation_directory)
    input_file.dump(input_file_path)

    # Create file that maps the global node IDs to the beam arc length.
    arc_length_file_path = simulation_directory / f"{simulation_name}_arc_length.yaml"
    data = {}
    for element in mesh.elements:
        data[element.i_global] = [
            float(element.nodes[node_index].arc_length) for node_index in (0, -1)
        ]
    with open(arc_length_file_path, "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    # Run the simulation and process the results line by line.
    with open(simulation_directory / f"{simulation_name}.log", "w") as logfile:
        # Command to run 4C
        if is_mybinder():
            four_c_exe = "/home/user/4C/build/4C"
        else:
            four_c_exe = "/data/a11bivst/dev/4C/release/4C"
        command = [four_c_exe, input_file_path.absolute(), simulation_name]

        # Start simulation
        if display_log:
            print("Start simulation")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout (optional)
            text=True,  # get str instead of bytes
            bufsize=1,  # line-buffered
            cwd=simulation_directory,
        )

        # Process the output line by line
        nonlinear_solver_step_count = 0
        is_error = False
        finished = False
        for line in process.stdout:
            if display_log:
                line = line.rstrip("\n")

                # Write line to logfile
                logfile.write(line + "\n")

                # Flush file so log is always up to date
                logfile.flush()

                # Process the line however you want
                if "Nonlinear Solver Step" in line:
                    nonlinear_solver_step_count = int(line.split(" ")[4])
                elif "||F||" in line:
                    if not nonlinear_solver_step_count == 0:
                        residuum = float(line.split(" ")[10])
                        print(
                            f"  Nonlinear Solver Step {nonlinear_solver_step_count}: Residuum = {residuum:.3e}"
                        )
                elif "Finalised step" in line:
                    split = line.split(" ")
                    step = int(split[2])
                    time = float(split[7])
                    print(f"Finished time step {step} for time {time:.3e}")
                elif "OK (0)" in line:
                    finished = True
                elif (
                    "========================================================================="
                    in line
                    and not finished
                ):
                    if is_error:
                        print(line)
                    is_error = not is_error

                if is_error:
                    print(line)

        _return_code = process.wait()


def plot_beam_2d(simulation_name):
    """Plot the beam deformation in 2D (XY-plane) over time steps."""

    # Load the result file
    simulation_path = Path.cwd() / "simulations" / simulation_name
    output_file = simulation_path / f"{simulation_name}-structure-beams.pvd"
    reader = pv.get_reader(output_file)

    # Load the element arc lengths
    with open(simulation_path / f"{simulation_name}_arc_length.yaml") as stream:
        arc_length_data = yaml.safe_load(stream)

    # Load the data for all time steps
    steps = reader.time_values
    grid_points = []
    cross_section_resultants_map = {
        "material_axial_force_GPs": "normal",
        "material_bending_moment_3_GPs": "bending",
        "material_shear_force_2_GPs": "shear",
    }
    cross_section_resultants_labels = {
        "normal": "Axial force",
        "bending": "Bending moment",
        "shear": "Shear force",
    }
    cell_data = {
        "normal": [],
        "bending": [],
        "shear": [],
    }
    for i_step, time in enumerate(steps):
        reader.set_active_time_point(i_step)
        mesh = reader.read()[0]
        grid_points.append(mesh.points.copy())
        for name in mesh.cell_data.keys():
            if name in cross_section_resultants_map:
                data_name = cross_section_resultants_map[name]
                cell_data[data_name].append(np.array(mesh.cell_data[name]))

    # Convert lists to numpy arrays for easier indexing
    grid_points = np.array(grid_points)
    for key in cell_data:
        data_from_pv = cell_data[key]
        data_time_steps = []
        for time_step in range(len(data_from_pv)):
            data_for_plot = []
            for i_element, value in enumerate(data_from_pv[time_step]):
                element_center = 0.5 * (
                    arc_length_data[i_element][1] + arc_length_data[i_element][0]
                )
                data_for_plot.append([element_center, value])
            data_time_steps.append(data_for_plot)
        cell_data[key] = np.array(data_time_steps)

    # Get the bounds for the displacement plot
    upper_bound = np.max(grid_points, axis=(0, 1))
    lower_bound = np.min(grid_points, axis=(0, 1))
    dimension = np.linalg.norm(upper_bound - lower_bound)

    # Check that the simulation is in plane
    if not np.isclose(upper_bound[2] - lower_bound[2], 0.0):
        raise ValueError(
            "The beam is not in the XY-plane. This plotting function only supports beams in the XY-plane."
        )

    #

    # -------------------------------------------------------
    # Output widget for the plot
    # -------------------------------------------------------
    out = Output()
    state = {"step": 0}

    def plot_step():
        """Plot the current step."""
        step = state["step"]
        with out:
            out.clear_output(wait=True)
            fig = plt.figure(figsize=(10, 6))
            gs = fig.add_gridspec(3, 2, width_ratios=[3, 1])

            # First plot the beam deformation in XZ-plane
            ax0 = fig.add_subplot(gs[:, 0])
            ax0.plot(
                grid_points[state["step"], :, 0],
                grid_points[state["step"], :, 1],
                "-o",
                linewidth=3,
            )
            ax0.set_title(f"Beam configuration in space at time {steps[step]:.3f}")
            ax0.set_xlim(
                lower_bound[0] - dimension * 0.1, upper_bound[0] + dimension * 0.1
            )
            ax0.set_ylim(
                lower_bound[1] - dimension * 0.1, upper_bound[1] + dimension * 0.1
            )
            ax0.grid(True)
            ax0.set_xlabel("X")
            ax0.set_ylabel("Y")
            ax0.set_aspect("equal", "box")

            # Plot the cross-section resultants
            ax = [
                fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 1]),
                fig.add_subplot(gs[2, 1]),
            ]
            color = ["b", "g", "r"]
            for i, (name, data) in enumerate(cell_data.items()):
                data = data[state["step"]]
                ax[i].plot(data[:, 0], data[:, 1], f"{color[i]}x-")
                data_no_nan = data[~np.isnan(data[:, 1]), 1]
                y_bounds = [
                    np.min([0.0, np.min(data_no_nan)]),
                    np.max([0.0, np.max(data_no_nan)]),
                ]
                size = np.max([0.1, y_bounds[1] - y_bounds[0]])
                ax[i].set_ylim(y_bounds[0] - size * 0.1, y_bounds[1] + size * 0.1)
                ax[i].set_title(f"{cross_section_resultants_labels[name]} along beam")
                ax[i].grid(True)

            plt.tight_layout()
            plt.show()

    # -------------------------------------------------------
    # Step label
    # -------------------------------------------------------
    step_label = Label(f"Step: {state['step']}")

    def update_step_label():
        """Update the label showing the current step."""
        step_label.value = f"Step: {state['step']}"

    # -------------------------------------------------------
    # Buttons + logic
    # -------------------------------------------------------
    btn_first = Button(description="⏮ First")
    btn_prev = Button(description="◀ Prev")
    btn_next = Button(description="Next ▶")
    btn_last = Button(description="Last ⏭")

    def go_first(b):
        """Go to the first step."""
        state["step"] = 0
        update_step_label()
        plot_step()

    def go_prev(b):
        """Go to the previous step."""
        if state["step"] > 0:
            state["step"] -= 1
        update_step_label()
        plot_step()

    def go_next(b):
        """Go to the next step."""
        if state["step"] < len(steps) - 1:
            state["step"] += 1
        update_step_label()
        plot_step()

    def go_last(b):
        """Go to the last step."""
        state["step"] = len(steps) - 1
        update_step_label()
        plot_step()

    btn_first.on_click(go_first)
    btn_prev.on_click(go_prev)
    btn_next.on_click(go_next)
    btn_last.on_click(go_last)

    # -------------------------------------------------------
    # Layout (buttons row, label row, then plot)
    # -------------------------------------------------------
    controls = HBox([btn_first, btn_prev, btn_next, btn_last])
    # labels = HBox([step_label])

    ui = VBox([controls, out])

    # Show initial plot
    plot_step()
    display(ui)


def get_force_displacement_data(simulation_name, point_coordinates):
    """Get the force and displacement data from the simulation results.

    The displacement is taken from the node at the given position.
    """

    simulation_path = Path.cwd() / "simulations" / simulation_name

    # Load the input file and extract the applied force.
    input_file_path = simulation_path / f"{simulation_name}.4C.yaml"
    input_file, _ = import_four_c_model(input_file_path)
    point_neumann_loads = input_file.fourc_input["DESIGN POINT NEUMANN CONDITIONS"]
    counter = 0
    for load in point_neumann_loads:
        if not np.all(np.array(load["FUNCT"]) == 0):
            force_value = np.linalg.norm(load["VAL"])
            # point_set = load["E"]
            counter += 1
    if counter != 1:
        raise ValueError(
            "Expected exactly one non-zero Neumann load in the input file."
        )

    # Load the result file
    output_file = simulation_path / f"{simulation_name}-structure-beams.pvd"
    reader = pv.get_reader(output_file)

    # Load the data for all time steps
    steps = reader.time_values
    displacement = []
    force = []
    for i_step, time in enumerate(steps):
        reader.set_active_time_point(i_step)
        mesh = reader.read()[0]
        reference_coordinates = mesh.points - mesh.point_data["displacement"]
        difference = reference_coordinates - [
            point_coordinates[0],
            point_coordinates[1],
            0.0,
        ]
        distances = np.linalg.norm(difference, axis=1)
        closest_point_index = np.argmin(distances)
        if distances[closest_point_index] > 1e-6:
            raise ValueError(
                f"Could not find a node close to the given position {point_coordinates}."
            )
        displacement.append(mesh.point_data["displacement"][closest_point_index][:2])
        force.append(force_value * time)

    return np.array(force), np.array(displacement)


def get_displacement_data(simulation_name, point_coordinates):
    """The displacement is taken from the node at the given position."""

    _, displacement = get_force_displacement_data(simulation_name, point_coordinates)
    return displacement[-1]
