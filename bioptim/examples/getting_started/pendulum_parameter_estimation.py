"""
This example is a clone of the getting_started/pendulum.py example with the difference that the
model now evolves in an environment where the gravity can be modified.
The goal of the solver it to find the optimal gravity (target = 8 N/kg), while performing the
pendulum balancing task
It is designed to show how one can define its own parameter objective functions if the provided ones are not
sufficient.
"""

import numpy as np
from casadi import MX
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    InterpolationType,
    ParameterList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    ParameterObjectiveList,
    ObjectiveList,
    PhaseDynamics,
    VariableScaling,
    SolutionMerge,
    PenaltyController,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from example_parameter_scaling import generate_dat_to_track

estimate_mass = True
mass_scale = np.array([1.0])
estimate_gravity = True
gravity_scale = np.array([1.0, 1.0, 1.0])


def visualize_data(t, data_ref, data, title, labels):
    """
    Visualizes the reference and actual data in a single interactive Plotly figure.

    Args:
        t (array): Time array.
        data_ref (array): Reference data (2D or 3D).
        data (array): Actual data (2D or 3D).
        title (str): Title for the plot.
        labels (list): Labels for each series, e.g., ["ref", "actual"].
        colormap (str): Colormap for line colors.
        muscle_names (list): List of muscle names if plotting individual muscle activations.
    """

    # print(data_ref)
    # print(data)
    # Ensure time array is 1D
    if t.ndim > 1:
        t = t.flatten()

    # Standard visualization for non-muscle data
    fig = make_subplots(rows=1, cols=1, subplot_titles=[title])
    num_series = data.shape[1] if len(data.shape) > 2 else data.shape[0]

    for i in range(num_series):
        # color = f"rgb({colors(i)[0]*255:.0f}, {colors(i)[1]*255:.0f}, {colors(i)[2]*255:.0f})"
        y_ref = data_ref[:, i, :].flatten() if len(data_ref.shape) > 2 else data_ref[i]
        y_data = data[:, i, :].flatten() if len(data.shape) > 2 else data[i]
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_ref,
                mode="lines",
                name=f"{labels[0]} {i}",
                # line=dict(color=color, width=2),
                line=dict(width=2),
                opacity=0.5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y_data,
                mode="lines",
                name=f"{labels[1]} {i}",
                # line=dict(color=color, dash="dash", width=4),
                line=dict(dash="dash", width=4),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        showlegend=True,
    )

    return fig


def set_mass_function(bio_model: BiorbdModel, value: MX):
    """
    The pre dynamics function is called right before defining the dynamics of the system. If one wants to
    modify the dynamics (e.g. optimize the gravity in this case), then this function is the proper way to do it.

    Parameters
    ----------
    bio_model: BiorbdModel
        The model to modify by the parameters
    value: MX
        The CasADi variables to modify the model
    """

    bio_model.segments[0].characteristics().setMass(value)


def set_gravity_function(bio_model: BiorbdModel, value: MX):
    bio_model.set_gravity(value)


def my_target_function(controller: PenaltyController, key: str) -> MX:
    return controller.parameters[key].cx


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    tau_ref,
    q_ref,
    qdot_ref,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:
    """
    Prepare the program

    Parameters
    ----------
    biorbd_model_path: str
        The path of the biorbd model
    final_time: float
        The time at the final node
    n_shooting: int
        The number of shooting points
    optim_gravity: bool
        If the gravity should be optimized
    optim_mass: bool
        If the mass should be optimized
    min_g: np.ndarray
        The minimal value for the gravity
    max_g: np.ndarray
        The maximal value for the gravity
    target_g: np.ndarray
        The target value for the gravity
    min_m: float
        The minimal value for the mass
    max_m: float
        The maximal value for the mass
    target_m: float
        The target value for the mass
    ode_solver: OdeSolverBase
        The type of ode solver used
    use_sx: bool
        If the program should be constructed using SX instead of MX (longer to create the CasADi graph, faster to solve)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)

    Returns
    -------
    The ocp ready to be solved
    """

    # Define the parameter to optimize
    parameters = ParameterList(use_sx=use_sx)
    parameter_objectives = ParameterObjectiveList()
    parameter_bounds = BoundsList()
    parameter_init = InitialGuessList()

    if estimate_mass:
        min_m, max_m, init_m = 0.05 / mass_scale, 2.0 / mass_scale, 0.5 / mass_scale
        m_scaling = VariableScaling("mass", np.array([mass_scale]))
        parameters.add(
            "mass",  # The name of the parameter
            set_mass_function,  # The function that modifies the biorbd model
            size=1,  # The number of elements this particular parameter vector has
            scaling=m_scaling,  # The scaling of the parameter
        )

        parameter_bounds.add("mass", min_bound=[min_m], max_bound=[max_m], interpolation=InterpolationType.CONSTANT)

        parameter_init["mass"] = init_m

    if estimate_gravity:
        min_g = np.array([0, -5, -50])
        max_g = np.array([0, 5, -5])

        g_scaling = VariableScaling("gravity_xyz", gravity_scale)
        parameters.add(
            "gravity_xyz",  # The name of the parameter
            set_gravity_function,  # The function that modifies the biorbd model
            size=3,  # The number of elements this particular parameter vector has
            scaling=g_scaling,  # The scaling of the parameter
        )

        # Give the parameter some min and max bounds and initial conditions
        parameter_bounds.add("gravity_xyz", min_bound=min_g, max_bound=max_g, interpolation=InterpolationType.CONSTANT)
        parameter_init["gravity_xyz"] = np.array([0.0, 0.5, -15])

        # parameter_objectives.add(
        #     my_target_function,
        #     weight=1,
        #     # quadratic=True,
        #     custom_type=ObjectiveFcn.Parameter,
        #     key="gravity_xyz",
        # )

    # --- Options --- #
    bio_model = BiorbdModel(biorbd_model_path, parameters=parameters)

    if not estimate_mass:
        set_mass_function(bio_model, 0.234)
    if not estimate_gravity:
        set_gravity_function(bio_model, np.array([0.0, 1.0, -20.0]))

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Lagrange.TRACK_CONTROL, key="tau", target=np.array([tau_ref[0]]), weight=1, index=0
    )
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", target=q_ref[:, :-1], weight=1)
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="qdot", target=qdot_ref[:, :-1], weight=1)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        # state_continuity_weight=1000,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, [0]] = 0
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0]] = 0

    # Define control path constraint
    u_bounds = BoundsList()
    # u_bounds.add("tau", min_bound=tau_ref, max_bound=tau_ref, interpolation=InterpolationType.EACH_FRAME)

    # Define initial guesses
    x_init = InitialGuessList()
    x_init.add("q", q_ref, interpolation=InterpolationType.EACH_FRAME)
    x_init.add("qdot", qdot_ref, interpolation=InterpolationType.EACH_FRAME)

    u_init = InitialGuessList()
    u_init.add("tau", tau_ref, interpolation=InterpolationType.EACH_FRAME)

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=x_init,
        u_init=u_init,
        objective_functions=objective_functions,
        parameters=parameters,
        parameter_objectives=parameter_objectives,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        ode_solver=ode_solver,
        use_sx=use_sx,
    )


def main():
    """
    Solve and print the optimized value for the gravity and animate the solution
    """
    final_time = 5
    n_shooting = 1000

    ocp_to_track = generate_dat_to_track(
        biorbd_model_path="models/pendulum_wrong_gravity.bioMod", final_time=final_time, n_shooting=n_shooting
    )
    sol_to_track = ocp_to_track.solve(Solver.IPOPT(show_online_optim=False))
    q_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot_to_track = sol_to_track.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau_to_track = sol_to_track.decision_controls(to_merge=SolutionMerge.NODES)["tau"]

    # q_ref = np.loadtxt("q.csv", delimiter=",")
    # qdot_ref = np.loadtxt("qdot.csv", delimiter=",")
    # tau_ref = np.loadtxt("tau.csv", delimiter=",")

    ocp = prepare_ocp(
        biorbd_model_path="models/pendulum.bioMod",
        final_time=final_time,
        n_shooting=n_shooting,
        tau_ref=tau_to_track,
        q_ref=q_to_track,
        qdot_ref=qdot_to_track,
    )

    # --- Solve the program --- #
    solver = Solver.IPOPT(show_online_optim=False)
    solver.set_convergence_tolerance(1e-8)
    sol = ocp.solve(solver)

    if estimate_mass:
        print("Mass parameter found: ", sol.parameters["mass"] * mass_scale)
    if estimate_gravity:
        print("gravity parameter found: ", sol.parameters["gravity_xyz"])

    # --- Show the results --- #
    t = np.linspace(0, final_time, n_shooting)  # Normalized time array
    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]
    qdot = sol.decision_states(to_merge=SolutionMerge.NODES)["qdot"]
    tau = sol.decision_controls(to_merge=SolutionMerge.NODES)["tau"]

    # Visualize q
    fig_q = visualize_data(t=t, data_ref=q_to_track, data=q, title="q Comparison", labels=["q_ref", "q"])

    # Visualize qdot
    fig_qdot = visualize_data(
        t=t,
        data_ref=qdot_to_track,
        data=qdot,
        title="qdot Comparison",
        labels=["qdot_ref", "qdot"],
    )

    # Visualize muscle activations
    fig_tau = visualize_data(
        t=t,
        data_ref=tau_to_track,
        data=np.array([tau[0]]),
        title="Torque Comparison",
        labels=["tau_ref", "tau"],
    )

    # Show figures
    fig_q.show()
    fig_qdot.show()
    fig_tau.show()

    # --- Show results --- #
    # sol.graphs()
    sol.animate(n_frames=200)


if __name__ == "__main__":
    main()
