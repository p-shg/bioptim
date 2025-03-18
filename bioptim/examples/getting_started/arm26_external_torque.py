from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    Dynamics,
    DynamicsFcn,
    BoundsList,
    OdeSolver,
    OdeSolverBase,
    Solver,
    PhaseDynamics,
    ExternalForceSetTimeSeries,
)

import numpy as np


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = False,
    phase_dynamics: PhaseDynamics = PhaseDynamics.ONE_PER_NODE,
    expand_dynamics: bool = True,
) -> OptimalControlProgram:

    external_force_set = ExternalForceSetTimeSeries(nb_frames=n_shooting)

    external_torque_elbow = np.zeros((6, n_shooting))
    external_torque_elbow[3:, :] = -10

    external_force_set.add_in_segment_frame(
        "r_ulna_radius_hand",
        external_torque_elbow,
        point_of_application_in_local=np.array([[0.0, 0.0, 0.0] for _ in range(n_shooting)]).T,
    )

    numerical_time_series = {"external_forces": external_force_set.to_numerical_time_series()}

    bio_model = BiorbdModel(biorbd_model_path, external_force_set=external_force_set)

    # Dynamics
    dynamics = Dynamics(
        DynamicsFcn.TORQUE_DRIVEN,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        numerical_data_timeseries=numerical_time_series,
        state_continuity_weight=1,
    )

    # Path constraint
    x_bounds = BoundsList()
    x_bounds["q"] = bio_model.bounds_from_ranges("q")
    x_bounds["q"][:, 0] = 0  # Initial pos
    x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, 0] = 0  # No initial speed

    # Define control path constraint
    u_bounds = BoundsList()
    u_bounds["tau"] = [-0] * bio_model.nb_tau, [0] * bio_model.nb_tau  # No control torque, only external

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        final_time,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        ode_solver=ode_solver,
        use_sx=use_sx,
    )


def main():
    ocp = prepare_ocp(
        biorbd_model_path="../muscle_driven_ocp/models/arm26.bioMod",
        final_time=2,
        n_shooting=50,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    sol.graphs()
    sol.animate()


if __name__ == "__main__":
    main()
