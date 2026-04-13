from bioptim import (
    OptimalControlProgram,
    DynamicsOptions,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    Solver,
    TorqueBiorbdModel,
    ControlType,
    PhaseDynamics,
    OrderingStrategy,
    SolutionMerge,
)

from bioptim.examples.utils import ExampleUtils

import numpy as np

from casadi import DM, Function
import biorbd


def vec3_to_list(v):
    for names in (("x", "y", "z"), ("X", "Y", "Z")):
        try:
            vals = []
            for n in names:
                a = getattr(v, n)
                vals.append(float(a() if callable(a) else a))
            return vals
        except Exception:
            pass
    raise RuntimeError(f"Could not read Vector3d coordinates from object of type {type(v)}")


def prepare_ocp(
    biorbd_model_path: str,
    final_time: float,
    n_shooting: int,
    ode_solver: OdeSolverBase = OdeSolver.RK4(),
    use_sx: bool = True,
    n_threads: int = 1,
    phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
    expand_dynamics: bool = True,
    control_type: ControlType = ControlType.CONSTANT,
    ordering_strategy: OrderingStrategy = OrderingStrategy.VARIABLE_MAJOR,
) -> OptimalControlProgram:

    # bio_model = TorqueBiorbdModel(
    #     biorbd_model_path,
    #     rigid_body_segment="Seg1",
    #     rigid_translation=[1, 2, -1],
    #     rigid_rotation=[0, 0, 0],
    # )

    model_ref = TorqueBiorbdModel(biorbd_model_path)
    model_shift = TorqueBiorbdModel(
        biorbd_model_path,
        rigid_body_segment="Seg1",
        rigid_translation=[1, 2, -1],
        rigid_rotation=[0.2, 0.5, -0.3],
    )

    # Use the symbolic q already created by BiorbdModel
    q_ref_sym = model_ref.q
    q_shift_sym = model_shift.q

    jcs_ref_expr = model_ref.model.globalJCS(q_ref_sym, 0).to_mx()
    jcs_shift_expr = model_shift.model.globalJCS(q_shift_sym, 0).to_mx()

    f_ref = Function("f_ref", [q_ref_sym], [jcs_ref_expr])
    f_shift = Function("f_shift", [q_shift_sym], [jcs_shift_expr])

    q0_ref = np.zeros((model_ref.nb_q, 1))
    q0_shift = np.zeros((model_shift.nb_q, 1))

    jcs_ref = np.array(f_ref(q0_ref))
    jcs_shift = np.array(f_shift(q0_shift))

    print("ref globalJCS:\n", jcs_ref)
    print("shift globalJCS:\n", jcs_shift)
    print("translation diff:", jcs_shift[:3, 3] - jcs_ref[:3, 3])

    marker_ref_expr = model_ref.model.markers(q_ref_sym)[0].to_mx()
    marker_shift_expr = model_shift.model.markers(q_shift_sym)[0].to_mx()

    f_marker_ref = Function("f_marker_ref", [q_ref_sym], [marker_ref_expr])
    f_marker_shift = Function("f_marker_shift", [q_shift_sym], [marker_shift_expr])

    marker_ref = np.array(f_marker_ref(q0_ref)).squeeze()
    marker_shift = np.array(f_marker_shift(q0_shift)).squeeze()

    print("ref marker:", marker_ref)
    print("shift marker:", marker_shift)
    print("marker diff:", marker_shift[:3] - marker_ref[:3])

    # Add objective functions
    objective_functions = Objective(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau")

    # DynamicsOptions
    dynamics = DynamicsOptions(
        ode_solver=ode_solver,
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
    )

    # Path bounds
    x_bounds = BoundsList()
    x_bounds["q"] = model_shift.bounds_from_ranges("q")
    x_bounds["q"][:, [0, -1]] = 0  # Start and end at 0...
    x_bounds["q"][1, -1] = 3.14  # ...but end with pendulum 180 degrees rotated
    x_bounds["qdot"] = model_shift.bounds_from_ranges("qdot")
    x_bounds["qdot"][:, [0, -1]] = 0  # Start and end without any velocity

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    x_init = InitialGuessList()
    x_init["q"] = [0] * model_shift.nb_q
    x_init["qdot"] = [0] * model_shift.nb_qdot

    # Define control path bounds
    n_tau = model_shift.nb_tau
    u_bounds = BoundsList()
    u_bounds["tau"] = [-100] * n_tau, [100] * n_tau  # Limit the strength of the pendulum to (-100 to 100)...
    u_bounds["tau"][1, :] = 0  # ...but remove the capability to actively rotate

    # Initial guess (optional since it is 0, we show how to initialize anyway)
    u_init = InitialGuessList()
    u_init["tau"] = [0] * n_tau

    return OptimalControlProgram(
        model_shift,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        control_type=control_type,
        use_sx=use_sx,
        n_threads=n_threads,
        ordering_strategy=ordering_strategy,
    )


def main():
    """
    If pendulum is run as a script, it will perform the optimization and animates it
    """

    # --- Prepare the ocp --- #
    biorbd_model_path = ExampleUtils.folder + "/models/pendulum.bioMod"
    ocp = prepare_ocp(biorbd_model_path=biorbd_model_path, final_time=1, n_shooting=100, n_threads=2)

    solver = Solver.IPOPT()
    sol = ocp.solve(solver)

    sol.graphs()

    q = sol.decision_states(to_merge=SolutionMerge.NODES)["q"]

    import pyorerun
    import numpy as np

    viz = pyorerun.PhaseRerun(t_span=np.concatenate(sol.decision_time()).squeeze())
    viz.add_animated_model(pyorerun.BiorbdModel(biorbd_model_path), q=q)

    viz.rerun("double_pendulum")


if __name__ == "__main__":
    main()
