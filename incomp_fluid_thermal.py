# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # A divergence conforming discontinuous Galerkin method for the Navier-Stokes equations
# This demo illustrates how to implement a divergence conforming
# discontinuous Galerkin method for the Navier-Stokes equations in
# FEniCSx. The method conserves mass exactly and uses upwinding. The
# formulation is based on a combination of "A fully divergence-free
# finite element method for magnetohydrodynamic equations" by Hiptmair
# et al., "A Note on Discontinuous Galerkin Divergence-free Solutions
# of the Navier-Stokes Equations" by Cockburn et al, and "On the Divergence
# Constraint in Mixed Finite Element Methods for Incompressible Flows" by
# John et al.

# ## Governing equations
# We consider the incompressible Navier-Stokes equations in a domain
# $\Omega \subset \mathbb{R}^d$, $d \in \{2, 3\}$, and time interval
# $(0, \infty)$, given by
# $$
#     \partial_t u - \nu \Delta u + (u \cdot \nabla)u + \nabla p = f
#     \textnormal{ in } \Omega_t,
# $$
# $$
#     \nabla \cdot u = 0
#     \textnormal{ in } \Omega_t,
# $$
# where $u: \Omega_t \to \mathbb{R}^d$ is the velocity field,
# $p: \Omega_t \to \mathbb{R}$ is the pressure field,
# $f: \Omega_t \to \mathbb{R}^d$ is a prescribed force, $\nu \in \mathbb{R}^+$
# is the kinematic viscosity, and
# $\Omega_t \coloneqq \Omega \times (0, \infty)$.

# The problem is supplemented with the initial condition
# $$
#     u(x, 0) = u_0(x) \textnormal{ in } \Omega
# $$
# and boundary condition
# $$
#     u = u_D \textnormal{ on } \partial \Omega \times (0, \infty),
# $$
# where $u_0: \Omega \to \mathbb{R}^d$ is a prescribed initial velocity field
# which satisfies the divergence free condition. The pressure field is only
# determined up to a constant, so we seek the unique pressure field satisfying
# $$
#     \int_\Omega p = 0.
# $$

# ## Discrete problem
# We begin by introducing the function spaces
# $$
#     V_h^g \coloneqq \left\{v \in H(\textnormal{div}; \Omega);
#     v|_K \in V_h(K) \; \forall K \in \mathcal{T}, v \cdot n = g \cdot n
#     \textnormal{ on } \partial \Omega \right\}
# $$,
# and
# $$
#     Q_h \coloneqq \left\{q \in L^2_0(\Omega);
#     q|_K \in Q_h(K) \; \forall K \in \mathcal{T} \right\}.
# $$
# The local spaces $V_h(K)$ and $Q_h(K)$ should satisfy
# $$
#     \nabla \cdot V_h(K) \subseteq Q_h(K),
# $$
# in order for mass to be conserved exactly. Suitable choices on
# affine simplex cells include
# $$
#     V_h(K) \coloneqq \mathbb{RT}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_k(K),
# $$
# or
# $$
#     V_h(K) \coloneqq \mathbb{BDM}_k(K) \textnormal{ and }
#     Q_h(K) \coloneqq \mathbb{P}_{k-1}(K).
# $$

# Let two cells $K^+$ and $K^-$ share a facet $F$. The trace of a piecewise
# smooth vector valued function $\phi$ on F taken approaching from inside $K^+$
# (resp. $K^-$) is denoted $\phi^{+}$ (resp. $\phi^-$). We now introduce the
# average
# $\renewcommand{\avg}[1]{\left\{\!\!\left\{#1\right\}\!\!\right\}}$
# $$
#     \avg{\phi} = \frac{1}{2} \left(\phi^+ + \phi^-\right)
# $$
# $\renewcommand{\jump}[1]{\llbracket #1 \rrbracket}$
# and jump
# $$
#     \jump{\phi} = \phi^+ \otimes n^+ + \phi^- \otimes n^-,
# $$
# operators, where $n$ denotes the outward unit normal to $\partial K$.
# Finally, let the upwind flux of $\phi$ with respect to a vector field
# $\psi$ be defined as
# $$
#     \hat{\phi}^\psi \coloneqq
#     \begin{cases}
#         \lim_{\epsilon \downarrow 0} \phi(x - \epsilon \psi(x)), \;
#         x \in \partial K \setminus \Gamma^\psi, \\
#         0, \qquad \qquad \qquad \qquad x \in \partial K \cap \Gamma^\psi,
#     \end{cases}
# $$
# where $\Gamma^\psi = \left\{x \in \Gamma; \; \psi(x) \cdot n(x) < 0\right\}$.

# The semi-discrete version problem (in dimensionless form) is: find
# $(u_h, p_h) \in V_h^{u_D} \times Q_h$ such that
# $$
#     \int_\Omega \partial_t u_h \cdot v + a_h(u_h, v_h) + c_h(u_h; u_h, v_h)
#     + b_h(v_h, p_h) = \int_\Omega f \cdot v_h + L_{a_h}(v_h) + L_{c_h}(v_h)
#      \quad \forall v_h \in V_h^0,
# $$
# $$
#     b_h(u_h, q_h) = 0 \quad \forall q_h \in Q_h,
# $$
# where
# $\renewcommand{\sumK}[0]{\sum_{K \in \mathcal{T}_h}}$
# $\renewcommand{\sumF}[0]{\sum_{F \in \mathcal{F}_h}}$
# $$
#     a_h(u, v) = Re^{-1} \left(\sumK \int_K \nabla u : \nabla v
#     - \sumF \int_F \avg{\nabla u} : \jump{v}
#     - \sumF \int_F \avg{\nabla v} : \jump{u} \\
#     + \sumF \int_F \frac{\alpha}{h_K} \jump{u} : \jump{v}\right),
# $$
# $$
#     c_h(w; u, v) = - \sumK \int_K u \cdot \nabla \cdot (v \otimes w)
#     + \sumK \int_{\partial_K} w \cdot n \hat{u}^{w} \cdot v,
# $$
# $$
# L_{a_h}(v_h) = Re^{-1} \left(- \int_{\partial \Omega} u_D \otimes n :
#   \nabla_h v_h + \frac{\alpha}{h} u_D \otimes n : v_h \otimes n \right),
# $$
# $$
#     L_{c_h}(v_h) = - \int_{\partial \Omega} u_D \cdot n \hat{u}_D \cdot v_h,
# $$
# and
# $$
#     b_h(v, q) = - \int_K \nabla \cdot v q.
# $$

# ## Implementation
# We begin by importing the required modules and functions

from dolfinx import fem, io
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
from ufl import (TrialFunction, TestFunction, CellDiameter, FacetNormal,
                 inner, grad, dx, dS, avg, outer, div, conditional,
                 gt, dot, Measure, as_vector)
from ufl import jump as jump_T
import mesh_generator

# We also define some helper functions that will be used later


def norm_L2(comm, v):
    """Compute the L2(Ω)-norm of v"""
    return np.sqrt(comm.allreduce(
        fem.assemble_scalar(fem.form(inner(v, v) * dx)), op=MPI.SUM))


def domain_average(msh, v):
    """Compute the average of a function over the domain"""
    vol = msh.comm.allreduce(
        fem.assemble_scalar(fem.form(
            fem.Constant(msh, PETSc.ScalarType(1.0)) * dx)), op=MPI.SUM)
    return 1 / vol * msh.comm.allreduce(
        fem.assemble_scalar(fem.form(v * dx)), op=MPI.SUM)


# We define some simulation parameters

num_time_steps = 100
t_end = 2
R_e = 1000  # Reynolds Number
h = 0.05
h_fac = 1 / 3  # Factor scaling h near the cylinder
k = 2  # Polynomial degree

comm = MPI.COMM_WORLD

# Next, we create a mesh and the required functions spaces over
# it. Since the velocity uses an $H(\textnormal{div})$-conforming function
# space, we also create a vector valued discontinuous Lagrange space
# to interpolate into for artifact free visualisation.

msh, mt, boundary_id = mesh_generator.generate(comm, h=h, h_fac=h_fac)

# Function space for the velocity
V = fem.FunctionSpace(msh, ("Raviart-Thomas", k + 1))
# Function space for the pressure
Q = fem.FunctionSpace(msh, ("Discontinuous Lagrange", k))
# Funcion space for visualising the velocity field
W = fem.VectorFunctionSpace(msh, ("Discontinuous Lagrange", k + 1))

# Define trial and test functions

u, v = TrialFunction(V), TestFunction(V)
p, q = TrialFunction(Q), TestFunction(Q)
T, w = TrialFunction(Q), TestFunction(Q)

delta_t = fem.Constant(msh, PETSc.ScalarType(t_end / num_time_steps))
alpha = fem.Constant(msh, PETSc.ScalarType(6.0 * k**2))
R_e_const = fem.Constant(msh, PETSc.ScalarType(R_e))
kappa = fem.Constant(msh, PETSc.ScalarType(0.01))

# List of tuples of form (id, expression)
dirichlet_bcs = [(boundary_id["inlet"],
                  lambda x: np.vstack(
                    ((1.5 * 4 * x[1] * (0.41 - x[1])) / 0.41**2,
                     np.zeros_like(x[0])))),
                 (boundary_id["wall"], lambda x: np.vstack(
                     (np.zeros_like(x[0]), np.zeros_like(x[0])))),
                 (boundary_id["obstacle"],
                  lambda x: np.vstack((
                    np.zeros_like(x[0]), np.zeros_like(x[0]))))]
neumann_bcs = [(boundary_id["outlet"], fem.Constant(
    msh, np.array([0.0, 0.0], dtype=PETSc.ScalarType)))]

ds = Measure("ds", domain=msh, subdomain_data=mt)

h = CellDiameter(msh)
n = FacetNormal(msh)


def jump(phi, n):
    return outer(phi("+"), n("+")) + outer(phi("-"), n("-"))


# We solve the Stokes problem for the initial condition, so the convective
# terms are omitted for now

a_00 = 1 / R_e_const * (inner(grad(u), grad(v)) * dx
                        - inner(avg(grad(u)), jump(v, n)) * dS
                        - inner(jump(u, n), avg(grad(v))) * dS
                        + alpha / avg(h) * inner(jump(u, n), jump(v, n)) * dS)
a_01 = - inner(p, div(v)) * dx
a_10 = - inner(div(u), q) * dx

f = fem.Function(W)
# NOTE: Arrived at Neumann BC term by rewriting inner(grad(u), outer(v, n))
# it is based on as inner(dot(grad(u), n), v) and then g = dot(grad(u), n)
# etc. TODO Check this. NOTE Consider changing formulation to one with momentum
# law in conservative form to be able to specify momentum flux
L_0 = inner(f, v) * dx
L_1 = inner(fem.Constant(msh, PETSc.ScalarType(0.0)), q) * dx

bcs = []
for bc in dirichlet_bcs:
    a_00 += 1 / R_e_const * (- inner(grad(u), outer(v, n)) * ds(bc[0])
                             - inner(outer(u, n), grad(v)) * ds(bc[0])
                             + alpha / h * inner(
                                outer(u, n), outer(v, n)) * ds(bc[0]))
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += 1 / R_e_const * (- inner(outer(u_D, n), grad(v)) * ds(bc[0])
                            + alpha / h * inner(
                                outer(u_D, n), outer(v, n)) * ds(bc[0]))

    bc_boundary_facets = mt.indices[mt.values == bc[0]]
    bc_dofs = fem.locate_dofs_topological(
        V, msh.topology.dim - 1, bc_boundary_facets)
    bcs.append(fem.dirichletbc(u_D, bc_dofs))


for bc in neumann_bcs:
    L_0 += 1 / R_e_const * inner(bc[1], v) * ds(bc[0])

a = fem.form([[a_00, a_01],
              [a_10, None]])
L = fem.form([L_0,
              L_1])

# Assemble Stokes problem

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Create and configure solver

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
opts = PETSc.Options()
# See https://graal.ens-lyon.fr/MUMPS/doc/userguide_5.5.1.pdf
# TODO Check
opts["mat_mumps_icntl_6"] = 2
opts["mat_mumps_icntl_14"] = 100
opts["ksp_error_if_not_converged"] = 1

if len(neumann_bcs) == 0:
    # Options to support solving a singular matrix (pressure nullspace)
    opts["mat_mumps_icntl_24"] = 1
    opts["mat_mumps_icntl_25"] = 0
ksp.setFromOptions()

# Solve Stokes for initial condition

x = A.createVecRight()
ksp.solve(b, x)

# Split the solution

u_h = fem.Function(V)
p_h = fem.Function(Q)
p_h.name = "p"
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
u_h.x.array[:offset] = x.array_r[:offset]
u_h.x.scatter_forward()
p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
p_h.x.scatter_forward()
if len(neumann_bcs) == 0:
    p_h.x.array[:] -= domain_average(msh, p_h)

u_vis = fem.Function(W)
u_vis.name = "u"
u_vis.interpolate(u_h)

# Write initial condition to file

u_file = io.VTXWriter(msh.comm, "u.bp", [u_vis._cpp_object])
p_file = io.VTXWriter(msh.comm, "p.bp", [p_h._cpp_object])
t = 0.0
u_file.write(t)
p_file.write(t)

T_n = fem.Function(Q)

dirichlet_bcs_T = [(boundary_id["inlet"], lambda x: np.zeros_like(x[0]))]
neumann_bcs_T = [(boundary_id["outlet"], lambda x: np.zeros_like(x[0]))]
robin_bcs_T = [(boundary_id["wall"], (0.0, 0.0)),
               (boundary_id["obstacle"], (1.0, 1.0))]


# Create function to store solution and previous time step

u_n = fem.Function(V)
u_n.x.array[:] = u_h.x.array

lmbda = conditional(gt(dot(u_n, n), 0), 1, 0)

# FIXME Use u_h not u_n
a_T = inner(T / delta_t, w) * dx - \
    inner(u_h * T, grad(w)) * dx + \
    inner(lmbda("+") * dot(u_h("+"), n("+")) * T("+") -
          lmbda("-") * dot(u_h("-"), n("-")) * T("-"), jump_T(w)) * dS + \
    inner(lmbda * dot(u_h, n) * T, w) * ds + \
    kappa * (inner(grad(T), grad(w)) * dx -
             inner(avg(grad(T)), jump_T(w, n)) * dS -
             inner(jump_T(T, n), avg(grad(w))) * dS +
             (alpha / avg(h)) * inner(jump_T(T, n), jump_T(w, n)) * dS)

L_T = inner(T_n / delta_t, w) * dx

for bc in dirichlet_bcs_T:
    T_D = fem.Function(Q)
    T_D.interpolate(bc[1])
    a_T += kappa * (- inner(grad(T), w * n) * ds(bc[0]) -
                    inner(grad(w), T * n) * ds(bc[0]) +
                    (alpha / h) * inner(T, w) * ds(bc[0]))
    L_T += - inner((1 - lmbda) * dot(u_h, n) * T_D, w) * ds(bc[0]) + \
        kappa * (- inner(T_D * n, grad(w)) * ds(bc[0]) +
                 (alpha / h) * inner(T_D, w) * ds(bc[0]))

for bc in neumann_bcs_T:
    g_T = fem.Function(Q)
    g_T.interpolate(bc[1])
    L_T += kappa * inner(g_T, w) * ds(bc[0])

for bc in robin_bcs_T:
    alpha_R, beta_R = bc[1]
    a_T += kappa * inner(alpha_R * T, w) * ds(bc[0])
    L_T += kappa * inner(beta_R, w) * ds(bc[0])

a_T = fem.form(a_T)
L_T = fem.form(L_T)

A_T = fem.petsc.create_matrix(a_T)
b_T = fem.petsc.create_vector(L_T)

ksp_T = PETSc.KSP().create(msh.comm)
ksp_T.setOperators(A_T)
ksp_T.setType("preonly")
ksp_T.getPC().setType("lu")
ksp_T.getPC().setFactorSolverType("superlu_dist")

T_file = io.VTXWriter(msh.comm, "T.bp", [T_n._cpp_object])
T_file.write(t)

# Now we add the time stepping, convective, and buoyancy terms
# TODO Figure out correct way of "linearising"
# For buoyancy term, see
# https://en.wikipedia.org/wiki/Boussinesq_approximation_(buoyancy)
# where I've omitted the rho g h part (can think of this is
# lumping gravity in with pressure, see 2P4 notes) and taken
# T_0 to be 0
g = as_vector((0.0, -9.81))
rho_0 = fem.Constant(msh, PETSc.ScalarType(1.0))
eps = fem.Constant(msh, PETSc.ScalarType(10.0))  # Thermal expansion coeff

u_uw = lmbda("+") * u("+") + lmbda("-") * u("-")
a_00 += inner(u / delta_t, v) * dx - \
    inner(u, div(outer(v, u_n))) * dx + \
    inner((dot(u_n, n))("+") * u_uw, v("+")) * dS + \
    inner((dot(u_n, n))("-") * u_uw, v("-")) * dS + \
    inner(dot(u_n, n) * lmbda * u, v) * ds
a = fem.form([[a_00, a_01],
              [a_10, None]])

L_0 += inner(u_n / delta_t - eps * rho_0 * T_n * g, v) * dx

for bc in dirichlet_bcs:
    u_D = fem.Function(V)
    u_D.interpolate(bc[1])
    L_0 += - inner(dot(u_n, n) * (1 - lmbda) * u_D, v) * ds(bc[0])
L = fem.form([L_0,
              L_1])

# Time stepping loop

for n in range(num_time_steps):
    t += delta_t.value

    A.zeroEntries()
    fem.petsc.assemble_matrix_block(A, a, bcs=bcs)
    A.assemble()

    with b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector_block(b, L, a, bcs=bcs)

    # Compute solution
    ksp.solve(b, x)
    u_h.x.array[:offset] = x.array_r[:offset]
    u_h.x.scatter_forward()
    p_h.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]
    p_h.x.scatter_forward()
    if len(neumann_bcs) == 0:
        p_h.x.array[:] -= domain_average(msh, p_h)

    A_T.zeroEntries()
    fem.petsc.assemble_matrix(A_T, a_T)
    A_T.assemble()

    with b_T.localForm() as b_T_loc:
        b_T_loc.set(0)
    fem.petsc.assemble_vector(b_T, L_T)
    b_T.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    ksp_T.solve(b_T, T_n.vector)
    T_n.x.scatter_forward()

    u_vis.interpolate(u_h)

    # Write to file
    u_file.write(t)
    p_file.write(t)
    T_file.write(t)

    # Update u_n
    u_n.x.array[:] = u_h.x.array

u_file.close()
p_file.close()

# Compute errors
e_div_u = norm_L2(msh.comm, div(u_h))
# This scheme conserves mass exactly, so check this
assert np.isclose(e_div_u, 0.0)
