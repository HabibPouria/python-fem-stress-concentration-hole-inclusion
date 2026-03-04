#!/usr/bin/env python
# coding: utf-8

# In[12]:


from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri


# ============================================================
# plot new
# ============================================================

def plot_stress_field_ax(ax, nodes, elems, vm, center, R, levels=30):
    """
    Plot von Mises stress on a provided matplotlib axis.
    Returns the contour object (for shared colorbar).
    """
    x = nodes[:, 0]
    y = nodes[:, 1]

    # quads -> triangles
    triangles = []
    for e in elems:
        triangles.append([e[0], e[1], e[2]])
        triangles.append([e[0], e[2], e[3]])
    triangles = np.array(triangles, dtype=np.int32)

    triang = tri.Triangulation(x, y, triangles)

    cf = ax.tricontourf(triang, vm, levels=levels)
    ax.set_aspect("equal")

    # inclusion outline
    circle = plt.Circle(center, R, color="white", fill=False, linewidth=0.5)
    ax.add_patch(circle)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"R = {R:.3f}")

    return cf

# ============================================================
# stress gallery 
# ============================================================
import os
import numpy as np
import matplotlib.pyplot as plt

def make_stress_gallery(cfg, R_list, filename="stress_gallery.png"):
    os.makedirs(cfg.outdir, exist_ok=True)

    # run simulations + collect results
    cases = []
    vmin = +1e30
    vmax = -1e30

    for R in R_list:
        # run one case (uses your new signature)
        out = run_one_radius(R, cfg.E_incl, cfg.Tx, cfg)

        # IMPORTANT: we need nodes/elems/vm to plot.
        # If your run_one_radius currently does not return them,
        # recompute quickly here by re-running the internal pieces:
        # (best solution: add "return_fields=True" option later)

        # --- simplest approach: re-run pieces to get fields for plotting ---
        nodes, elems = build_mesh_Q4(cfg.Lx, cfg.Ly, cfg.nx, cfg.ny, cfg.center)
        region = tag_inclusion_elements(nodes, elems, R, cfg.center)
        K, f = assemble_system(nodes, elems, region, cfg)
        K, f, _ = apply_dirichlet(K, f, nodes, cfg)
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import spsolve
        U = spsolve(csr_matrix(K), f)
        vm = compute_nodal_von_mises(nodes, elems, region, U, cfg)

        vmin = min(vmin, float(np.min(vm)))
        vmax = max(vmax, float(np.max(vm)))
        cases.append((R, nodes, elems, vm))

    # create subplots
    n = len(cases)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 3), constrained_layout=True)
    if n == 1:
        axes = [axes]

    # shared levels for consistent color scale
    levels = np.linspace(vmin, vmax, 30)

    last_cf = None
    for ax, (R, nodes, elems, vm) in zip(axes, cases):
        last_cf = plot_stress_field_ax(ax, nodes, elems, vm, cfg.center, R, levels=levels)

    # one shared colorbar
    cbar = fig.colorbar(last_cf, ax=axes, shrink=0.9)
    cbar.set_label("von Mises stress (Pa)")

    outpath = os.path.join(cfg.outdir, filename)
    fig.savefig(outpath, dpi=200)
    plt.close(fig)

    print("Saved:", outpath)
# ============================================================
# plot
# ============================================================
def plot_stress_field(nodes, elems, vm, center, R, filename):

    x = nodes[:,0]
    y = nodes[:,1]

    # Convert quads to triangles for plotting
    triangles = []
    for e in elems:
        triangles.append([e[0], e[1], e[2]])
        triangles.append([e[0], e[2], e[3]])

    triangles = np.array(triangles)

    triang = tri.Triangulation(x, y, triangles)

    plt.figure(figsize=(6,3))
    plt.tricontourf(triang, vm, levels=30)
    plt.colorbar(label="von Mises stress")

    plt.gca().set_aspect("equal")

    # ----------------------------
    # draw inclusion circle
    # ----------------------------
    circle = plt.Circle(center, R, color="white", fill=False, linewidth=0.5)
    plt.gca().add_patch(circle)

    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# ============================================================
# Config
# ============================================================
class Cfg:
    # geometry
    Lx = 1.0
    Ly = 0.2
    center = (0.0, 0.0)

    # mesh
    nx = 240
    ny = 48

    # loading
    Tx = 100e6  # traction on right edge (Pa)

    # plane condition: "stress" or "strain"
    plane = "stress"  # change to "strain" for plane strain

    # materials
    E_matrix = 70e9
    nu_matrix = 0.33

    E_incl = 1e-6 * E_matrix
    nu_incl = 0.30

    # sweep
    R_list = [0.01, 0.02, 0.03, 0.04, 0.05]

    # outputs
    outdir = "results_param"
    export_vtu = False  # set True to write VTU (requires meshio)

cfg = Cfg()

# ============================================================
# Mesh generation (structured Q4)
# ============================================================
def build_mesh_Q4(Lx, Ly, nx, ny, center=(0.0,0.0)):
    x = np.linspace(-Lx/2, Lx/2, nx+1)
    y = np.linspace(-Ly/2, Ly/2, ny+1)
    X, Y = np.meshgrid(x, y, indexing="xy")
    nodes = np.c_[X.ravel(), Y.ravel()]  # (nnode,2)

    def nid(i, j): return j*(nx+1) + i
    elems = []
    for j in range(ny):
        for i in range(nx):
            elems.append([nid(i,j), nid(i+1,j), nid(i+1,j+1), nid(i,j+1)])
    elems = np.array(elems, dtype=np.int32)
    return nodes, elems

# ============================================================
# Inclusion tagging by element centroid
# ============================================================
def tag_inclusion_elements(nodes, elems, R, center=(0.0,0.0)):
    xy = nodes[elems]              # (nelem,4,2)
    c = xy.mean(axis=1)            # (nelem,2)
    dx = c[:,0] - center[0]
    dy = c[:,1] - center[1]
    rr = np.sqrt(dx*dx + dy*dy)
    region = np.zeros(len(elems), dtype=np.int32)
    region[rr <= R] = 1
    return region  # 0 matrix, 1 inclusion

# ============================================================
# Constitutive matrix
# ============================================================
def D_matrix(E, nu, plane="stress"):
    if plane == "stress":
        c = E / (1.0 - nu**2)
        D = c * np.array([[1.0, nu, 0.0],
                          [nu, 1.0, 0.0],
                          [0.0, 0.0, (1.0-nu)/2.0]])
    elif plane == "strain":
        c = E / ((1.0+nu)*(1.0-2.0*nu))
        D = c * np.array([[1.0-nu, nu, 0.0],
                          [nu, 1.0-nu, 0.0],
                          [0.0, 0.0, (1.0-2.0*nu)/2.0]])
    else:
        raise ValueError("plane must be 'stress' or 'strain'")
    return D

# ============================================================
# Q4 shape functions and derivatives in parent coords
# ============================================================
def q4_shape(xi, eta):
    N = 0.25*np.array([
        (1-xi)*(1-eta),
        (1+xi)*(1-eta),
        (1+xi)*(1+eta),
        (1-xi)*(1+eta)
    ])
    dN_dxi = 0.25*np.array([
        -(1-eta),
         (1-eta),
         (1+eta),
        -(1+eta)
    ])
    dN_deta = 0.25*np.array([
        -(1-xi),
        -(1+xi),
         (1+xi),
         (1-xi)
    ])
    return N, dN_dxi, dN_deta

# ============================================================
# Element stiffness (2x2 Gauss)
# ============================================================
def element_stiffness_Q4(xy, D):
    # 2x2 Gauss points
    gp = 1.0/np.sqrt(3.0)
    gauss = [(-gp,-gp), (gp,-gp), (gp,gp), (-gp,gp)]
    Ke = np.zeros((8,8), dtype=float)

    for (xi,eta) in gauss:
        _, dN_dxi, dN_deta = q4_shape(xi, eta)

        # Jacobian
        J = np.zeros((2,2), dtype=float)
        J[0,0] = np.dot(dN_dxi,  xy[:,0])
        J[0,1] = np.dot(dN_dxi,  xy[:,1])
        J[1,0] = np.dot(dN_deta, xy[:,0])
        J[1,1] = np.dot(dN_deta, xy[:,1])

        detJ = np.linalg.det(J)
        if detJ <= 1e-14:
            raise ValueError(f"Bad detJ={detJ}")

        invJ = np.linalg.inv(J)
        dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
        dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta

        # B matrix (3x8)
        B = np.zeros((3,8), dtype=float)
        for a in range(4):
            B[0, 2*a]   = dN_dx[a]
            B[1, 2*a+1] = dN_dy[a]
            B[2, 2*a]   = dN_dy[a]
            B[2, 2*a+1] = dN_dx[a]

        Ke += (B.T @ D @ B) * detJ  # weight=1 for 2x2

    return Ke

# ============================================================
# Assemble global K and traction force vector
# ============================================================
def assemble_system(nodes, elems, region, cfg):
    nnode = nodes.shape[0]
    ndof = 2*nnode
    K = lil_matrix((ndof, ndof), dtype=float)
    f = np.zeros(ndof, dtype=float)

    # Precompute D for both materials
    Dm = D_matrix(cfg.E_matrix, cfg.nu_matrix, cfg.plane)
    Di = D_matrix(cfg.E_incl, cfg.nu_incl, cfg.plane)

    # Assemble stiffness
    for e, conn in enumerate(elems):
        xy = nodes[conn]
        D = Di if region[e] == 1 else Dm
        Ke = element_stiffness_Q4(xy, D)

        dofs = np.zeros(8, dtype=np.int32)
        for a in range(4):
            dofs[2*a]   = 2*conn[a]
            dofs[2*a+1] = 2*conn[a] + 1

        # scatter add
        K[np.ix_(dofs, dofs)] += Ke

    # Traction on right edge x = +Lx/2
    xR = cfg.Lx/2
    tol = 1e-12
    right_nodes = np.where(np.abs(nodes[:,0] - xR) < tol)[0]

    # sort by y to integrate along the edge
    ys = nodes[right_nodes,1]
    right_nodes = right_nodes[np.argsort(ys)]

    # edge integration using segments between consecutive nodes (uniform traction)
    Tx = cfg.Tx
    for i in range(len(right_nodes)-1):
        n1 = right_nodes[i]
        n2 = right_nodes[i+1]
        y1 = nodes[n1,1]
        y2 = nodes[n2,1]
        Le = abs(y2 - y1)

        # consistent nodal loads for line traction (2-node line element)
        # f1 = Tx * Le/2, f2 = Tx * Le/2
        f[2*n1] += Tx * Le * 0.5
        f[2*n2] += Tx * Le * 0.5

    return K, f

# ============================================================
# Apply essential BC: left edge fixed
# ============================================================
def apply_dirichlet(K, f, nodes, cfg):
    xL = -cfg.Lx/2
    tol = 1e-12
    left_nodes = np.where(np.abs(nodes[:,0] - xL) < tol)[0]
    fixed_dofs = []
    for n in left_nodes:
        fixed_dofs.extend([2*n, 2*n+1])
    fixed_dofs = np.array(fixed_dofs, dtype=np.int32)

    # Modify K, f (simple elimination)
    for dof in fixed_dofs:
        K[dof,:] = 0.0
        K[:,dof] = 0.0
        K[dof,dof] = 1.0
        f[dof] = 0.0
    return K, f, fixed_dofs

# ============================================================
# Stress computation and nodal von Mises (averaging)
# ============================================================
def compute_nodal_von_mises(nodes, elems, region, U, cfg):
    nnode = len(nodes)
    vm_node = np.zeros(nnode, dtype=float)
    count = np.zeros(nnode, dtype=float)

    Dm = D_matrix(cfg.E_matrix, cfg.nu_matrix, cfg.plane)
    Di = D_matrix(cfg.E_incl, cfg.nu_incl, cfg.plane)

    # evaluate at element center (xi=0, eta=0) for simplicity
    xi = 0.0; eta = 0.0
    _, dN_dxi, dN_deta = q4_shape(xi, eta)

    for e, conn in enumerate(elems):
        xy = nodes[conn]
        D = Di if region[e] == 1 else Dm

        # Jacobian at center
        J = np.zeros((2,2), dtype=float)
        J[0,0] = np.dot(dN_dxi,  xy[:,0])
        J[0,1] = np.dot(dN_dxi,  xy[:,1])
        J[1,0] = np.dot(dN_deta, xy[:,0])
        J[1,1] = np.dot(dN_deta, xy[:,1])

        detJ = np.linalg.det(J)
        if detJ <= 1e-14:
            continue
        invJ = np.linalg.inv(J)
        dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
        dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta

        B = np.zeros((3,8), dtype=float)
        for a in range(4):
            B[0, 2*a]   = dN_dx[a]
            B[1, 2*a+1] = dN_dy[a]
            B[2, 2*a]   = dN_dy[a]
            B[2, 2*a+1] = dN_dx[a]

        ue = np.zeros(8, dtype=float)
        for a in range(4):
            ue[2*a]   = U[2*conn[a]]
            ue[2*a+1] = U[2*conn[a]+1]

        eps = B @ ue              # [exx, eyy, gxy]
        sig = D @ eps             # [sxx, syy, sxy]

        sxx, syy, sxy = sig
        vm = np.sqrt(sxx*sxx + syy*syy - sxx*syy + 3.0*sxy*sxy)

        # distribute to nodes (simple average)
        for a in conn:
            vm_node[a] += vm
            count[a] += 1.0

    vm_node = np.divide(vm_node, np.maximum(count, 1.0))
    return vm_node

def displacement_magnitude(U):
    Uxy = U.reshape(-1,2)
    return np.linalg.norm(Uxy, axis=1)

# ============================================================
# VTU export (optional)
# ============================================================
def export_vtu(nodes, elems, U, vm, umag, filename, outdir):
    import meshio
    os.makedirs(outdir, exist_ok=True)

    points = np.c_[nodes, np.zeros(len(nodes))]  # 3D points for VTU
    cells = [("quad", elems)]
    point_data = {
        "U": np.c_[U.reshape(-1,2), np.zeros(len(nodes))],
        "umag": umag,
        "von_Mises": vm,
    }
    mesh = meshio.Mesh(points=points, cells=cells, point_data=point_data)
    mesh.write(os.path.join(outdir, filename))

# ============================================================
# One run for a given radius
# ============================================================
def run_one_radius(R, E_incl, Tx, cfg):
    """
    Run one case for a given inclusion radius R, inclusion modulus E_incl, and traction Tx.

    Returns a dict of metrics suitable for DOE / pandas analytics.
    """
    import copy
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import spsolve

    # ------------------------------------------------------------
    # 0) Make a local cfg copy (safer for DOE loops)
    # ------------------------------------------------------------
    cfg_local = copy.copy(cfg)
    cfg_local.E_incl = float(E_incl)
    cfg_local.Tx = float(Tx)

    # ------------------------------------------------------------
    # 1) Build mesh + tag inclusion region
    # ------------------------------------------------------------
    nodes, elems = build_mesh_Q4(cfg_local.Lx, cfg_local.Ly, cfg_local.nx, cfg_local.ny, cfg_local.center)
    region = tag_inclusion_elements(nodes, elems, R, cfg_local.center)

    # ------------------------------------------------------------
    # 2) Assemble and apply Dirichlet BC
    # ------------------------------------------------------------
    K, f = assemble_system(nodes, elems, region, cfg_local)
    K, f, fixed = apply_dirichlet(K, f, nodes, cfg_local)

    # ------------------------------------------------------------
    # 3) Solve (sparse)
    # ------------------------------------------------------------
    K = csr_matrix(K)
    U = spsolve(K, f)

    # ------------------------------------------------------------
    # 4) Postprocess
    # ------------------------------------------------------------
    umag = displacement_magnitude(U)
    vm = compute_nodal_von_mises(nodes, elems, region, U, cfg_local)

    plot_stress_field(
        nodes,
        elems,
        vm,
        cfg_local.center,
        R,
        os.path.join(cfg_local.outdir, f"stress_R{int(R*1000):04d}.png")
    )

    max_vm = float(np.max(vm))

    # Interface-band metric (ring around r=R)
    band = 2.0 * (cfg_local.Lx / cfg_local.nx)  # ~2 element sizes
    mask = interface_band_mask(nodes, cfg_local.center, R, band)
    max_vm_interface = float(np.max(vm[mask])) if np.any(mask) else float("nan")

    # Nominal SCF definition
    scf = max_vm / cfg_local.Tx

    # ------------------------------------------------------------
    # 5) Optional VTU export (include R, E_incl, Tx in filename)
    # ------------------------------------------------------------
    if getattr(cfg_local, "export_vtu", False):
        tag = f"R{int(R*1000):04d}_E{int(cfg_local.E_incl/1e9):03d}GPa_Tx{int(cfg_local.Tx/1e6):04d}MPa"
        export_vtu(
            nodes, elems, U, vm, umag,
            filename=f"plate_inclusion_{tag}.vtu",
            outdir=cfg_local.outdir
        )

    # ------------------------------------------------------------
    # 6) Return metrics
    # ------------------------------------------------------------
    return {
        "R": float(R),
        "E_incl": float(cfg_local.E_incl),
        "Tx": float(cfg_local.Tx),
        "max_vm": max_vm,
        "max_vm_interface": max_vm_interface,
        "SCF": scf,
        "max_umag": float(np.max(umag)),
        "nnode": int(len(nodes)),
        "nelem": int(len(elems)),
    }

# ============================================================
# Parametric sweep Parallel execution (n_jobs=2) use 2 CPU cores only
# ============================================================
def run_parametric(cfg):
    os.makedirs(cfg.outdir, exist_ok=True)

    # -------------------------------------------------
    # Parallel execution of simulations
    # -------------------------------------------------
    rows = Parallel(n_jobs=-1)(
        delayed(run_one_radius)(R, cfg.E_incl, cfg.Tx, cfg)
        for R in cfg.R_list
    )

    # Print results (optional)
    for m in rows:
        print("DONE:", m)

    # -------------------------------------------------
    # DataFrame
    # -------------------------------------------------
    df = pd.DataFrame(rows).sort_values("R")
    df.to_csv(os.path.join(cfg.outdir, "summary.csv"), index=False)

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    plt.figure()
    plt.plot(df["R"], df["SCF"], marker="o")
    plt.xlabel("Inclusion radius R")
    plt.ylabel("SCF = max_vm / Tx")
    plt.grid(True)
    plt.savefig(os.path.join(cfg.outdir, "SCF_vs_R.png"), dpi=200)

    plt.figure()
    plt.plot(df["R"], df["max_vm"], marker="o")
    plt.xlabel("Inclusion radius R")
    plt.ylabel("max von Mises (Pa)")
    plt.grid(True)
    plt.savefig(os.path.join(cfg.outdir, "max_vm_vs_R.png"), dpi=200)

    return df
# ============================================================
# interface_band_mask
# ============================================================
def interface_band_mask(nodes, center, R, band_thickness):
    x = nodes[:,0] - center[0]
    y = nodes[:,1] - center[1]
    rr = np.sqrt(x*x + y*y)
    return (rr >= (R - band_thickness)) & (rr <= (R + band_thickness))
# ============================================================
# RUN
# ============================================================
cfg.export_vtu = True
df = run_parametric(cfg)
print(df.to_string(index=False))
print("Saved to:", os.path.abspath(cfg.outdir))

# ------------------------------------------------------------
# create gallery of stress fields
# ------------------------------------------------------------
make_stress_gallery(
    cfg,
    R_list=[0.01, 0.02, 0.03, 0.05],
    filename="stress_gallery.png"
)


# In[ ]:




