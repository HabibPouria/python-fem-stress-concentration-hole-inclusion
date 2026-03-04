
---

## Motivation

This project demonstrates **computational mechanics workflows in Python**, including:

- finite element implementation
- sparse linear solvers
- automated parametric studies
- visualization and post-processing

---

## Author

Habib Pouriayevali  
Computational Mechanics / FEM / Materials Modeling
# Python FEM: Plate with Circular Hole and Elastic Inclusion
### Parametric Stress Concentration Study

This repository contains a lightweight **2D Finite Element Method (FEM) solver implemented in Python** to investigate **stress concentration in plates containing a circular hole or elastic inclusion** under uniaxial tension.

The solver is implemented **from scratch** using **bilinear Q4 finite elements with 2×2 Gauss integration** and supports automated **parametric studies of defect size**.

The project demonstrates computational mechanics workflows using **scientific Python**, including FEM implementation, sparse solvers, automated simulation studies, and visualization.

---

# Features

- 2D **plane stress FEM solver**
- **Bilinear Q4 finite elements**
- **2×2 Gauss integration**
- Sparse global stiffness matrix assembly
- Linear system solution using **SciPy sparse solvers**
- **von Mises stress computation**
- Automated **parametric studies (radius sweep)**
- Stress contour visualization using **Matplotlib**
- Export results to **VTU format** for visualization in **ParaView**

---

# Material Model

The plate consists of a **matrix material** containing a circular **elastic inclusion** located at the center.

Material properties used in the simulations:

```python
# Matrix material
E_matrix = 70e9
nu_matrix = 0.33

# Inclusion material
E_incl = 200e9
nu_incl = 0.30

# Approximate hole behavior (near-zero stiffness)
E_incl = 1e-6 * E_matrix


## Example Result

Stress distribution around circular inclusion for different radii.

![Stress Gallery](results/stress_gallery.png)

---

## Output Files

The simulation generates:
