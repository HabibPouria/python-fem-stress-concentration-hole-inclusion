
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


# Python FEM: Plate with Circular Hole and Inclusion

This project implements a simple **2D Finite Element Method (FEM) solver in Python**
to study stress concentration in plates containing a **circular hole or elastic inclusion**.

The solver uses **bilinear Q4 elements with 2×2 Gauss integration** and performs
a **parametric study of inclusion radius**.

---

## Features

- 2D FEM solver (plane stress)
- Bilinear **Q4 elements**
- **2×2 Gauss integration**
- Sparse matrix solver using **SciPy**
- von Mises stress computation
- Automatic **parametric studies**
- Stress contour visualization
- Export to **ParaView (VTU)**

---

## Technologies

Python  
NumPy  
SciPy  
Matplotlib  
pandas  
meshio  
ParaView  

---

## Example Result

Stress distribution around circular inclusion for different radii.

![Stress Gallery](results/stress_gallery.png)

---

## Output Files

The simulation generates:
