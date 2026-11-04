# Cross-section warping in atomic rods

Numerical approximation of the **relaxed quadratic form** in the model for **ultrathin 
elastic rods** from [1] (see equation (3.3)), which is inspired by nanowires.

The implementation in its current version is only illustrative and not optimized 
for best performance on medium or large cross sections of the rod.

The **cell energy** function originates from a mass-spring model with 
nearest-neighbour and next-to-nearest-neighbour interactions on a cubic lattice 
(see [1, Example 2.1]).

To obtain the value of the relaxed quadratic form at a given 
skew-symmetric matrix $A$, minimization over variables $\alpha$ and $g$ is performed. 
$\alpha$ is related to **warping** of the rod's cross section in torsion and **lateral contraction** due to bending, whereas $g$ is a **stretch** vector, uniform
in the cross-sectional plane.

The minimizer is visualized using 3D plotting in Matplotlib.

For higher physical relevance, surface terms would also have to be added 
to the energy functional $J$, depending on the chosen cross section shape.

![Example plot.](plot_example.jpeg "Example plot")

<p align="center"><i>Example plot.</i></p>

---
## References

[1] Bernd Schmidt and Jiří Zeman. A bending-torsion theory for thin
and ultrathin rods as a Γ-limit of atomistic models. *Multiscale Modeling & Simulation*, 21(4):1717--1745, 2023. [arXiv:2208.04199](https://arxiv.org/abs/2208.04199).
