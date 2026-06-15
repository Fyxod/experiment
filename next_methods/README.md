# Next Methods

This folder contains the geometry-oriented methods explored after the pixel-space baseline.

The important direction was to avoid arbitrary pixel noise and instead optimize differentiable coordinate transforms:

- TPS control-point displacement
- fixed-topology Delaunay/piecewise displacement
- low-resolution free-form displacement grids
- DCT/spectral coordinate fields
- rolling shutter coefficients
- affine and radial/lens coordinate warps
- face-local masks and method gates

The results were weaker than the pixel-space adversarial perturbation baseline under strict visual constraints. The best geometry examples show identity/style drift rather than full edit prevention.

