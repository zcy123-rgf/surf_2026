# Final validation metric definitions

This release has no official lane-boundary ground truth for the reported run.
The metrics below are therefore diagnostics and consistency measurements, not
lane-detection accuracy.

## Pose curvature

`analyze_pose_curvature.py` computes the planar pose curvature

```text
kappa = abs(x' z'' - z' x'') / (x'^2 + z'^2)^(3/2)
```

using the KITTI camera trajectory in X/Z. It writes a per-frame CSV, a summary,
and a plot. This is the road/camera trajectory shape signal used to identify
straight, transition, and curved portions. It is not derived from CLRNet lane
pixels.

The polynomial fitter uses this persistent curvature rule for its diagnostic
window label: `kappa <= 0.004 1/m` is straight, `kappa >= 0.005 1/m` is curve,
and the interval between them is transition. A label must persist for three
frames and occupy at least 60% of a 15-frame window. All windows still use the
same quadratic parametric polynomial; the label does not select a different
model.

## Reported metrics

- `Mean`, `RMSE`, `Median`, `P90`, `P95`, `Max`: residuals between a fitted
  window and held-out CLRNet/IPM observations. They measure pipeline
  self-consistency, not distance to a real lane boundary.
- `Chamfer`: symmetric nearest-neighbour distance between aggregate observed
  CLRNet/IPM points and fitted window samples. In this release it is explicitly
  named `observed_fit_chamfer_m` because it uses observed fit points, not ground
  truth.
- `Coverage`: observed left/right frame fraction in each window. It measures
  availability and does not prove semantic lane identity.
- `Continuity`: adjacent-window overlap gap and tangent checks. A failed check
  starts a new output segment instead of connecting unsupported geometry.
- `Lane width`: distance between left/right blended curves where both are
  supported. It is a plausibility diagnostic, not a label accuracy score.

## Sequence 01 range policy

The validated core remains frames `857-961`. A candidate extension such as
`851-1005` should be accepted only after the scan confirms both-side coverage
and the curvature plot shows the intended straight-to-curve-to-straight span.
The extension is not allowed to fill missing turning-frame lanes by invention.

## Excluded claims

Do not call held-out residuals "official accuracy", call the SemanticKITTI road
class a painted lane boundary, or claim validation of all KITTI sequences from
one successful Sequence 01 interval.
