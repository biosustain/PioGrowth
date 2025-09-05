# Curve fitting methods

## Querve approach

- based on approach in Querve paper, which based it on the approach layed out in
  grofit R-package: https://www.jstatsoft.org/article/view/v033i07
- cubic smoothing spline from R was used:
  [docs](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/smooth.spline)

- the curves were fitted on the log-transformed data of the plots, shifted by the minimal
  value

## This app

In this app we use cubic splines using scipy's
[`make_splrep` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html#scipy.interpolate.make_splrep)
on the original, interpolated data.
