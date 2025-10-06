# Curve fitting methods

## QurvE approach

- based on approach in QurvE paper, which based it on the approach laid out in
  grofit R-package. See their 
  [README](https://github.com/NicWir/QurvE?tab=readme-ov-file#growth-profiling-methods)
  and 
  [Kahm et.al. 2014](https://www.jstatsoft.org/article/view/v033i07)
- cubic smoothing spline from R was used:
  [docs](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/smooth.spline)

- the curves were fitted on the log-transformed data of the plots, shifted by the minimal
  value

## This app

In this app we use cubic splines using scipy's
[`make_splrep` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.make_splrep.html#scipy.interpolate.make_splrep)
on the original, interpolated data.

- the rolling median was used to pre-smooth the data before fitting the splines
- the smoothing factor is based on the total number of measurements in the app,
  the minimum is used from the suggested interval by scipy
  - maybe the rolling median should only be used for missing values in the filtered
    data?
- per default we do not apply log transformation.
