# PioGrowth
Repository for ShinyLive based app for analysis of Optical Density mesurements 
produced mainly with a [PioReactor](https://pioreactor.com/)

It was created on the basis of three individual 

- [milnus/pioreactor_turbidostat_shiny](https://github.com/milnus/pioreactor_turbidostat_shiny)
- [milnus/Batch_analysis](https://github.com/milnus/Batch_analysis)
- [milnus/OD_calibration_bioreactor](https://github.com/milnus/OD_calibration_bioreactor)

The code translated to Python was based on the original R code from the Shiny apps.

## Install and start app

In a new environment you can install the app using the app's 
[`requirements.txt`](app/requirements.txt)

```bash
pip install -r app/requirements.txt
```

Start the app from the root of the repository (as it's done on Streamlit Cloud):

```bash
streamlit run app/main.py
```

## History

The joint app combining three Shiny apps for PioReactor tools was started as well as an
shiny app, in the original repository of this fork 
(see: [milnus/PioGrowth](https://github.com/milnus/PioGrowth)). 

