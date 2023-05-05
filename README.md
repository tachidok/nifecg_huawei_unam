# Fetal and mother heart rate identification on NI-FECG signals

## Main objective
Develop a deep learning based strategy for the fetal and mother heart rate identification on sintetic NI-FECG signals.

## Main expected outcomes
* A deep neural network design for the fetal and mother heart rate identification on sintetic NI-FECG signals.
* A deep neural network implementation for the fetal and mother hearh rate identification on sintetic NI-FECG signals.
* A set of sintetic NI-FECG signals generated by the PhysioNet simulator
* Publish the research results

## Install software tools
Generate the conda environment

```
conda env create --file nifecg.yml
```

Install all python dependecies required for the project
```
pip install -r requirements.txt
```

## Update the conda and pip requirements files
If you required to install additional tools please update the
nifecg.yml and requiremets.txt files.

```
rm nifecg.yml
conda env export --name nifecg > nifecg.yml

rm requirements.txt
pip freeze > requirements.txt
```

