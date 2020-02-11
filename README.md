# DNA-Comet-Analysis
A complete deep learning based Comet analysis to assist Research scientists

## Motivation and Application
The DNA comet analysis is used for estimating the damage caused by any genotoxin to the cells. The Dyed DNAs of any cell that has been tested with any elememt shows damage if the element is a genotoxin. 

## Modules
1. Detecting Valid comets from comet assays
2. Classifying the validated comets based on the damage in the comet
3. Quantifying the damage of a comet based on damage parameters

### Detecting the Valid comets
The comets are detected by the state of the art object detection technique with Deep Learning where the datasets are collected and annotated with experts help.
The module gives only the valid comets for further evaluation on damaged comets

### Classifying the validated comets
Valid comets can be damaged or undamaged. This module gives only the damaged comets to further processing on quantifying the level of damage

### Quantification of damaged comets
The comets can express the destruction in varying levels based on the damage caused by the genotoxin under evaluation; which can be determined by the following damage parameters
* Head length
* Tail length
* Head intensity
* Tail intensity
* Head DNA percentage
* Tail DNA percentage
* Olive Tail moment

All the parameters are estimated by Image processing techniques embedded with Deep learning model.

