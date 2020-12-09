# Human Protein Atlas Image Classification


## Introduction

Proteins are complex molecules made of thousands of amino acids, being responsible for many important functions in human body such as execution and regulation of issues and organs. Identifying the pattern and organelle localization of proteins would provide more insights about human living cells and accelerate the diagnostic of diseases. Moreover, understanding the complexity of cell structure plays a key role in developing medicine and treatment. Therefore, the classification of proteins has become a field of interest for many scientists and biomedical researchers.

## Objectives

The primary objective is to classify mixed patterns of proteins from microscope images of different human cell types. Since the complexity and highly various morphology of human cells, it is difficult to identify the structure and the number of protein patterns in organelles. Another challenge is the imbalanced distribution of proteins. While coarse grained cellular classes such as nucleus, plasma membrane and cytosol are very popular, small components such as endosomes, lysosomes and microtubule ends are rarely observed in cell structures. Consequently, the classification would project to the majority classes. The project will explore different techniques and methods to handle these two difficulties.

## Dataset

The dataset for the project comes from the [Human Protein Atlas Kaggle competition](https://www.kaggle.com/c/human-protein-atlas-image-classification/data) which is originally provided by the Human Protein Atlas – a Sweden-based program researching proteins in cells, tissues and organs: 

The “train.csv” data contains 31072 samples of 27 different living cells. Each sample is represented by four images, the protein of interest which is the main filter and three cellular landmarks as references: nucleus, microtubules and endoplasmic reticulum. The target is the protein pattern including 28 categories labeled from 0 to 27. Each sample may have one or more labels.

## About this repository

- "firstname-lastname-individual-project" folders: individual work on the project with code and a detailed individual report.
- Code: the codes for best model.
- Final-Group-Project-Report: the combination and summary of our individual reports. Please check our individual reports for clear and detailed about our work.

