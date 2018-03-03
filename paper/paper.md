---
title: 'Flux: Elegant machine learning with Julia'
tags:
  - deep learning
  - machine learning
  - natural language processing
  - computer vision
  - reinforcement learning
  - robotics
  - automatic differentiation
  - compiler
authors:
  - name: Mike Innes
    orcid: 0000-0003-0788-0242
    affiliation: 1
affiliations:
  - name: Julia Computing
    index: 1
date: 16 February 2018
bibliography: paper.bib
---

# Summary

Flux is library for machine learning (ML), written using the numerical computing language Julia [@Julia]. The package allows models to be written using Julia's simple mathematical syntax, and applies automatic differentiation (AD) to seamlessly calculate derivatives and train the model. Meanwhile, it makes heavy use of Julia's language and compiler features to carry out code analysis and make optimisations. For example, Julia's GPU compilation support [@besard:2017] can be used to JIT-compile custom GPU kernels for model layers [@CuArrays].

The machine learning community has traditionally been divided between "static" and "dynamic" frameworks that are easy to optimise and easy to use, respectively [@MLPL]. Flux blurs the line between these two approaches, combining a highly intuitive programming model with the compiler techniques needed by ML. As a result of this approach, it already supports several features not available in any other dynamic framework, such as kernel fusion [@Fusion], memory usage optimisations, importing of models via ONNX, and deployment of models to JavaScript for running in the browser.

Flux has been used heavily for natural language processing, but can also support state-of-the-art research models in areas like computer vision, reinforcement learning and robotics.

# References
