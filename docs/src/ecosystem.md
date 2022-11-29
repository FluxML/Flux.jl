# The Julia Ecosystem around Flux

One of the main strengths of Julia lies in an ecosystem of packages 
globally providing a rich and consistent user experience.

This is a non-exhaustive list of Julia packages, nicely complementing `Flux` in typical
machine learning and deep learning workflows. To add your project please send a [PR](https://github.com/FluxML/Flux.jl/pulls).
See also academic work [citing Flux](https://scholar.google.com/scholar?cites=9731162218836700005&hl=en) or [citing Zygote](https://scholar.google.com/scholar?cites=11943854577624257878&hl=en).

## Flux models

- Flux's [model-zoo](https://github.com/FluxML/model-zoo) contains examples from many domains.

### Computer vision

- [ObjectDetector.jl](https://github.com/r3tex/ObjectDetector.jl) provides ready-to-go image detection via YOLO.
- [Metalhead.jl](https://github.com/FluxML/Metalhead.jl) includes many state-of-the-art computer vision models which can easily be used for transfer learning.
- [UNet.jl](https://github.com/DhairyaLGandhi/UNet.jl) is a generic UNet implementation.

### Natural language processing

- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provides components for Transformer models for NLP, as well as providing several trained models out of the box.
- [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) provides several NLP algorithms that use Flux models under the hood.

### Reinforcement learning
 
- [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl) provides a generic, simple and fast implementation of Deepmind's AlphaZero algorithm.
- [ReinforcementLearning.jl](https://juliareinforcementlearning.org/) offers a collection of tools for doing reinforcement learning research in Julia.

### Graph learning

- [GraphNeuralNetworks.jl](https://github.com/CarloLucibello/GraphNeuralNetworks.jl) is a fresh, performant and flexible graph neural network library based on Flux.jl.
- [GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) is the first graph neural network library for julia. 
- [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) enables training infinite dimensional PDEs by learning a continuous function instead of using the finite element method.
- [SeaPearl.jl](https://github.com/corail-research/SeaPearl.jl) is a Constraint Programming solver that uses Reinforcement Learning based on graphs as input.

### Time series

- [FluxArchitectures.jl](https://github.com/sdobber/FluxArchitectures.jl) is a collection of advanced network architectures for time series forecasting.

---

## Tools closely associated with Flux

Utility tools you're unlikely to have met if you never used Flux!

### High-level training flows

- [FastAI.jl](https://github.com/FluxML/FastAI.jl) is a Julia port of Python's fast.ai library.
- [FluxTraining.jl](https://github.com/FluxML/FluxTraining.jl) is a package for using and writing powerful, extensible training loops for deep learning models. It supports callbacks for many common use cases like hyperparameter scheduling, metrics tracking and logging, checkpointing, early stopping, and more. It powers training in FastAI.jl

### Datasets

Commonly used machine learning datasets are provided by the following packages in the julia ecosystem:

- [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl) focuses on downloading, unpacking, and accessing benchmark datasets.
- [GraphMLDatasets.jl](https://github.com/yuehhua/GraphMLDatasets.jl): a library for machine learning datasets on graph.

### Plumbing
 
Tools to put data into the right order for creating a model.
 
- [Augmentor.jl](https://github.com/Evizero/Augmentor.jl) is a real-time library augmentation library for increasing the number of training images.
- [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl) aims to make it easy to build stochastic, label-preserving augmentation pipelines for vision use cases involving images, keypoints and segmentation masks.
- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) (replaces [MLDataUtils.jl](https://github.com/JuliaML/MLDataUtils.jl) and [MLLabelUtils.jl](https://github.com/JuliaML/MLLabelUtils.jl)) is a library for processing Machine Learning datasets.

### Parameters

- [ParameterSchedulers.jl](https://github.com/darsnack/ParameterSchedulers.jl) standard scheduling policies for machine learning.

---

## Differentiable programming

Packages based on differentiable programming but not necessarily related to Machine Learning. 

- The [SciML](https://sciml.ai/) ecosystem uses Flux and Zygote to mix neural nets with differential equations, to get the best of black box and mechanistic modelling.
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) provides tools for creating Neural Differential Equations.
- [Flux3D.jl](https://github.com/FluxML/Flux3D.jl) shows off machine learning on 3D data.
- [RayTracer.jl](https://github.com/avik-pal/RayTracer.jl) combines ML with computer vision via a differentiable renderer.
- [Duckietown.jl](https://github.com/tejank10/Duckietown.jl) Differentiable Duckietown simulator.
- The [Yao.jl](https://github.com/QuantumBFS/Yao.jl) project uses Flux and Zygote for Quantum Differentiable Programming.
- [AtomicGraphNets.jl](https://github.com/Chemellia/AtomicGraphNets.jl) enables learning graph based models on atomic systems used in chemistry.
- [DiffImages.jl](https://github.com/FluxML/DiffImages.jl) differentiable computer vision modeling in Julia with the Images.jl ecosystem.

### Probabilistic programming
 
- [Turing.jl](https://github.com/TuringLang/Turing.jl) extends Flux's differentiable programming capabilities to probabilistic programming.
- [Omega.jl](https://github.com/zenna/Omega.jl) is a research project aimed at causal, higher-order probabilistic programming.
- [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) provides flexible Gaussian processes.

### Statistics

- [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) provides single-pass algorithms for statistics.

---

## Useful miscellaneous packages

Some useful and random packages!

- [AdversarialPrediction.jl](https://github.com/rizalzaf/AdversarialPrediction.jl) provides a way to easily optimize generic performance metrics in supervised learning settings using the [Adversarial Prediction](https://arxiv.org/abs/1812.07526) framework.
- [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) helps to prototype flexible multi-instance learning models.
- [MLMetrics.jl](https://github.com/JuliaML/MLMetrics.jl) is a utility for scoring models in data science and machine learning.
- [Torch.jl](https://github.com/FluxML/Torch.jl) exposes torch in Julia.
- [ValueHistories.jl](https://github.com/JuliaML/ValueHistories.jl) is a utility for efficient tracking of optimization histories, training curves or other information of arbitrary types and at arbitrarily spaced sampling times.
- [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl/) Building blocks for invertible neural networks in the Julia programming language.
- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl) progress meters for long-running computations.
- [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) easy peasy logging to [tensorboard](https://www.tensorflow.org/tensorboard) in Julia
- [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) is a package for parsing command-line arguments to Julia programs.
- [Parameters.jl](https://github.com/mauro3/Parameters.jl) types with default field values, keyword constructors and (un-)pack macros.
- [BSON.jl](https://github.com/JuliaIO/BSON.jl) is a package for working with the Binary JSON serialisation format.
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) in-memory tabular data in Julia.
- [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) is a scientific project assistant software.

This tight integration among Julia packages is shown in some of the examples in the [model-zoo](https://github.com/FluxML/model-zoo) repository.

---

## Alternatives to Flux

Julia has several other libraries for making neural networks. 

* [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) is focused on making small, simple, CPU-based, neural networks fast. Uses [LoopVectorization.jl](https://github.com/JuliaSIMD/LoopVectorization.jl). (Was `FastChain` in DiffEqFlux.jl) 

* [Knet.jl](https://github.com/denizyuret/Knet.jl) is a neural network library built around [AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl), with beautiful documentation.

* [Lux.jl](https://github.com/avik-pal/Lux.jl) (earlier ExplicitFluxLayers.jl) shares much of the design, use-case, and NNlib.jl / Optimisers.jl back-end of Flux. But instead of encapsulating all parameters within the model structure, it separates this into 3 components: a model, a tree of parameters, and a tree of model states.

