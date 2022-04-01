# The Julia Ecosystem

One of the main strengths of Julia lies in an ecosystem of packages 
globally providing a rich and consistent user experience.

This is a non-exhaustive list of Julia packages, nicely complementing `Flux` in typical
machine learning and deep learning workflows. To add your project please send a [PR](https://github.com/FluxML/Flux.jl/pulls).
See also academic work citing Flux or Zygote.

## Advanced models

- [FluxArchitectures.jl](https://github.com/sdobber/FluxArchitectures.jl) is a collection of slightly more advanced network architectures.

## Computer vision

- [ObjectDetector.jl](https://github.com/r3tex/ObjectDetector.jl) provides ready-to-go image analysis via YOLO.
- [Metalhead.jl](https://github.com/FluxML/Metalhead.jl) includes many state-of-the-art computer vision models which can easily be used for transfer learning.
- [UNet.jl](https://github.com/DhairyaLGandhi/UNet.jl) is a generic UNet implementation.

## Datasets

- [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl) focuses on downloading, unpacking, and accessing benchmark datasets.

## Differentiable programming

- The [SciML](https://sciml.ai/) ecosystem uses Flux and Zygote to mix neural nets with differential equations, to get the best of black box and mechanistic modelling.
- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) provides tools for creating Neural Differential Equations.
- [Flux3D.jl](https://github.com/FluxML/Flux3D.jl) shows off machine learning on 3D data.
- [RayTracer.jl](https://github.com/avik-pal/RayTracer.jl) combines ML with computer vision via a differentiable renderer.
- [Duckietown.jl](https://github.com/tejank10/Duckietown.jl) Differentiable Duckietown simulator.
- The [Yao.jl](https://github.com/QuantumBFS/Yao.jl) project uses Flux and Zygote for Quantum Differentiable Programming.
- [AtomicGraphNets.jl](https://github.com/Chemellia/AtomicGraphNets.jl) enables learning graph based models on atomic systems used in chemistry.
- [DiffImages.jl](https://github.com/SomTambe/DiffImages.jl) differentiable computer vision modeling in Julia with the Images.jl ecosystem.

## Graph learning

- [GeometricFlux.jl](https://github.com/FluxML/GeometricFlux.jl) makes it easy to build fast neural networks over graphs.
- [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) enables training infinite dimensional PDEs by learning a continuous function instead of using the finite element method.
- [SeaPearl.jl](https://github.com/corail-research/SeaPearl.jl) is a Constraint Programming solver that uses Reinforcement Learning based on graphs as input.

## Natural language processing

- [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) provides components for Transformer models for NLP, as well as providing several trained models out of the box.
- [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) provides several NLP algorithms that use Flux models under the hood.

## Pipeline extensions
 
- [DLPipelines.jl](https://github.com/lorenzoh/DLPipelines.jl) is an interface for defining deep learning data pipelines.

## Plumbing
 
Tools to put data into the right order for creating a model.
 
- [Augmentor.jl](https://github.com/Evizero/Augmentor.jl) is a real-time library augmentation library for increasing the number of training images.
- [DataAugmentation.jl](https://github.com/lorenzoh/DataAugmentation.jl) aims to make it easy to build stochastic label-preserving augmentation pipelines for your datasets.
- [MLDataUtils.jl](https://github.com/JuliaML/MLDataUtils.jl) is a utility for generating, loading, partitioning, and processing Machine Learning datasets.
- [MLLabelUtils.j](https://github.com/JuliaML/MLLabelUtils.jl) is a utility for working with classification targets. It provides the necessary functionality for interpreting class-predictions, as well as converting classification targets from one encoding to another. 

## Probabilistic programming
 
- [Turing.jl](https://github.com/TuringLang/Turing.jl) extends Flux's differentiable programming capabilities to probabilistic programming.
- [Omega.jl](https://github.com/zenna/Omega.jl) is a research project aimed at causal, higher-order probabilistic programming.
- [Stheno.jl](https://github.com/willtebbutt/Stheno.jl) provides flexible Gaussian processes.

## Reinforcement learning
 
- [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl) provides a generic, simple and fast implementation of Deepmind's AlphaZero algorithm.
- [ReinforcementLearning.jl](https://juliareinforcementlearning.org/) offers a collection of tools for doing reinforcement learning research in Julia.

## Parameters

- [Parameters.jl](https://github.com/mauro3/Parameters.jl) types with default field values, keyword constructors and (un-)pack macros.
- [ParameterSchedulers.jl](https://github.com/darsnack/ParameterSchedulers.jl) standard scheduling policies for machine learning.

## Statistics

- [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) provides single-pass algorithms for statistics.

## Miscellaneous

- [AdversarialPrediction.jl](https://github.com/rizalzaf/AdversarialPrediction.jl) provides a way to easily optimize generic performance metrics in supervised learning settings using the [Adversarial Prediction](https://arxiv.org/abs/1812.07526) framework.
- [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) helps to prototype flexible multi-instance learning models.
- [MLMetrics.jl](https://github.com/JuliaML/MLMetrics.jl) is a utility for scoring models in data science and machine learning.
- [Torch.jl](https://github.com/FluxML/Torch.jl) exposes torch in Julia.
- [ValueHistories.jl](https://github.com/JuliaML/ValueHistories.jl) is a utility for efficient tracking of optimization histories, training curves or other information of arbitrary types and at arbitrarily spaced sampling times.
- [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl/) Building blocks for invertible neural networks in the Julia programming language.
- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl) progress meters for long-running computations.
- [TensorBoardLogger.jl](https://github.com/PhilipVinc/TensorBoardLogger.jl) easy peasy logging to [tensorboard](https://www.tensorflow.org/tensorboard) in Julia
- [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) is a package for parsing command-line arguments to Julia programs.
- [BSON.jl](https://github.com/JuliaIO/BSON.jl) is a package for working with the Binary JSON serialisation format.
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) in-memory tabular data in Julia.
- [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) is a scientific project assistant software.

This tight integration among Julia packages is shown in some of the examples in the [model-zoo](https://github.com/FluxML/model-zoo) repository.
