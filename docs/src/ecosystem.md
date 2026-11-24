# The Julia Ecosystem around Flux

These Julia packages provide models and tools for working with Flux. To suggest a package, [open a pull request](https://github.com/FluxML/Flux.jl/pulls).
See also academic work [citing Flux](https://scholar.google.com/scholar?cites=9731162218836700005&hl=en) or [citing Zygote](https://scholar.google.com/scholar?cites=11943854577624257878&hl=en).

## Flux models

- [Metalhead.jl](@ref man-pretrained-models) provides pretrained vision, language and image generation models for Flux.

### Computer vision

- [ObjectDetector.jl](https://github.com/r3tex/ObjectDetector.jl) provides ready-to-go image detection via YOLO.

### Natural language processing

- [TextAnalysis.jl](https://github.com/JuliaText/TextAnalysis.jl) provides several NLP algorithms that use Flux models under the hood.

### Reinforcement learning

- [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl) provides a generic, simple and fast implementation of DeepMind's AlphaZero algorithm.

### Graph learning

- [GraphNeuralNetworks.jl](https://github.com/JuliaGraphs/GraphNeuralNetworks.jl) provides graph neural network layers for Flux.

### Robust networks

- [RobustNeuralNetworks.jl](https://github.com/acfr/RobustNeuralNetworks.jl) includes classes of neural networks that are constructed to naturally satisfy robustness constraints.

### Recurrent neural networks

- [RecurrentLayers.jl](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl) offers a collection of recurrent neural network layers.

## Tools closely associated with Flux

### High-level training flows

- [Ignite.jl](https://github.com/jondeuce/Ignite.jl) is a Julia port of the Python library `ignite` for simplifying neural network training and validation loops, using events and handlers.
- [Tsunami.jl](https://github.com/CarloLucibello/Tsunami.jl) adds high-level ways to control training, parameter schedules and logging, inspired by PyTorch Lightning.

### Datasets

- [MLDatasets.jl](https://github.com/JuliaML/MLDatasets.jl) provides access to benchmark datasets.

### [Data processing](@id Plumbing)

- [DataAugmentation.jl](https://github.com/FluxML/DataAugmentation.jl) provides augmentation pipelines for images, keypoints and segmentation masks.
- [MLUtils.jl](https://github.com/JuliaML/MLUtils.jl) provides tools for processing, splitting and batching datasets.

### Parameters

- [ParameterSchedulers.jl](https://github.com/darsnack/ParameterSchedulers.jl) provides schedules for learning rates and other training parameters.

## Differentiable programming

- [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) provides layers for neural differential equations. It can use Flux models through Lux's `FromFluxAdaptor`.
- [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) provides neural operators for learning maps between functions, such as PDE solution operators.
- [Yao.jl](https://github.com/QuantumBFS/Yao.jl) provides tools for simulating and differentiating quantum circuits.

### Probabilistic programming

- [Turing.jl](https://github.com/TuringLang/Turing.jl) supports probabilistic programming and Bayesian inference in Julia.

### Simulation-based inference

- [NeuralEstimators.jl](https://github.com/msainsburydale/NeuralEstimators.jl) enables amortized simulation-based inference using neural networks built with Flux.

### Statistics

- [OnlineStats.jl](https://github.com/joshday/OnlineStats.jl) provides single-pass algorithms for statistics.

## Useful miscellaneous packages

- [Mill.jl](https://github.com/CTUAvastLab/Mill.jl) helps to prototype flexible multi-instance learning models.
- [ValueHistories.jl](https://github.com/JuliaML/ValueHistories.jl) provides data structures for tracking learning curves and other optimisation information.
- [InvertibleNetworks.jl](https://github.com/slimgroup/InvertibleNetworks.jl/) provides building blocks for invertible neural networks.
- [ProgressMeter.jl](https://github.com/timholy/ProgressMeter.jl) provides progress meters for long-running computations.
- [TensorBoardLogger.jl](https://github.com/JuliaLogging/TensorBoardLogger.jl) logs training metrics to TensorBoard.
- [ArgParse.jl](https://github.com/carlobaldassi/ArgParse.jl) provides tools for parsing command-line arguments.
- [Parameters.jl](https://github.com/mauro3/Parameters.jl) provides types with default field values, keyword constructors and unpacking macros.
- [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) provides in-memory tabular data structures.
- [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl) provides tools for managing scientific projects.

## Alternatives to Flux

- [SimpleChains.jl](https://github.com/PumasAI/SimpleChains.jl) focuses on fast execution for small neural networks on the CPU.
- [Lux.jl](https://github.com/LuxDL/Lux.jl) represents a model, its parameters and its state separately.
