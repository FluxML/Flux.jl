# Transferring data across devices

Flux relies on the [MLDataDevices.jl](https://github.com/LuxDL/MLDataDevices.jl/blob/main/src/public.jl) package to manage devices and transfer data across them. You don't have to explicitly use the package, as Flux re-exports the necessary functions and types.

```@docs
MLDataDevices.cpu_device
MLDataDevices.default_device_rng
MLDataDevices.functional
MLDataDevices.get_device
MLDataDevices.gpu_device
MLDataDevices.gpu_backend!
MLDataDevices.get_device_type
MLDataDevices.isleaf
MLDataDevices.loaded
MLDataDevices.reset_gpu_device!
MLDataDevices.set_device!
MLDataDevices.supported_gpu_backends
MLDataDevices.DeviceIterator
```
