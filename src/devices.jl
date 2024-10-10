# TODO get docstring from MLDataDevices.get_device
get_device(x) = MLDataDevices.get_device(x)

function (device::MLDataDevices.AbstractDevice)(d::MLUtils.DataLoader)
    MLUtils.DataLoader(MLUtils.mapobs(device, d.data),
        d.batchsize,
        d.buffer,
        d.partial,
        d.shuffle,
        d.parallel,
        d.collate,
        d.rng,
    )
end
