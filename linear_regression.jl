### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ e6c4d3d0-212c-11ed-04dc-91f7d92441a1
using Flux, Plots

# ╔═╡ 54b7fffb-2d7a-469a-b66e-b6b7aa466387
using Statistics, DataFrames

# ╔═╡ d9ab4748-4b6a-473e-b728-d1936612e1c1
using MLDatasets: BostonHousing

# ╔═╡ 2e286889-de4f-4147-8255-2522a509495f
md"# Linear Regression

The following page contains a step-by-step walkthrough of the linear regression algorithm in `Julia` using `Flux`! We will start by creating a simple linear regression model for dummy data and then move on to a real dataset. The first part would involve writing some parts of the model on our own, which will later be replaced by `Flux`.
"

# ╔═╡ c727549c-dcab-48b0-b000-bcd63f35603b
md"## A simple linear regression model
Let us start by building a simple linear regression model. This model would be trained on the data points of the form `(x₁, y₁), (x₂, y₂), ... , (xₙ, yₙ)`. In the real world, these `x`s can have multiple features, and the `y`s denote a label. In our example, each `x` has a single feature; hence, our data would have `n` data points, each point mapping a single feature to a single label.

Importing the required `Julia` packages -
"

# ╔═╡ 6a85e00d-c15d-42bf-9483-eede897fbd1d
md"### Generating a dataset
The data usually comes from the real world, which we will be exploring in the last part of this guide, but we don't want to jump straight to the relatively harder part. Here we will generate the `x`s of our data points and map them to the respective `y`s using a simple function. Remember, here each `x` is equivalent to a feature, and each `y` is the corresponding label. Combining all the `x`s and `y`s would create the complete dataset.
"

# ╔═╡ 5509254a-a076-421d-9646-d226ef62c49f
x = hcat(collect(Float32, -3:0.1:3)...)

# ╔═╡ b5ffe55a-3405-4ef8-bf9d-78890fadf9bd
md"The `hcat` call generates a `Matrix` with numbers ranging from `-3.0` to `3.0` with a gap of `0.1` between them. Each column of this matrix holds a single `x`, a total of 61 `x`s. The next step would be to generate the corresponding labels or the `y`s."

# ╔═╡ 572bd321-aa20-4090-b819-d81ccf1e244b
f(x) = @. 3x + 2;

# ╔═╡ 1c1b0764-f926-489d-833e-835fdcaad7db
y = f(x)

# ╔═╡ b2c6b0f6-e388-48a3-a4f0-c59554187f07
md"The function `f` maps each `x` to a `y`, and as `x` is a `Matrix`, the expression broadcasts the scalar values using `@.` macro. Our data points are ready, but they are too perfect. In a real-world scenario, we will not have an `f` function to generate `y` values, but instead, the labels would be manually added."

# ╔═╡ 3b0b47e0-e38e-42f0-9d71-6e9a01f8a72f
x_noise = x .* reshape(rand(Float32, 61), (1, 61))

# ╔═╡ a7fd511c-be36-4c65-b3c2-1b362ec4268b
md"Visualizing the final data -"

# ╔═╡ 059bfedb-aac5-42cf-baa8-a10dab14f032
plot(vec(x_noise), vec(y), lw = 3, seriestype = :scatter, label = "", title = "Generated data", xlabel = "x", ylabel= "y")

# ╔═╡ 357853d5-fd49-46fd-a604-c766f33814e7
md"The data looks random enough now! The `x` and `y` values are still somewhat correlated; hence, the linear regression algorithm should work fine on our dataset.

We can now proceed ahead and build a model for our dataset!"

# ╔═╡ 0087fcc0-b2d5-4a5a-95fb-2e04d1a23b4c
md"### Building a model

A linear regression model is defined mathematically as -

```math
model(W, b, x) = Wx + b
```

where `W` is the weight matrix and `b` is the bias. For our case, the weight matrix (`W`) would constitute only a single element, as we have only a single feature. We can define our model in `Julia` using the exact same notation!"

# ╔═╡ f561fc45-52bd-447d-955f-aac44de097e9
custom_model(W, b, x) = @. W*x + b;

# ╔═╡ 2d5b3faa-d2d8-4277-9e3e-b389bdf38e64
md"The `@.` macro allows you to perform the calculations by broadcasting the scalar quantities (for example - the bias).

The next step would be to initialize the model parameters, which are the weight and the bias. There are a lot of initialization techniques available for different machine learning models, but for the sake of this example, let's pull out the weight from a uniform distribution and initialize the bias as `0`."

# ╔═╡ e948aeaa-03da-4471-88ae-61adc4b92d4c
W = rand(Float32, 1, 1)

# ╔═╡ 1b378240-f6ee-40ec-9f46-74750a2f2b96
b = [0.0f0]

# ╔═╡ c21b2236-9bfc-4fad-a36f-287bfd52dcf9
md"Time to test if our model works!"

# ╔═╡ 3e443f49-9b29-4a7b-a33f-64a42a965a01
custom_model(W, b, x_noise) |> size

# ╔═╡ 60dc1355-b348-40af-bf3b-2657384c23c3
custom_model(W, b, x_noise)[1], y[1]

# ╔═╡ b7c959c9-e8fb-4284-9695-8d8a12acca31
md"It does! But the predictions are way off. We need to train the model to improve the predictions, but before training the model we need to define the loss function. The loss function would ideally output a quantity that we will try to minimize during the entire training process. Here we will use the mean sum squared error loss function."

# ╔═╡ d2ec350e-f9b3-414f-b7ba-c54be7f78479
function custom_loss(W, b, x, y)
   ŷ = custom_model(W, b, x)
   sum((y .- ŷ).^2) / length(x)
end;

# ╔═╡ 85bd60df-d9e6-4bb8-9b3b-511d19468e9e
custom_loss(W, b, x_noise, y)

# ╔═╡ 1b491658-13cf-4c40-9256-378e371d8d3e
md"Calling the loss function on our `x`s and `y`s shows how far our predictions (`ŷ`) are from the real labels. More precisely, it calculates the sum of the squares of residuals and divides it by the total number of data points.

We have successfully defined our model and the loss function, but surprisingly, we haven't used `Flux` anywhere till now. Let's see how we can write the same code using `Flux`."

# ╔═╡ cda0bc35-5fff-4a64-ac8f-33acce5d3582
flux_model = Dense(1 => 1)

# ╔═╡ 337a5e1b-d6cd-44f9-aed0-1e71eb673c08
md"A [`Dense(1 => 1)`](@ref Dense) layer denotes a layer of one neuron with one input (one feature) and one output. This layer is exactly same as the mathematical model defined by us above! Under the hood, `Flux` too calculates the output using the same expression! But, we don't have to initialize the parameters ourselves this time, instead `Flux` does it for us."

# ╔═╡ 934ffccc-b5f6-4bf6-a1f9-ddaa35082cf2
flux_model.weight, flux_model.bias

# ╔═╡ 0be2ff21-c71a-4cf3-bd16-26ae640c519d
md"Now we can check if our model is acting right. We can pass the complete data in one go, with each `x` having exactly one feature (one input) -"

# ╔═╡ d82a8520-65b8-479d-ae8b-04a589d6b258
flux_model(x_noise) |> size

# ╔═╡ ab461ace-fb19-420d-bca4-717671cbf472
flux_model(x_noise)[1], y[1]

# ╔═╡ 5278c0fb-a1b9-4fa3-bef3-31d7e38ae55d
md"It is! The next step would be defining the loss function using `Flux`'s functions -"

# ╔═╡ a55fd878-f4d4-4591-8812-5ce1f0686df8
function flux_loss(flux_model, x, y)
   ŷ = flux_model(x)
   Flux.mse(ŷ, y)
end;

# ╔═╡ ab7ab7ce-9c72-48b3-871c-fa6ccdb4090e
flux_loss(flux_model, x_noise, y)

# ╔═╡ fd4f39bc-f3ed-4a50-abb5-5ffef941ff50
md"Everything works as before! It almost feels like `Flux` provides us with smart wrappers for the functions we could have written on our own. Now, as the last step of this section, let's see how different the `flux_model` is from our custom `model`. A good way to go about this would be to fix the parameters of both models to be the same. Let's change the parameters of our custom `model` to match that of the `flux_model` -"

# ╔═╡ 8d034f87-26c8-44b8-9220-12aabb721a62
W_new = Float32[-0.602168]

# ╔═╡ 70ff0680-3720-4e23-9b9d-c3a8ce6e48a5
md"To check how both the models are performing on the data, let's find out the losses using the `loss` and `flux_loss` functions -"

# ╔═╡ 263b2120-f60a-42f9-90c2-57fda79c1986
custom_loss(W_new, b, x_noise, y), flux_loss(flux_model, x_noise, y)

# ╔═╡ 4e8cd44c-5156-4b33-9373-9b46c4bf2a08
md"The losses are identical! This means that our `model` and the `flux_model` are identical on some level, and the loss functions are completely identical! The difference in models would be that `Flux`'s [`Dense`](@ref) layer supports many other arguments that can be used to customize the layer further. But, for this guide, let us stick to our simple custom `model`.

Let's train our model using the classic Gradient Descent algorithm. According to the gradient descent algorithm, the weights and biases should be iteratively updated using the following mathematical equations -

```math
\begin{aligned}
W &= W - \eta * \frac{dL}{dW} \\
b &= b - \eta * \frac{dL}{db}
\end{aligned}
```

Here, `W` is the weight matrix, `b` is the bias vector, ``\eta`` is the learning rate, ``\frac{dL}{dW}`` is the derivative of the loss function with respect to the weight, and ``\frac{dL}{db}`` is the derivative of the loss function with respect to the bias.

The derivatives are calculated using an Automatic Differentiation tool, and `Flux` uses [`Zygote.jl`](https://github.com/FluxML/Zygote.jl) for the same. Since `Zygote.jl` is an independent Julia package, it can be used outside of Flux as well! Refer to the documentation of `Zygote.jl` for more information on the same.

Our first step would be to obtain the gradient of the loss function with respect to the weights and the biases. `Flux` re-exports `Zygote`'s `gradient` function; hence, we don't need to import `Zygote` explicitly to use the functionality."

# ╔═╡ 58531fc8-f382-40c9-bdad-c1ee0ebe8182
dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x_noise, y);

# ╔═╡ e13d356b-fdb3-4d76-9477-0f535b80c4c8
md"We can now update the parameters, following the gradient descent algorithm -"

# ╔═╡ b649f852-0c05-46a1-af72-eb77688b86ba
W .= W .- 0.1 .* dLdW

# ╔═╡ b0359723-4f79-4055-be42-936029d2405e
b .= b .- 0.1 .* dLdb

# ╔═╡ 07d6fa2f-8b8d-4eb7-b99e-59af0370e329
md"The parameters have been updated! We can now check the value of the loss function -"

# ╔═╡ 1823eba1-4469-4762-b2a7-f3baf7a5216b
custom_loss(W, b, x_noise, y)

# ╔═╡ 583b8c06-6894-4d24-aea6-d4b06380c077
md"The loss went down! This means that we successfully trained our model for one epoch. We can plug the training code written above into a loop and train the model for a higher number of epochs. It can be customized either to have a fixed number of epochs or to stop when certain conditions are met, for example, `change in loss < 0.1`. The loop can be tailored to suit the user's needs, and the conditions can be specified in plain `Julia`!

Let's plug our super training logic inside a function and test it again -"

# ╔═╡ 1745db83-00b3-429e-b13a-3464c4e8a4de
function train_custom_model()
   dLdW, dLdb, _, _ = gradient(custom_loss, W, b, x_noise, y)
   @. W = W - 0.1 * dLdW
   @. b = b - 0.1 * dLdb
end;

# ╔═╡ f28f724a-880d-4cc5-9087-b73547c48bef
train_custom_model();

# ╔═╡ b058ebcc-e2b0-4049-9196-02ac6a29125d
W, b, custom_loss(W, b, x_noise, y)

# ╔═╡ 2d11751e-75fb-4a18-bf6d-75bfd5127bf8
md"It works, and the loss went down again! This was the second epoch of our training procedure. Let's plug this in a for loop and train the model for 40 epochs."

# ╔═╡ 9d96ebc0-d470-435c-989f-4952bfd97ac9
for i = 1:40
  train_custom_model()
end

# ╔═╡ b06228cf-128c-4a3c-9a04-82f3fa3b99e1
W, b, custom_loss(W, b, x_noise, y)

# ╔═╡ 8e545a62-120a-42f0-8f8b-54fe6c88e442
md"There was a significant reduction in loss, and the parameters were updated!

We can train the model even more or tweak the hyperparameters to achieve the desired result faster, but let's stop here. We trained our model for 42 epochs, and loss went down from `27.75534` to `6.71062`. Time for some visualization!

### Results
The main objective of this guide was to fit a line to our dataset using the linear regression algorithm. The training procedure went well, and the loss went down significantly! Let's see what the fitted line looks like. Remember, `Wx + b` is nothing more than a line's equation, with `slope = W[1]` and `y-intercept = b[1]` (indexing at `1` as `W` and `b` are iterable)."

# ╔═╡ 5e877dc6-ab57-4f52-b828-14629e312407
begin
	plot(vec(x_noise), vec(y), lw = 3, seriestype = :scatter, label = "", title = "Simple Linear Regression", xlabel = "x", ylabel= "y")
	plot!((x) -> b[1] + W[1] * x, -3, 3, label="Custom model", lw=2);
end

# ╔═╡ 6f958796-e26c-4940-99d1-61fa4436b6c9
md"The line fits well! There is room for improvement, but we leave that up to you! You can play with the optimisers, the number of epochs, learning rate, etc. to improve the fitting and reduce the loss!"

# ╔═╡ ea5cfbcb-83e4-4e54-ad7d-280d5100ac17
md"## Linear regression model on a real dataset
We now move on to a relatively complex linear regression model. Here we will use a real dataset from [`MLDatasets.jl`](https://github.com/JuliaML/MLDatasets.jl), which will not confine our data points to have only one feature. Let's start by importing the required packages -"

# ╔═╡ 56c79294-3cad-40b9-bb2b-fdc92ec852d3
md"### Data
Let's start by initializing our dataset. We will be using the [`BostonHousing`](https://juliaml.github.io/MLDatasets.jl/stable/datasets/misc/#MLDatasets.BostonHousing) dataset consisting of `506` data points. Each of these data points has `13` features and a corresponding label, the house's price. The `x`s are still mapped to a single `y`, but now, a single `x` data point has 13 features. "

# ╔═╡ ca59df52-2f41-4f30-9529-09c0558eccc0
x_real = BostonHousing.features()

# ╔═╡ 3d22dfee-f08a-4019-a90a-d25dc574f10d
y_real = BostonHousing.targets()

# ╔═╡ 1852b26d-e673-451b-b04c-90e0777003ea
md"We can now split the obtained data into training and testing data -"

# ╔═╡ c25915c9-e991-4f52-8780-c593cc6b312a
x_train, x_test, y_train, y_test = x_real[:, 1:400], x_real[:, 401:end], y_real[:, 1:400], y_real[:, 401:end];

# ╔═╡ 1964319c-91cf-40e8-9633-9bdec2c47253
x_train |> size, x_test |> size, y_train |> size, y_test |> size

# ╔═╡ 5e0fe0ea-2efb-4d97-8a65-ef02cf655822
md"This data contains a diverse number of features, which means that the features have different scales. A wise option here would be to `normalise` the data, making the training process more efficient and fast. Let's check the standard deviation of the training data before normalising it."

# ╔═╡ 4314d7c7-da0d-432c-9432-f986045ee016
std(x_train)

# ╔═╡ 409531fa-77c6-47a1-a6ba-f4616d18f6c2
md"The data is indeed not normalised. We can use the [`Flux.normalise`](@ref) function to normalise the training data."

# ╔═╡ 076f84d7-3586-41c1-a659-78bc45b844eb
x_train_n = Flux.normalise(x_train);

# ╔═╡ 8e7da7e8-b422-4743-908e-467940f6d897
std(x_train_n)

# ╔═╡ 5ec0f9ca-f48d-47e8-af68-1a6f669ce4c2
md"The standard deviation is now close to one! Our data is ready!

### Model
We can now directly use `Flux` and let it do all the work internally! Let's define a model that takes in 13 inputs (13 features) and gives us a single output (the label). We will then pass our entire data through this model in one go, and `Flux` will handle everything for us! Remember, we could have declared a model in plain `Julia` as well. The model will have 14 parameters: 13 weights and 1 bias."

# ╔═╡ c61dcdc4-538c-4d1d-bce4-7ffa15182a0f
model = Dense(13 => 1)

# ╔═╡ b2a5f069-189f-448e-8cd8-72ff2d979c97
md"Same as before, our next step would be to define a loss function to quantify our accuracy somehow. The lower the loss, the better the model!"

# ╔═╡ 83e37331-c2e5-48b0-af36-6bc6fe752963
function loss(model, x, y)
   ŷ = model(x)
   Flux.mse(ŷ, y)
end;

# ╔═╡ e2e9e513-b16d-4ffb-a32a-ab27df8eb65d
loss(model, x_train_n, y_train)

# ╔═╡ 14b7c355-352a-44a3-9056-be7b0faf7519
md"We can now proceed to the training phase!

### Training
The training procedure would make use of the same mathematics, but now we can pass in the model inside the `gradient` call and let `Flux` and `Zygote` handle the derivatives!"

# ╔═╡ 3311405f-fe17-46ce-898a-fdfe7afd2c69
function train_model()
   dLdm, _, _ = gradient(loss, model, x_train_n, y_train)
   @. model.weight = model.weight - 0.000001 * dLdm.weight
   @. model.bias = model.bias - 0.000001 * dLdm.bias
end;

# ╔═╡ 0e2c8462-18ee-497e-9789-25a44f2f6f91
md"Contrary to our last training procedure, let's say that this time we don't want to hardcode the number of epochs. We want the training procedure to stop when the loss converges, that is, when `change in loss < δ`. The quantity `δ` can be altered according to a user's need, but let's fix it to `10⁻³` for this guide.

We can write such custom training loops effortlessly using `Flux` and plain `Julia`!"

# ╔═╡ 5139732e-6fb9-4ceb-a61b-74d0b29adf98
loss_init = Inf;

# ╔═╡ 2f1c2716-c3e1-49b7-b487-44b031d0ce9c
while true
   train_model()
   if loss_init == Inf
	   loss_init = loss(model, x_train_n, y_train)
	   continue
   end
   if abs(loss_init - loss(model, x_train_n, y_train)) < 1e-5
	   break
   else
	   loss_init = loss(model, x_train_n, y_train)
   end
end;

# ╔═╡ b7195a97-a2ee-45a5-9c97-f2d8544c3c5c
md"The code starts by initializing an initial value for the loss, `infinity`. Next, it runs an infinite loop that breaks if `change in loss < 10⁻³`, or the code changes the value of `loss_init` to the current loss and moves on to the next iteration.

This custom loop works! This shows how easily a user can write down any custom training routine using Flux and Julia!

Let's have a look at the loss -"

# ╔═╡ 835396c0-44d2-4878-a3f4-41aeb3a35479
loss(model, x_train_n, y_train)

# ╔═╡ a065f4ed-f9f8-43fe-8c54-7f3490889c94
md"The loss went down significantly! It can be minimized further by choosing an even smaller `δ`.

### Testing
The last step of this guide would be to test our model using the testing data. We will first normalise the testing data and then calculate the corresponding loss."

# ╔═╡ 8b544789-bfbe-41bb-9899-35c7b8176cd2
x_test_n = Flux.normalise(x_test);

# ╔═╡ 72ec286d-6832-4924-95cf-b438741eb3c1
loss(model, x_test_n, y_test)

# ╔═╡ daef8ba8-5f8b-496f-9276-3f33e68a22fc
md"The loss is not as small as the loss of the training data, but it looks good! This also shows that our model is not overfitting!

---

Summarising this guide, we started by generating a random yet correlated dataset for our custom model. We then saw how a simple linear regression model could be built with and without `Flux`, and how they were almost identical. 

Next, we trained the model by manually writing down the Gradient Descent algorithm and optimising the loss. We also saw how `Flux` provides various wrapper functionalities and keeps the API extremely intuitive and simple for the users. 

After getting familiar with the basics of `Flux` and `Julia`, we moved ahead to build a machine learning model for a real dataset. We repeated the exact same steps, but this time with a lot more features and data points, and by harnessing `Flux`'s full capabilities. In the end, we developed a training loop that was smarter than the hardcoded one and ran the model on our normalised dataset to conclude the tutorial."

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
MLDatasets = "eb30cadb-4394-5ae3-aed4-317e484a6458"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
DataFrames = "~1.3.4"
Flux = "~0.13.5"
MLDatasets = "~0.5.15"
Plots = "~1.31.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "c877a35749324754d3c8fffb09fc1f9db144ff8f"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.18"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["ArrayInterfaceCore", "Compat", "IfElse", "LinearAlgebra", "Static"]
git-tree-sha1 = "0582b5976fc76523f77056e888e454f0f7732596"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "6.0.22"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "40debc9f72d0511e12d817c7ca06a721b6423ba3"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.17"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "a598ecb0d717092b5539dbbe890c98bac842b072"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.2.0"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "b15a6bc52594f5e4a3b825858d1089618871bf9d"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.36"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BinDeps]]
deps = ["Libdl", "Pkg", "SHA", "URIParser", "Unicode"]
git-tree-sha1 = "1289b57e8cf019aede076edab0587eb9644175bd"
uuid = "9e28174c-4ba2-5203-b857-d8d62c4213ee"
version = "1.0.2"

[[deps.BinaryProvider]]
deps = ["Libdl", "Logging", "SHA"]
git-tree-sha1 = "ecdec412a9abc8db54c0efc5548c64dfce072058"
uuid = "b99e7846-7c00-51b0-8f62-c81ae34c0232"
version = "0.5.10"

[[deps.BufferedStreams]]
git-tree-sha1 = "bb065b14d7f941b8617bc323063dbe79f55d16ea"
uuid = "e1450e63-4bb3-523b-b2a4-4ffa8c0fd77d"
version = "1.1.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "TimerOutputs"]
git-tree-sha1 = "49549e2c28ffb9cc77b3689dc10e46e6271e9452"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "3.12.0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "650415ad4c2a007b17f577cb081d9376cc908b6f"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.44.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "80ca332f6dcb2508adba68f22f551adb2d00a624"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.3"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "9be8be1d8a6f44b96482c8af52238ea7987da3e3"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.45.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59d00b3139a9de4eb961057eabb65ac6522be954"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.4.0"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "8ccaa8c655bc1b83d2da4d569c9b28254ababd6e"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.2"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "fb5f5316dd3fd4c5e7c30a24d50643b73e37cd40"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.10.0"

[[deps.DataDeps]]
deps = ["BinaryProvider", "HTTP", "Libdl", "Reexport", "SHA", "p7zip_jll"]
git-tree-sha1 = "8e2713f4c3394fba835c0babd0bfd2b3cd007fd9"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.9"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "daa21eb85147f72e41f6352a57fccea377e310a9"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.3.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "28d605d9a0ac17118fe2c5e9ce0fbb76c3ceb120"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "5158c2b41018c5f7eb1470d558127ac274eca0c9"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.1"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bad72f730e9e91c08d9427d5e8db95478a3c323d"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.8+0"

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

[[deps.Extents]]
git-tree-sha1 = "5e1e4c53fa39afe63a7d356e30452249365fba99"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.1"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccd479984c7838684b3ac204b716c89955c76623"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+0"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "4391d3ed58db9dc5a9883b23a0578316b4798b1f"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.0"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "246621d23d1f43e3b9c368bf3b72b2331a27c286"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Flux]]
deps = ["Adapt", "ArrayInterface", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "Optimisers", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "Test", "Zygote"]
git-tree-sha1 = "9b5419ad6f043ac2b52f1b7f9a8ecb8762231214"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.5"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "187198a4ed8ccd7b5d99c41b69c679269ea2b2d4"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.32"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "241552bc2209f0fa068b6415b1942cc0aa486bcc"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.2"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "a2657dd0f3e8a61dbe70fc7c122038bd33790af5"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.3.0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "d972031d28c8c8d9d7b41a536ad7bb0c2579caca"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.8+0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "45d7deaf05cbb44116ba785d147c518ab46352d7"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.5.0"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "122d7bcc92abf94cf1a86281ad7a4d0e838ab9e0"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.16.3"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "cf0a9940f250dc3cb6cc6c6821b4bf8a4286cf9c"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.66.2"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "2d908286d120c584abbe7621756c341707096ba4"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.66.2+0"

[[deps.GZip]]
deps = ["Libdl"]
git-tree-sha1 = "039be665faf0b8ae36e089cd694233f5dee3f7d6"
uuid = "92fee26a-97fe-5a0c-ad85-20a5f3185b63"
version = "0.5.1"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "fb28b5dc239d0174d7297310ef7b84a11804dfab"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.0.1"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "a7a97895780dab1085a97769316aa348830dc991"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.3"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HDF5]]
deps = ["Compat", "HDF5_jll", "Libdl", "Mmap", "Random", "Requires"]
git-tree-sha1 = "9ffc57b9bb643bf3fce34f3daf9ff506ed2d8b7a"
uuid = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
version = "0.16.10"

[[deps.HDF5_jll]]
deps = ["Artifacts", "JLLWrappers", "LibCURL_jll", "Libdl", "OpenSSL_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "c003b31e2e818bc512b0ff99d7dce03b0c1359f5"
uuid = "0234f1f7-429e-5d53-9886-15a909be8d59"
version = "1.12.2+1"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "Dates", "IniFile", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "f0956f8d42a92816d2bf062f8a6a6a0ad7f9b937"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.2.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "af14a478780ca78d5eb9908b263023096c2b9d64"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.6"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InternedStrings]]
deps = ["Random", "Test"]
git-tree-sha1 = "eb05b5625bc5d821b8075a77e4c421933e20c76b"
uuid = "7d512f48-7fb1-5a58-b986-67e6dc259f01"
version = "0.7.0"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "b3364212fb5d870f724876ffcd34dd8ec6d98918"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "fd6f0cae36f42525567108a42c1c674af2ac620d"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "e7e9184b0bf0158ac4e4aa9daf00041b5909bf1a"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "4.14.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "771bfe376249626d3ca12bcd58ba243d3f961576"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.16+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "1a43be956d433b5d0321197150c2f94e16c0aaa0"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.16"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "361c2b088575b07946508f135ac556751240091c"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.17"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "5d4d2d9904227b8bd66386c1138cf4d5ffa826bf"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "0.4.9"

[[deps.MAT]]
deps = ["BufferedStreams", "CodecZlib", "HDF5", "SparseArrays"]
git-tree-sha1 = "971be550166fe3f604d28715302b58a3f7293160"
uuid = "23992714-dd62-5051-b70f-ba57cb901cac"
version = "0.10.3"

[[deps.MLDatasets]]
deps = ["BinDeps", "ColorTypes", "DataDeps", "DelimitedFiles", "FixedPointNumbers", "GZip", "JSON3", "MAT", "Pickle", "Requires", "SparseArrays"]
git-tree-sha1 = "f1ff456828cfceb8fd64f2f212dea67b1414be96"
uuid = "eb30cadb-4394-5ae3-aed4-317e484a6458"
version = "0.5.15"

[[deps.MLStyle]]
git-tree-sha1 = "c4f433356372cc8838da59e3608be4b0c4c2c280"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.13"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "DelimitedFiles", "FLoops", "FoldsThreads", "Random", "ShowCases", "Statistics", "StatsBase", "Transducers"]
git-tree-sha1 = "7fd41b7edef1d58062a75c2f129e839a8d168fe9"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.2.10"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "Random", "Sockets"]
git-tree-sha1 = "d9ab10da9de748859a7780338e1d6566993d1f25"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "6bb7786e4f24d44b4e29df03c69add1b63d88f01"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NNlib]]
deps = ["Adapt", "ChainRulesCore", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "415108fd88d6f55cedf7ee940c7d4b01fad85421"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.9"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics"]
git-tree-sha1 = "4429261364c5ea5b7308aecaa10e803ace101631"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e60321e3f2616584ff98f0a4f18d98ae6f89bbb3"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.17+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1ef34738708e3f31994b52693286dabcb3d29f6b"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.9"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "0044b23da09b5608b4ecacb4e5e6c6332f833a7e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.2"

[[deps.Pickle]]
deps = ["DataStructures", "InternedStrings", "Serialization", "SparseArrays", "Strided", "StringEncodings", "ZipFile"]
git-tree-sha1 = "e6a34eb1dc0c498f0774bbfbbbeff2de101f4235"
uuid = "fbb45041-c46e-462f-888f-7c521cafbc2c"
version = "0.3.2"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "8162b2f8547bc23876edd0c5181b27702ae58dce"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.0.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "9888e59493658e476d3073f1ce24348bdc086660"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.3.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "a19652399f43938413340b2068e11e55caa46b65"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.31.7"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "dfb54c4e414caa595a1f2ed759b160f5a3ddcba5"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "7a1a306b72cfa60634f03a911405f4e64d1b718b"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.0"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "e7eac76a958f8664f2718508435d058168c7953d"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.3"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "22c5201127d7b243b9ee1de3b43c408879dff60f"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.3.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "f94f779c94e58bf9ea243e77a37e16d9de9126bd"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "38d88503f695eb0301479bc9b0d4320b378bafe5"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.8.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "39c9f91521de844bad65049efd4f9223e7ed43f9"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.14"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f94f9d627ba3f91e41a815b9f9f977d729e2e06f"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.7.6"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "85bc4b051546db130aeb1e8a696f1da6d4497200"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.5"

[[deps.StaticArraysCore]]
git-tree-sha1 = "5b413a57dd3cea38497d745ce088ac8592fbb5be"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.1.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.Strided]]
deps = ["LinearAlgebra", "TupleTools"]
git-tree-sha1 = "a7a664c91104329c88222aa20264e1a05b6ad138"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "1.2.3"

[[deps.StringEncodings]]
deps = ["Libiconv_jll"]
git-tree-sha1 = "50ccd5ddb00d19392577902f0079267a72c5ab04"
uuid = "69024149-9ee7-55f6-a4c4-859efe599b68"
version = "0.3.5"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArraysCore", "Tables"]
git-tree-sha1 = "8c6ac65ec9ab781af05b08ff305ddc727c25f680"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.12"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "79aa7175f0149ba2fe22b96a271f4024429de02d"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.9.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "9dfcb767e17b0849d6aaf85997c98a5aea292513"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.21"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "4ad90ab2bbfdddcae329cba59dab4a8cdfac3832"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.7"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "c76399a3bbe6f5a88faa33c8f8a65aa631d95013"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.73"

[[deps.TupleTools]]
git-tree-sha1 = "3c712976c47707ff893cf6ba4354aa14db1d8938"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.3.0"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "e59ecc5a41b000fa94423a578d29290c7266fc10"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "58443b63fb7e465a8a7210828c91c08b92132dff"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.14+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "ef4f23ffde3ee95114b461dc667ea4e6906874b2"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.10.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "8ac61a92a33b3fd2a4cbf92951817831e313a004"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.44"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9ebfc140cc56e8c2156a15ceac2f0302e327ac0a"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+0"
"""

# ╔═╡ Cell order:
# ╟─2e286889-de4f-4147-8255-2522a509495f
# ╟─c727549c-dcab-48b0-b000-bcd63f35603b
# ╠═e6c4d3d0-212c-11ed-04dc-91f7d92441a1
# ╟─6a85e00d-c15d-42bf-9483-eede897fbd1d
# ╠═5509254a-a076-421d-9646-d226ef62c49f
# ╟─b5ffe55a-3405-4ef8-bf9d-78890fadf9bd
# ╠═572bd321-aa20-4090-b819-d81ccf1e244b
# ╠═1c1b0764-f926-489d-833e-835fdcaad7db
# ╟─b2c6b0f6-e388-48a3-a4f0-c59554187f07
# ╠═3b0b47e0-e38e-42f0-9d71-6e9a01f8a72f
# ╟─a7fd511c-be36-4c65-b3c2-1b362ec4268b
# ╠═059bfedb-aac5-42cf-baa8-a10dab14f032
# ╟─357853d5-fd49-46fd-a604-c766f33814e7
# ╟─0087fcc0-b2d5-4a5a-95fb-2e04d1a23b4c
# ╠═f561fc45-52bd-447d-955f-aac44de097e9
# ╟─2d5b3faa-d2d8-4277-9e3e-b389bdf38e64
# ╠═e948aeaa-03da-4471-88ae-61adc4b92d4c
# ╠═1b378240-f6ee-40ec-9f46-74750a2f2b96
# ╟─c21b2236-9bfc-4fad-a36f-287bfd52dcf9
# ╠═3e443f49-9b29-4a7b-a33f-64a42a965a01
# ╠═60dc1355-b348-40af-bf3b-2657384c23c3
# ╟─b7c959c9-e8fb-4284-9695-8d8a12acca31
# ╠═d2ec350e-f9b3-414f-b7ba-c54be7f78479
# ╠═85bd60df-d9e6-4bb8-9b3b-511d19468e9e
# ╟─1b491658-13cf-4c40-9256-378e371d8d3e
# ╠═cda0bc35-5fff-4a64-ac8f-33acce5d3582
# ╟─337a5e1b-d6cd-44f9-aed0-1e71eb673c08
# ╠═934ffccc-b5f6-4bf6-a1f9-ddaa35082cf2
# ╟─0be2ff21-c71a-4cf3-bd16-26ae640c519d
# ╠═d82a8520-65b8-479d-ae8b-04a589d6b258
# ╠═ab461ace-fb19-420d-bca4-717671cbf472
# ╟─5278c0fb-a1b9-4fa3-bef3-31d7e38ae55d
# ╠═a55fd878-f4d4-4591-8812-5ce1f0686df8
# ╠═ab7ab7ce-9c72-48b3-871c-fa6ccdb4090e
# ╟─fd4f39bc-f3ed-4a50-abb5-5ffef941ff50
# ╠═8d034f87-26c8-44b8-9220-12aabb721a62
# ╟─70ff0680-3720-4e23-9b9d-c3a8ce6e48a5
# ╠═263b2120-f60a-42f9-90c2-57fda79c1986
# ╟─4e8cd44c-5156-4b33-9373-9b46c4bf2a08
# ╠═58531fc8-f382-40c9-bdad-c1ee0ebe8182
# ╟─e13d356b-fdb3-4d76-9477-0f535b80c4c8
# ╠═b649f852-0c05-46a1-af72-eb77688b86ba
# ╠═b0359723-4f79-4055-be42-936029d2405e
# ╟─07d6fa2f-8b8d-4eb7-b99e-59af0370e329
# ╠═1823eba1-4469-4762-b2a7-f3baf7a5216b
# ╟─583b8c06-6894-4d24-aea6-d4b06380c077
# ╠═1745db83-00b3-429e-b13a-3464c4e8a4de
# ╠═f28f724a-880d-4cc5-9087-b73547c48bef
# ╠═b058ebcc-e2b0-4049-9196-02ac6a29125d
# ╟─2d11751e-75fb-4a18-bf6d-75bfd5127bf8
# ╠═9d96ebc0-d470-435c-989f-4952bfd97ac9
# ╠═b06228cf-128c-4a3c-9a04-82f3fa3b99e1
# ╟─8e545a62-120a-42f0-8f8b-54fe6c88e442
# ╠═5e877dc6-ab57-4f52-b828-14629e312407
# ╟─6f958796-e26c-4940-99d1-61fa4436b6c9
# ╟─ea5cfbcb-83e4-4e54-ad7d-280d5100ac17
# ╠═54b7fffb-2d7a-469a-b66e-b6b7aa466387
# ╠═d9ab4748-4b6a-473e-b728-d1936612e1c1
# ╟─56c79294-3cad-40b9-bb2b-fdc92ec852d3
# ╠═ca59df52-2f41-4f30-9529-09c0558eccc0
# ╠═3d22dfee-f08a-4019-a90a-d25dc574f10d
# ╟─1852b26d-e673-451b-b04c-90e0777003ea
# ╠═c25915c9-e991-4f52-8780-c593cc6b312a
# ╠═1964319c-91cf-40e8-9633-9bdec2c47253
# ╟─5e0fe0ea-2efb-4d97-8a65-ef02cf655822
# ╠═4314d7c7-da0d-432c-9432-f986045ee016
# ╟─409531fa-77c6-47a1-a6ba-f4616d18f6c2
# ╠═076f84d7-3586-41c1-a659-78bc45b844eb
# ╠═8e7da7e8-b422-4743-908e-467940f6d897
# ╟─5ec0f9ca-f48d-47e8-af68-1a6f669ce4c2
# ╠═c61dcdc4-538c-4d1d-bce4-7ffa15182a0f
# ╠═b2a5f069-189f-448e-8cd8-72ff2d979c97
# ╠═83e37331-c2e5-48b0-af36-6bc6fe752963
# ╠═e2e9e513-b16d-4ffb-a32a-ab27df8eb65d
# ╟─14b7c355-352a-44a3-9056-be7b0faf7519
# ╠═3311405f-fe17-46ce-898a-fdfe7afd2c69
# ╟─0e2c8462-18ee-497e-9789-25a44f2f6f91
# ╠═5139732e-6fb9-4ceb-a61b-74d0b29adf98
# ╠═2f1c2716-c3e1-49b7-b487-44b031d0ce9c
# ╟─b7195a97-a2ee-45a5-9c97-f2d8544c3c5c
# ╠═835396c0-44d2-4878-a3f4-41aeb3a35479
# ╟─a065f4ed-f9f8-43fe-8c54-7f3490889c94
# ╠═8b544789-bfbe-41bb-9899-35c7b8176cd2
# ╠═72ec286d-6832-4924-95cf-b438741eb3c1
# ╟─daef8ba8-5f8b-496f-9276-3f33e68a22fc
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
