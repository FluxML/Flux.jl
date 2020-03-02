using .Optimise: stop
using BSON

"""
   init_cb()

Function to initialise global variables required in various callback functions.
"""
function init_cb()
    global ctr = 0                     #global counter for how many times the callback has been called
    global stop_itr = false
    global loss_history = []
    global best_acc = 0
    global best_loss = Inf
    global last_improvement = 0 
end

"""
   terminateOnNaN(x,y)
 
Callback that terminates training when a NaN loss is encountered. Here, x and y could be training data (x,y) or can be different.
"""
function terminateOnNaN(x,y)
       if isnan(loss(x,y))
           stop_itr = true
           @info "NaN loss, Terminating execution!!!!"
           stop()
           end
       end

"""
    history(x,y,accuracy)

Callback that records loss and accuracy into an array. Array can be accessed using `loss_history`.
Arguments:
    x,y: Could be training data (x,y) or can be different
    
    accuracy(optional): function to be used to calculate accuracy on x,y. If not provided, loss_history returns only loss
"""
function history(x,y)
           push!(loss_history,loss(x,y))
           end

function history(x,y,accuracy)
           push!(loss_history,[loss(x,y),accuracy(x,y)])
           end

"""
    lrdecay(x,y;factor = 0.2,loss = loss,accuracy_fn = nothing,patience=5,min_lr = 0.000001,monitor="acc")

Learning Rate Decay based on previous performance. Reduce learning rate when a metric has stopped improving.
Arguments:
    x,y: could be training data (x,y) or can be different

    factor: factor by which learning rate is reduced in every updation. Default set to 0.2
    
    loss: function to be used to calculate loss, when monitor set to 'loss'. Otherwise, optional.
    
    accuracy_fn: function to be used to calculate accuracy, when monitor set to 'acc'. Otherwise, optional
    
    patience: number of epochs that produced the monitored quantity with no improvement, 
	      after which learning rate will be reduced or training stopped. Default value set to 5.
    
    min_lr: lower bound on the learning rate. If no improvement seen even below that after 'patience' number of steps, the training is terminated.
    
    monitor: Quantity to be monitored for the provided (x,y). Can take values 'acc' or 'loss'. Default set to 'loss'
"""

function lrdecay(x,y;factor = 0.2,loss = loss,accuracy_fn = nothing,patience=5,min_lr = 1e-5,monitor="loss")
            global ctr+=1
            global best_acc,last_improvement,best_loss,stop_itr
            if monitor == "acc"
               acc = accuracy_fn(x,y)
               if acc>best_acc
                   best_acc = acc
                   last_improvement = ctr
               elseif (ctr-last_improvement)>patience
                   if opt.eta>min_lr
                       @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
                       opt.eta = factor*opt.eta
                       last_improvement = ctr
                   else 
                       stop_itr = true
                       @warn("We are calling this converged")
                       stop()
                   end
               end
            elseif monitor== "loss"
                loss_ = loss(x,y)
                if loss_<best_loss
                   best_loss = loss_
                   last_improvement = ctr
                elseif (ctr-last_improvement)>patience
                   if opt.eta>min_lr
                       @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")
                       opt.eta = factor*opt.eta
                       last_improvement = ctr
                   else 
                       stop_itr = true
                       @warn("We are calling this as converged")
                       stop()
                   end
                end
        else
            @warn("Invalid argument for 'monitor'")
            end
end

"""
    model_checkpoint(x,y,model_arr;loss = loss,accuracy_fn=nothing,monitor="acc",
				filename= "model.bson",path = "./",verbose=1,save_best_model_only =1)

Saves the model after each epoch whenever the monitored qunatity improves.
Arguments:
    x,y:  Could be training data (x,y) or can be different
    
    model_arr: Array of sub-models being used in the model. For a single model input should be [model]. 
	       For more sub_models provide models in sequence,[model1,model2,....] 
    
    loss: function to be used to calculate loss, when monitor set to 'loss'. Otherwise, optional.
    
    accuracy_fn: function to be used to calculate accuracy, when monitor set to 'acc'. Otherwise, optional.
    
    monitor: monitor: quantity to be monitored for the provided (x,y). Can take values 'acc' or 'loss'. Default set to 'loss'
    
    filename: name of the model file with extension .BSON to which model is needed to be saved. 
	      By default, saved to 'model.bson'.
    
    filepath: Path to the file where the model is needed to be saved. By default, set to "./", 
	      i.e. saves files to current directory
    
    save_best_model_only: save only the best model to `filename` provided, if set to 1. 
			  Else save models at every improvement to different files as {epoch}_filename. Default set to 1.

In order to load the saved model, use:  `BSON.@load joinpath(path, filename) m ctr acc` or `BSON.@load joinpath(path, filename) m ctr l`<br> 
where m represent a chain of all the sub-models of the function. To access each individual model, use m.layers[i] for ith sub-model.
"""
function model_checkpoint(x,y,model_arr;loss = loss,accuracy_fn = nothing,monitor = "loss",
		filename= "model.bson",path = "./",verbose=1,save_best_model_only=1)   
	global ctr = ctr + 1
    global best_acc,last_improvement,best_loss
    if  monitor == "acc"
            acc = accuracy_fn(x,y)
        if acc>=best_acc
            if verbose==1
                if save_best_model_only ==1
                @warn(" -> Monitored `$monitor` improved ! Saving model out to $path$filename")
                else
                @warn(" -> Monitored `$monitor` improved! Saving model out to $path{$ctr}_$filename")
                end
            end
            m = Chain(model_arr...)
            if save_best_model_only ==1
                BSON.@save joinpath(path, filename) m ctr acc
            else
                BSON.@save joinpath(path, string("{$ctr}_",filename)) m ctr acc
            end
            global best_acc = acc
        end
    elseif monitor == "loss"
            l = loss(x,y)
        if l<best_loss  
            if verbose==1
                if save_best_model_only ==1
                @warn(" -> Monitored `$monitor` improved ! Saving model out to $path$filename")
                else
                @warn(" -> Monitored `$monitor` improved! Saving model out to $path{$ctr}_$filename")
                end
            end
            m = Chain(model_arr...)
            if save_best_model_only ==1
                BSON.@save joinpath(path, filename) m ctr l
            else
                BSON.@save joinpath(path, string("{ctr}_",filename)) m ctr l
            end
            global best_loss = l
        end
    else
        @warn("Invalid argument for 'monitor'")
    end
end
