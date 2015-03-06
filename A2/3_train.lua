require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

-- CUDA?
if opt.type == 'cuda' then
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- classes
classes = {'1','2','3','4','5','6','7','8','9','0'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end

----------------------------------------------------------------------
print '==> configuring optimizer'

if opt.optimization == 'CG' then
   optimState = {
      maxIter = opt.maxIter
   }
   optimMethod = optim.cg

elseif opt.optimization == 'LBFGS' then
   optimState = {
      learningRate = opt.learningRate,
      maxIter = opt.maxIter,
      nCorrection = 10
   }
   optimMethod = optim.lbfgs

elseif opt.optimization == 'SGD' then
   optimState = {
      learningRate = opt.learningRate,
      weightDecay = opt.weightDecay,
      momentum = opt.momentum,
      evalCounter = epoch or 0,
      learningRateDecay = opt.learningRateDecay
   }
   optimMethod = optim.sgd

elseif opt.optimization == 'ASGD' then
   optimState = {
      eta0 = opt.learningRate,
      t0 = trsize * opt.t0
   }
   optimMethod = optim.asgd

else
   error('unknown optimization method')
end

----------------------------------------------------------------------
print '==> defining training procedure'

function train(save)

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- set model to training mode (for modules that differ in training and testing, like Dropout)
   model:training()

   -- shuffle at each epoch
   shuffle = torch.randperm(trainData:size())

   -- do one epoch
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   parameters, gradParameters = model:getParameters()
   for t = 0,trainData:size() - opt.batchSize, opt.batchSize do
      -- disp progress
      --xlua.progress(t, trainData:size())

      -- create mini batch
      local inputs = torch.Tensor(128,3,96,96)
      local targets = torch.Tensor(128)
      for i = 1, opt.batchSize do
      --for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
         -- load new sample
         inputs[i] = trainData.data[shuffle[t+i]]
         targets[i] = trainData.labels[shuffle[t+i]]
         --local input = trainData.data[shuffle[i]]
         --local target = trainData.labels[shuffle[i]]
      end
      if opt.type == 'double' then inputs = inputs:double()
      elseif opt.type == 'cuda' then
         inputs = inputs:cuda()
         targets = targets:cuda()
      end
      


      local clr
      gradParameters:zero()
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      model:backward(inputs, criterion:backward(outputs, targets))
      clr = opt.learningRate * (0.5 ^ math.floor(epoch / opt.learningRateDecay))
      parameters:add(-clr, gradParameters)
      for i = 1, opt.batchSize do
         confusion:add(outputs[i], targets[i])
      end
      

      -- create closure to evaluate f(X) and df/dX
      -- local feval = function(x)
      --                  -- get new parameters
      --                  if x ~= parameters then
      --                     parameters:copy(x)
      --                  end

      --                  -- reset gradients
      --                  gradParameters:zero()

      --                  -- f is the average of all criterions

      --                  -- evaluate function for complete mini batch
      --                  local output = model:forward(inputs)
      --                  local f = criterion:forward(output, targets)
      --                  local df_do = criterion:backward(output, targets)
      --                  model:backward(inputs, df_do)
      --                  -- for i = 1,#inputs do
      --                  --    -- estimate f
      --                  --    local output = model:forward(inputs[i])
      --                  --    local err = criterion:forward(output, targets[i])
      --                  --    f = f + err

      --                  --    -- estimate df/dW
      --                  --    local df_do = criterion:backward(output, targets[i])
      --                  --    model:backward(inputs[i], df_do)

      --                  --    -- update confusion
      --                  for i = 1, opt.batchSize do
      --                     confusion:add(output[i], targets[i])
      --                  end
      --                  -- end


      --                  -- return f and df/dX
      --                  return f,gradParameters
      --               end

      -- -- optimize on current mini-batch
      -- if optimMethod == optim.asgd then
      --    _,_,average = optimMethod(feval, parameters, optimState)
      -- else
      --    optimMethod(feval, parameters, optimState)
      -- end
   end

   -- time taken
   time = sys.clock() - time
   time = time / trainData:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      trainLogger:plot()
   end

   -- save/log current net
   local filename = paths.concat(opt.save, 'model.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   print('==> saving model to '..filename)
   if save then
      torch.save(filename, model)
   end

   -- next epoch
   confusion:zero()
   epoch = epoch + 1
end
