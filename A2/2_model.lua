
if opt.model == 'simple' then
   model = nn.Sequential();  -- make a multi-layer perceptron
   inputs = 27648; outputs = 10; HUs = 200; -- parameters
   model:add(nn.Reshape(inputs))
   model:add(nn.Linear(inputs, HUs))
   model:add(nn.Tanh())
   model:add(nn.Linear(HUs, outputs))
   model:add(nn.LogSoftMax())

   else
      if opt.type == 'cuda' then
         print 'error'
      else
         model = nn.Sequential()

         -- model:add(nn.Reshape(27648))
         -- model:add(nn.Reshape(3,96,96,false)) -- delete 'false' when working with batches
         -- model:add(nn.SpatialZeroPadding(2,2,2,2))
         model:add(nn.SpatialConvolutionMM(3,23,7,7,2,2)) -- 23 channels, 7x7 filters, stride (step) 2
         model:add(nn.ReLU())
         model:add(nn.SpatialMaxPooling(3,3,2,2)) -- stride 2
         model:add(nn.Dropout())

         -- fully connected layer (50 units)
         model:add(nn.Reshape(23*22*22)) -- delete 'false' when working with batches
         model:add(nn.Linear(23*22*22,50))
         model:add(nn.Tanh())
         model:add(nn.Linear(50,10))

         model:add(nn.LogSoftMax())
      end
end

criterion = nn.ClassNLLCriterion()

print '==> here is the model:'
print(model)
