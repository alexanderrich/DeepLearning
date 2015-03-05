
if opt.model == 'simple' then
   model = nn.Sequential();  -- make a multi-layer perceptron
   inputs = 27648; outputs = 10; HUs = 200; -- parameters
   model:add(nn.Reshape(inputs))
   model:add(nn.Linear(inputs, HUs))
   model:add(nn.Tanh())
   model:add(nn.Linear(HUs, outputs))
   model:add(nn.LogSoftMax())

else
   model = nn.Sequential()

   --model:add(nn.SpatialZeroPadding(2,2,2,2))
   model:add(nn.SpatialConvolution(3,23,7,7,2,2)) -- 23 channels, 7x7 filters, stride (step) 2
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3,3,2,2)) -- stride 2
   model:add(nn.Dropout())

   -- fully connected layer (50 units)
   model:add(nn.View(22*22*23))
   model:add(nn.Linear(22*22*23,50))
   model:add(nn.Tanh())
   model:add(nn.Linear(50,10))

   model:add(nn.LogSoftMax())
end

criterion = nn.ClassNLLCriterion()

print '==> here is the model:'
print(model)
