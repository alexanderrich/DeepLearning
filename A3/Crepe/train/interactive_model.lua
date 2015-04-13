require 'torch'
require 'nn'
require 'model_average'


-- simple script to give live rating predictions for input review strings
numitems = io.read()

model = Model_Average()

for i = 1, tonumber(numitems) do
   item = io.read()
   stringToTensor(item)
   output = model:forward()
   nll, best = torch.max(output, 1)
   print(best[1])
end


