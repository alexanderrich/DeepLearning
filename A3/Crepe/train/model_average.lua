require 'nn'
require 'torch'
require 'data'


-- simple model averaging class for making predictions using 2 models
local Model_Average = torch.class("Model_Average")

function Model_Average:__init()
   self.model1 = torch.load("")
   self.model2 = torch.load("")
end

function Model_Average:forward(data)
   self.output1 = self.model1:forward(data)
   self.output2 = self.model2:forward(data)
   -- sum log probabilities to get a measure of the "average" likelihood of the data
   -- no longer gives normalized probablity distribution but works fine for prediction
   return self.output1 + self.output2
end   
