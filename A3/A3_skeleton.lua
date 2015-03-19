-------------------------------------------------------------------------
-- In this part of the assignment you will become more familiar with the
-- internal structure of torch modules and the torch documentation.
-- You must complete the definitions of updateOutput and updateGradInput
-- for a 1-d log-exponential pooling module as explained in the handout.
-- 
-- Refer to the torch.nn documentation of nn.TemporalMaxPooling for an
-- explanation of parameters kW and dW.
-- 
-- Refer to the torch.nn documentation overview for explanations of the 
-- structure of nn.Modules and what should be returned in self.output 
-- and self.gradInput.
-- 
-- Don't worry about trying to write code that runs on the GPU.
--
-- Your submission should run on Mercer and contain: 
-- a completed TEAMNAME_A3_skeleton.lua,
--
-- a script TEAMNAME_A3_baseline.lua that is just the provided A3_baseline.lua modified
-- to use your TemporalLogExpPooling module instead of nn.TemporalMaxPooling,
--
-- a saved trained model from TEAMNAME_A3_baseline.lua for which you have done some basic
-- hyperparameter tuning on the training data,
-- 
-- and a script TEAMNAME_A3_gradientcheck.lua that takes as input from stdin:
-- a float epsilon, an integer N, N strings, and N labels (integers 1-5)
-- and prints to stdout the ratios |(FD_epsilon_ijk - exact_ijk) / exact_ijk|
-- where exact_ijk is the backpropagated gradient for weight ijk for the given input
-- and FD_epsilon_ijk is the second-order finite difference of order epsilon
-- of weight ijk for the given input.
------------------------------------------------------------------------

local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
   -----------------------------------------------
   -- your code here
   -----------------------------------------------
   return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
