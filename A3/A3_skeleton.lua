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

beta = 1
-- Does the temporal pooling operation
function temporalLogExpPooling(input)
    local n = input:size()
    local parens = input:exp():sum()
    tlep = math.log(parens/n[1])/beta
    return tlep
end

--Computes gradient for above function
function temporalLogExpPoolingGradient(input, gradOutput)
    sumInp = input:apply(expb):sum()
    p_k = input:apply(expb):div(sumInp)
    grad = gradOutput * p_k
    return grad
end

--Subfunction for gradient
function expb(x)
    bx = - beta * x
    return math.exp(bx)
end

function TemporalLogExpPooling:updateOutput(input)
   kW = self.kW
   dW = self.dW
   
   if (input:nDimension()==2) then -- no batches
       nInputFrame = input:size(1)
       inputFrameSize = input:size(2)
       nOutputFrame = math.floor((nInputFrame - kW) / dW + 1)

       output = torch.Tensor(nOutputFrame, inputFrameSize)
       for f = 1, inputFrameSize do -- per feature column
           i = 1
           for j = 1, nOutputFrame do -- per output element
               vec = input[{{},{f}}]
               subset = vec:narrow(1,i,kW)
               output[j][f] = temporalLogExpPooling(subset)
               i = i + dW
           end
       end
   end

   if (input:nDimension()==3) then -- with batches
       minibatchSize = input:size(1)
       nInputFrame = input:size(2)
       inputFrameSize = input:size(3)
       nOutputFrame = math.floor((nInputFrame - kW) / dW + 1)

       output = torch.Tensor(minibatchSize, nOutputFrame, inputFrameSize)
       for b = 1, minibatchSize do -- per batch element
           for f = 1, inputFrameSize do -- per feature column
               i = 1
               for j = 1, nOutputFrame do -- per output element
                   vec = input[b][{{},{f}}]
                   subset = vec:narrow(1,i,kW)
                   output[b][j][f] = temporalLogExpPooling(subset)
                   i = i + dW
               end
           end
       end
   end
    
    self.output = output
    
    return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
    kW = self.kW
    dW = self.dW
    if (input:nDimension()==2) then -- no batches
        nInputFrame = input:size(1)
        inputFrameSize = input:size(2)
        nOutputFrame = gradOutput:size(1)

        gradInput = torch.Tensor(nInputFrame, inputFrameSize)
        gradInput = gradInput:fill(0)
        for f = 1, inputFrameSize do -- per feature column
            i = 1
            vec = input[{{},{f}}]
            subset = vec:narrow(1,i,kW)
            gradI = temporalLogExpPoolingGradient(subset)
            for s = 1, kW do -- per subset element
                ii = i+s-1
                gradInput[ii][f] = gradInput[ii][f] + gradI[s] -- update gradInput
            i = i + dW
            end
        end
    end
    if (input:nDimension()==3) then -- with batches
        minibatchSize = input:size(1)
        nInputFrame = input:size(2)
        inputFrameSize = input:size(3)
        nOutputFrame = gradOutput:size(2)

        gradInput = torch.Tensor(minibatchSize, nInputFrame, inputFrameSize)
        gradInput = gradInput:fill(0)
        for b = 1, minibatchSize do -- per batch element
            for f = 1, inputFrameSize do -- per feature column
                i = 1
                vec = input[b][{{},{f}}]
                subset = vec:narrow(1,i,kW)
                gradI = temporalLogExpPoolingGradient(subset)
                for s = 1, kW do -- per subset element
                    ii = i+s-1
                    gradInput[b][ii][f] = gradInput[b][ii][f] + gradI[s] -- update gradInput
                i = i + dW
                end
            end
        end
    end
    
    self.gradInput = gradInput
    
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
-- End self made module
