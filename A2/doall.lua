require 'torch'
require 'image'
require 'nn'


cmd = torch.CmdLine()
cmd:text()
cmd:text('STL-10')
cmd:text()
cmd:text('Options:')
cmd:option('-dataSource', 'mat', 'matlab or binary data: mat | bin')
cmd:option('-dataDir', '.', 'directory holding data folders')
cmd:option('-dataAugmentation', 1, 'augment training data set by this factor (1 = no augmentation)')
cmd:option('-type', 'double', 'type: double | float | cuda')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 2, 'number of threads')
cmd:option('-model', 'convnet', 'type of model to construct: convnet | simple')
cmd:option('-size', 'full', 'how many samples do we load: small | full')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-learningRateDecay', 0.002, 'learning rate decay per epoch')
-- add learning decay
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:text()
opt = cmd:parse(arg or {})
-- nb of threads and fixed seed (for repeatable experiments)
if opt.type == 'float' then
   print('==> switching to floats')
   torch.setdefaulttensortype('torch.FloatTensor')
elseif opt.type == 'cuda' then
   print('==> switching to CUDA')
   require 'cunn'
   torch.setdefaulttensortype('torch.FloatTensor')
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)


print '==> executing all'

dofile '1_data.lua'
dofile '2_model.lua'
dofile '3_train.lua'
dofile '4_test.lua'


print '==> training!'
i = 0
while i<5 do
   train()
   test(true)
   i = i+1
end
test(false)
