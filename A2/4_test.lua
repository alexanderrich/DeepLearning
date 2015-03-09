--require 'csvigo'  -- for saving predictions

print '==> defining test procedure'

-- test function
function test(validation)
   -- local vars
   if validation then
      dataset = valData
   else
      dataset = testData
   end
   
   local time = sys.clock()
   predictions = {Prediction={},Id={}}
   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
   model:evaluate()

   -- test over test data
   print('==> testing on test set:')
   datastring = 'Id,Category\n'
   for t = 1,dataset:size() do
      -- disp progress
      --xlua.progress(t, dataset:size())
      
      -- get new sample
      local input = dataset.data[t]
      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() end
      local target = dataset.labels[t]

      -- test sample
      local pred = model:forward(input)
      -- print("\n" .. target .. "\n")
      confusion:add(pred, target)
      predictions.Id[t] = t
      trash_var, predictions.Prediction[t] = torch.max(pred, 1)
      predictions.Prediction[t] = predictions.Prediction[t][1]
      datastring = datastring .. t .. ', ' .. predictions.Prediction[t] .. '\n'
      
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update log/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   if opt.plot then
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      testLogger:plot()
   end

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end
   if not validation then
      file = io.open("results/pred.csv", "w")
      io.output(file)
      io.write(datastring)
      io.close(file)
   end
   --csvigo.save({data=predictions, path = "results/pred.csv"})
   -- next iteration:
   confusion:zero()
end
