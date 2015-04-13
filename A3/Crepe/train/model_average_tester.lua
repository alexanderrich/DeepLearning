require 'nn'
require 'torch'
require 'data'

val_data = Data({file = "valdata.t7b",
                 alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 length = 1014,
                 batch_size = 128})

for batch, labels, n in val_data:iterator() do
    current_batch = current_batch or batch:transpose(2,3):contiguous()
    current_labels = current_labels or labels
    current_batch:copy(batch:transpose(2, 3):contiguous())
    current_labels:copy(labels)
    -- Forward propagation
    current_output = current_model:forward(current_batch)
    current_max, current_decision = current_output:double():max(2)
    current_max = current_max:squeeze():double()
    current_decision = current_decision:squeeze():double()
    current_err = torch.ne(current_decision,current_labels:double()):sum()/current_labels:size(1)
    -- Accumulate the errors and losses
    current_e = current_e*(current_n/(current_n+n)) +  current_err*(n/(current_n+n))
    current_n = current_n + n
    print("n: " ..current_n.. ", e: " ..string.format("%.2e", current_e) ", err: " ..string.format("%.2e", current_err))
end
