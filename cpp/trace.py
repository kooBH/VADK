import torch
import sys
sys.path.append(os.path.abspath("../src/models"))

from GPV import GPV

model = GPV()
example_input = torch.rand(1, 3, 32, 32)
print('loading pre-trainded model..')
net.load_state_dict(torch.load('save/cifar10_resnet18_epoch1_state.pt'))
print('eval')
net.eval()

print('tracing')
traced_model = torch.jit.trace(net, example_input)
print('saving')
traced_model.save('traced-state2.pt')