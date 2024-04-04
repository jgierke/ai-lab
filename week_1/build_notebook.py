import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()

title = new_markdown_cell("# AI lab - week 1")
nb.cells.append(title)

imports_md = new_markdown_cell("## Imports and pytorch configuration")
nb.cells.append(imports_md)

imports_src = new_code_cell("""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# global pytorch configuration
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
#device = torch.device('cpu')
device = torch.device('cuda')

if device == torch.device('cuda') and not torch.cuda.is_available():
  device = torch.device('cpu')
  print("WARNING: No GPU found. Falling back on CPU.")
""")
nb.cells.append(imports_src)

dataloader_md = new_markdown_cell("## Load MNIST dataset")
nb.cells.append(dataloader_md)

dataloader_src = new_code_cell("""
batch_size_train = 64
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(
    '../data/', train=True, download=True,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
  ),
  batch_size=batch_size_train, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(
    '../data/', train=False, download=True,
    transform=torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
  ),
  batch_size=batch_size_test, shuffle=True
)
""")
nb.cells.append(dataloader_src)

model_definition_md = new_markdown_cell("## Define the model")
nb.cells.append(model_definition_md)

model_definition_src = new_code_cell("""
class TwoLayerNet(nn.Module):
  def __init__(self, D_in, H, D_out):
    super(TwoLayerNet, self).__init__()
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(D_in, H)
    self.linear2 = nn.Linear(H, D_out)

  def forward(self, x):
    flattened = self.flatten(x)
    h_relu = F.relu(self.linear1(flattened))
    y = self.linear2(h_relu)
    return F.log_softmax(y)
""")
nb.cells.append(model_definition_src)

model_instantiation_md = new_markdown_cell("## Instantiate the model")
nb.cells.append(model_instantiation_md)

model_instantiation_src = new_code_cell("""
def create_model(dim_in, dim_h, dim_out, lr, momentum, device):
  model = TwoLayerNet(dim_in, dim_h, dim_out)
  model.to(device)
  optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
  return (model, optimizer)

D_in = 28*28
H = 100
D_out = 10

learning_rate = 0.01
momentum = 0.5

network, optimizer = create_model(D_in, H, D_out, learning_rate, momentum, device)
""")
nb.cells.append(model_instantiation_src)

helper_md = new_markdown_cell("## Helper methods")
nb.cells.append(helper_md)

helper_src = new_code_cell("""
def train(model, optimizer, loader, epoch, log_interval, losses, counters, debug=False):
  model.train()
  for batch_idx, (data, target) in enumerate(loader):
    data.to(device)
    target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      if debug:
        print(f\"train epoch: {epoch} [{batch_idx*len(data)} / {len(loader.dataset)}, ({(100.*batch_idx/len(loader)):.0f}%)]\\tloss: {loss.item():.6f}\")
      losses.append(loss.item())
      counters.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test(model, loader, losses):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in loader:
      data.to(device)
      target.to(device)
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(loader.dataset)
  losses.append(test_loss)
  print(f\"\\ntest set: avg. loss: {test_loss:.4f}, accuracy: {correct} / {len(loader.dataset)} ({(100.*correct/len(loader.dataset)):.0f}%)\\n\")
""")
nb.cells.append(helper_src)

train_md = new_markdown_cell("## Train the model")
nb.cells.append(train_md)

train_src = new_code_cell("""
n_epochs = 50
log_interval = 10

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

test(network, test_loader, test_losses)
for epoch in range(1, n_epochs + 1):
  train(network, optimizer, train_loader, epoch, log_interval, train_losses, train_counter)
""")
nb.cells.append(train_src)

visualize_md = new_markdown_cell("## Visualize training")
nb.cells.append(visualize_md)

visualize_src = new_code_cell("""
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
#plt.scatter(test_counter, test_losses, color='red')
#plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
#plt.savefig("plot.png", dpi=300)
fig
""")
nb.cells.append(visualize_src)

evaluate_md = new_markdown_cell("## Evaluate model quality on test dataset")
nb.cells.append(evaluate_md)

evaluate_src = new_code_cell("""
test(network, test_loader, test_losses)
""")
nb.cells.append(evaluate_src)

benchmark_md = new_markdown_cell("## Benchmarking")
nb.cells.append(benchmark_md)

benchmark_src = new_code_cell("""
import torch.utils.benchmark as benchmark

device = torch.device('cpu')
n_threads = torch.get_num_threads()

network_cpu, optimizer_cpu = create_model(D_in, H, D_out, learning_rate, momentum, device)
t_cpu = benchmark.Timer(
  stmt='train(model, optimizer, loader, epoch, interval, losses, counters)',
  setup='from __main__ import train',
  globals={
    'model': network_cpu,
    'optimizer': optimizer_cpu,
    'loader': train_loader,
    'epoch': 1,
    'interval': 10,
    'losses': [],
    'counters': []
  },
    num_threads=n_threads
)
print(t_cpu.timeit(1))

#device = torch.device('cuda')
#n_threads = torch.get_num_threads()
#
#network_cuda, optimizer_cuda = create_model(D_in, H, D_out, learning_rate, momentum, device)
#t_cuda = benchmark.Timer(
#  stmt='train(model, optimizer, loader, epoch, interval, losses, counters)',
#  setup='from __main__ import train',
#  globals={
#    'model': network_cuda,
#    'optimizer': optimizer_cuda,
#    'loader': train_loader,
#    'epoch': 1,
#    'interval': 10,
#    'losses': [],
#    'counters': []
#  },
#    num_threads=n_threads
#)
#print(t_cuda.timeit(1))
""")
nb.cells.append(benchmark_src)

with open("week_1.ipynb", "w") as f:
  nbformat.write(nb, f)
