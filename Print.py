import torch
import matplotlib.pyplot as plt

fileNum = 4
fileName = f'Data/results_{fileNum}.pth'
results = torch.load(fileName)

print(results['results'])


plt.plot(results['results'])
plt.show()

