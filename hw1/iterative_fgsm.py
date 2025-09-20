import torch
import torch.nn as nn

# fix seed
torch.manual_seed(13)

# define network
N = nn.Sequential(
    nn.Linear(10, 10, bias=False),
    nn.ReLU(),
    nn.Linear(10, 10, bias=False),
    nn.ReLU(),
    nn.Linear(10, 3, bias=False)
)

# random input
x = torch.rand((1, 10))
x.requires_grad_()

t = 1  # target class
original_class = N(x).argmax(dim=1).item()
print("Original Class:", original_class)
assert original_class == 2

# Search grid for eps and alpha
epsilons = [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
alphas = [0.01, 0.05, 0.1, 0.2, 0.3]
num_iters = 100
loss_fn = nn.CrossEntropyLoss()

results = []

for epsReal in epsilons:
    eps = epsReal - 1e-7
    for alpha in alphas:
        x_adv = x.clone().detach()
        for i in range(num_iters):
            x_adv.requires_grad_(True)

            # Forward pass
            outputs = N(x_adv)
            loss = loss_fn(outputs, torch.tensor([t], dtype=torch.long))
            
            # Backward pass
            loss.backward()
            
            # Compute perturbation (targeted FGSM -> subtract gradient)
            print(x_adv.grad)
            eta = alpha * torch.sign(x_adv.grad)
            with torch.no_grad():
                x_adv = x_adv - eta

        new_class = N(x_adv).argmax(dim=1).item()
        # Measure actual L∞ perturbation size
        final_perturbation = torch.norm((x - x_adv), p=float('inf')).item()
        results.append((epsReal, alpha, new_class, final_perturbation))

# Print results in a nice table
print("\nResults:")
print("epsReal | alpha | final_class | L∞_perturb | success?")
for epsReal, alpha, new_class, final_perturbation in results:
    print(f"{epsReal:6.2f} | {alpha:5.2f} | {new_class:11d} | {final_perturbation:11.4f} | {new_class == t}")
