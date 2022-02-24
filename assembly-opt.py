
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt


def k_cap(input, cap_size):
    output = np.zeros_like(input)
    if len(input.shape) == 1:
        idx = np.argsort(input)[-cap_size:]
        output[idx] = 1
    else:
        idx = np.argsort(input, axis=-1)[..., -cap_size:]
        np.put_along_axis(output, idx, 1, axis=-1)
    return output

class BrainNet():
    def __init__(self, n_inputs, n_neurons, cap_size, n_rounds, sparsity):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.cap_size = cap_size
        self.n_rounds = n_rounds
        self.sparsity = sparsity
        self.generate_graph()
        self.reset_weights()
        self.plasticity_params = np.zeros((2, 6))
        
    def generate_graph(self):
        self.input_adj = rng.random((n_inputs, n_neurons)) < self.sparsity
        self.rec_adj = np.logical_and(rng.random((n_neurons, n_neurons)) < self.sparsity, 
                                      np.logical_not(np.eye(n_neurons, dtype=bool)))
        
    def reset_weights(self):
        self.input_weights = np.ones((n_inputs, n_neurons))
        self.rec_weights = np.ones((n_neurons, n_neurons))
        
    def forward(self, inputs, update=False):
        self.activations = np.zeros((inputs.shape[0], self.n_rounds, self.n_neurons))
        self.activations[:, 0] = k_cap(inputs @ (self.input_adj * self.input_weights), self.cap_size)
        for i in range(n_rounds-1):
            self.activations[:, i+1] = k_cap(inputs @ (self.input_adj * self.input_weights) + self.activations[:, i] @ (self.rec_adj * self.rec_weights), self.cap_size)
            if update: 
                self.update_weights(i, inputs)
        return self.activations
    
    def update_weights(self, step, inputs):
        input_plasticity_terms = np.zeros((6, self.n_inputs, self.n_neurons))
        rec_plasticity_terms = np.zeros((6, self.n_neurons, self.n_neurons))
        
        input_plasticity_terms[0] = self.input_weights * (inputs[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[1] = self.input_weights * ((1 - inputs)[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[2] = self.input_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[3] = self.input_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[4] = self.input_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        input_plasticity_terms[5] = self.input_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         input_plasticity_terms[6] = self.input_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         input_plasticity_terms[7] = self.input_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)

        rec_plasticity_terms[0] = self.rec_weights * (self.activations[:, step][:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[1] = self.rec_weights * ((1 - self.activations[:, step])[:, :, np.newaxis] * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[2] = self.rec_weights * ((inputs[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[3] = self.rec_weights * (((1 - inputs)[:, :, np.newaxis] * (self.input_adj * self.input_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[4] = self.rec_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        rec_plasticity_terms[5] = self.rec_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=1, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         rec_plasticity_terms[6] = self.rec_weights * ((self.activations[:, step][:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
#         rec_plasticity_terms[7] = self.rec_weights * (((1 - self.activations[:, step])[:, :, np.newaxis] * (self.rec_adj * self.rec_weights)[np.newaxis, :, :]).mean(axis=2, keepdims=True) * self.activations[:, step+1][:, np.newaxis, :]).mean(axis=0)
        
        self.input_weights += (self.plasticity_params[0][:, np.newaxis, np.newaxis] * input_plasticity_terms).sum(axis=0)
        self.rec_weights += (self.plasticity_params[1][:, np.newaxis, np.newaxis] * rec_plasticity_terms).sum(axis=0)
        
        density = np.sum(self.activations[:, step] * self.input_adj * self.activations[:, step].T)

# %%
n_inputs = 500
n_neurons = 500
cap_size = 10
n_rounds = 10
sparsity = 0.05
net = BrainNet(n_inputs, n_neurons, cap_size, n_rounds, sparsity)
inputs = np.zeros((1, n_inputs))
inputs[0, :cap_size] = 1
net.plasticity_params = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]) * 5e-2

# %%
n_iter = 2000
step_size = 1e-3
obj = np.zeros(n_iter)
w_obj = np.zeros(n_iter)
obj_r = np.zeros(n_iter)
w_obj_r = np.zeros(n_iter)


net.plasticity_params = np.zeros((2, 6))
net.plasticity_params = np.array([[1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]) * 1e-2
# print(net.plasticity_params)

net.reset_weights()
outputs = net.forward(inputs, update=True)[:, -1]
obj[0] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
w_obj[0] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean()
obj_r[0] = obj[0]
w_obj_r[0] = w_obj[0]



for i in range(1, n_iter):
    net.reset_weights()
#     direction = rng.standard_normal((2, 6))
#     direction /= np.linalg.norm(direction)
    direction = np.zeros((2, 6))
    direction[rng.integers(2), rng.integers(6)] = rng.integers(2) * 2 - 1
    breakpoint()
    old_params = net.plasticity_params.copy()
    net.plasticity_params += direction * 1e-3
    breakpoint()
    outputs = net.forward(inputs, update=True)[:, -1]
    obj[i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
    w_obj[i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean()

    densities.append(i)
    breakpoint()
    if obj[i] < obj[i-1]:
        net.plasticity_params = old_params.copy()
    obj_r[i] = max(obj_r[i-1], obj[i])

# %%
net.plasticity_params

# %%
net.plasticity_params

# %%
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(obj)
axes[0].plot(obj_r)
axes[1].plot(w_obj)
axes[1].plot(np.maximum.accumulate(w_obj))

# %%
net.reset_weights()
outputs = net.forward(inputs, update=True)
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean())
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean())
idx = outputs[0, -1].argsort(axis=-1)

fig, axes = plt.subplots(1, n_rounds, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(outputs[:, i][:, idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()

# %%
net.reset_weights()
net.generate_graph()
outputs = net.forward(inputs, update=True)
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean())
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean())
idx = outputs[0, -1].argsort(axis=-1)

fig, axes = plt.subplots(1, n_rounds, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(outputs[:, i][:, idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()

# %%
n_iter = 100
n_nets = 5
step_size = 1e-3
obj = np.zeros((n_iter, n_nets))
w_obj = np.zeros((n_iter, n_nets))
obj_p = np.zeros((n_iter, n_nets))
w_obj_p = np.zeros((n_iter, n_nets))


overlap_penalty = 5e-1

nets = [BrainNet(n_inputs, n_neurons, cap_size, n_rounds, sparsity) for i in range(n_nets)]
inputs = np.zeros((2, n_inputs))
inputs[0, :cap_size] = 1
inputs[1, cap_size:2*cap_size] = 1

for i, net in enumerate(nets):
    net.plasticity_params = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]) * 1e-2
    net.reset_weights()
    outputs = net.forward(inputs, update=True)[:, -1]

    
    obj[0, i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
    w_obj[0, i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum()

    obj_p[0, i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean() \
            - overlap_penalty * (outputs[:, :, np.newaxis] * outputs[::-1, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
    w_obj_p[0, i] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum() \
              - overlap_penalty * (outputs[:, :, np.newaxis] * outputs[::-1, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum()

# old_params = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]]) * 1e-2
old_params = np.zeros((2, 8))
for i in range(1, n_iter):
#     direction = rng.standard_normal((2, 6))
#     direction /= np.linalg.norm(direction)
    direction = np.zeros((2, 8))
#     direction[rng.integers(2), rng.integers(8)] = rng.integers(2) * 2 - 1
    direction[:, 0] = (rng.random() > 0.5) * 2 - 1
    new_params = old_params + direction * 1e-3
    
    for j, net in enumerate(nets):
        net.reset_weights()
        net.plasticity_params = new_params.copy()

        outputs = net.forward(inputs, update=True)[:, -1]
        obj[i, j] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
        w_obj[i, j] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum()
        
        obj_p[i, j] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean() \
                - overlap_penalty * (outputs[:, :, np.newaxis] * outputs[::-1, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean()
        w_obj_p[i, j] = (outputs[:, :, np.newaxis] * outputs[:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum() \
                  - overlap_penalty * (outputs[:, :, np.newaxis] * outputs[::-1, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean() / (net.rec_adj * net.rec_weights).sum()

    if w_obj_p[i].mean() > w_obj_p[i-1].mean():
        old_params = new_params.copy()

# %%
net.plasticity_params

# %%
net.plasticity_params

# %%
net.plasticity_params

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes[0, 0].plot(obj.mean(axis=-1))
axes[0, 1].plot(w_obj.mean(axis=-1))
axes[1, 0].plot(obj_p.mean(axis=-1))
axes[1, 1].plot(w_obj_p.mean(axis=-1))

# %%
net.reset_weights()
outputs = net.forward(inputs, update=True)
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean())
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean())
idx = outputs[0, -1].argsort(axis=-1)

fig, axes = plt.subplots(2, n_rounds, figsize=(10, 4))
for i, ax in enumerate(axes[0]):
    ax.imshow(outputs[0, i][idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()
for i, ax in enumerate(axes[1]):
    ax.imshow(outputs[1, i][idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()

# %%
net.reset_weights()
net.generate_graph()
outputs = net.forward(inputs, update=True)
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * net.rec_adj[np.newaxis, :, :]).sum(axis=(1,2)).mean())
print((outputs[:, -1][:, :, np.newaxis] * outputs[:, -1][:, np.newaxis, :] * (net.rec_adj * net.rec_weights)[np.newaxis, :, :]).sum(axis=(1,2)).mean())
idx = outputs[0, -1].argsort(axis=-1)

fig, axes = plt.subplots(2, n_rounds, figsize=(10, 4))
for i, ax in enumerate(axes[0]):
    ax.imshow(outputs[0, i][idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()
for i, ax in enumerate(axes[1]):
    ax.imshow(outputs[1, i][idx] * np.ones((n_neurons, 1)))
    ax.set_axis_off()