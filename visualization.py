import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description="create visualizations")

# Add arguments
parser.add_argument("--model_option", type=str, help="Which model to run")


# Parse arguments
args = parser.parse_args()
model_id = "../" + args.model_option

print(model_id)

percentage_dict = torch.load(os.path.join(model_id, "percentage_dict.pt"))
totals = torch.load(os.path.join(model_id, "total_params.pt"))

count = 0 

# percentage dict stores all params and the percent of values that are NOT masked
layer_dict = {} # for all attention/mlp layers store the layer num and their respective mask params

#layer_dict ends up being {layer1: [%, %...%]} an array for each layer
for key in percentage_dict.keys():
    
    # we skip the layernorm values
    if "layernorm" in key:
        continue
    
    #get the current layer number - if length > 2 and contains a number, then add to dict
    if len(key.split(".")) < 3: continue
    
    if key.split(".")[2].isdigit():
        layer = "layer_" + key.split(".")[2]

    else:
        continue
            
    
        
    #If the layer # isn't in the dict, add it, otherwise, addend it the existing dict
    if layer in layer_dict:
        layer_dict[layer].append(percentage_dict[key])
        
    else:
        layer_dict[layer] = [percentage_dict[key]]

        
        
### now plot the model


# stack the layers together
count = 0
for key in layer_dict.keys():
    
    if len(layer_dict[key]) != 7:
        continue
    
    if count == 0:
        data = layer_dict[key]
        count += 1
    else:
        data = np.vstack( (data, layer_dict[key]))

        
# better format for visualization    
data = data.T



col_labels = [i for i in layer_dict.keys()]
row_labels = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

fig, ax = plt.subplots(figsize=(30, 90))

# Plot the heatmap
im = ax.imshow(data, cmap='bwr')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)

# Rotate the tick labels and set their alignment
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Create colorbar
cbar = ax.figure.colorbar(im, ax=ax, fraction=0.01, pad=0.04)
cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

# Loop over data dimensions and create text annotations
for i in range(len(row_labels)):
    for j in range(len(col_labels)):
        text = ax.text(j, i, f"{data[i, j]:.3f}",
                       ha="center", va="center", color="w")

ax.set_title(f"Heatmap of {model_id} model (# of params above the 70% threshold)")
fig.tight_layout(pad = 2.0)
plt.savefig(os.path.join(model_id, str(model_id[3:]) + "_heatmap.png"), bbox_inches="tight")



#################################################
  ####### Now plot things layer by layer #######
#################################################
attention_params = 26214400 
mlp_params = 70778880

# 4 atten params, 3 mlp params per layer
total_params = 4*attention_params + 3*mlp_params


layer_aggregated_dict = {}

for key in layer_dict.keys():
    
    params_in_layer = 0
    for i in range(len(layer_dict[key])):
        if i <4:
            params = attention_params * layer_dict[key][i]
        else:
            params = mlp_params * layer_dict[key][i]
        params_in_layer += params
    
    layer_aggregated_dict[key] = params_in_layer / total_params
    
    

    
# Convert the dictionary to a NumPy array
data = list(layer_aggregated_dict.values())
data = np.array(data)

# Reshape the array to a 2D array for plotting
data = data.reshape(1, -1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 1))

# Plot the heatmap
img = ax.imshow(data, cmap='bwr', aspect='auto')

# Add a colorbar
cbar = fig.colorbar(img, ax=ax)

# Set the x-ticks to the dictionary keys
ax.set_xticks(np.arange(len(layer_aggregated_dict)))
ax.set_xticklabels(list(layer_aggregated_dict.keys()), rotation=90)

# Remove the y-ticks
ax.set_yticks([])

# Set the title and adjust the layout
ax.set_title('Heatmap of percentage weights, layer by layer (# of params above the 70% threshold)')
plt.tight_layout()

# Show the plot
plt.savefig(os.path.join(model_id, str(model_id[3:]) + "_layer_heatmap.png"), bbox_inches="tight")