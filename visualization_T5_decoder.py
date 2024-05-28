import matplotlib.pyplot as plt
import torch
import argparse
import os
import numpy as np



model_id = "T5"

print(model_id)

percentage_dict = torch.load(os.path.join("T5", "percentage_dict.pt"))
totals = torch.load(os.path.join("T5", "total_params.pt"))

count = 0 

# percentage dict stores all params and the percent of values that are NOT masked
decoder_layer_dict = {} # for all attention/mlp layers store the layer num and their respective mask params

#decoder_layer_dict ends up being {layer1: [%, %...%]} an array for each layer
for key in percentage_dict.keys():
    
    # we skip the layernorm values
    if "layer_norm" in key or "shared.weight" in key or "encoder" in key or "relative_attention" in key:
        continue

    block = "block_" + key.split(".")[2]

    #If the layer # isn't in the dict, add it, otherwise, append it the existing dict
    if block in decoder_layer_dict:
        decoder_layer_dict[block].append(percentage_dict[key])
        
    else:
        decoder_layer_dict[block] = [percentage_dict[key]]


# stack the layers together
count = 0

for key in decoder_layer_dict.keys():

    if count == 0:
        data = decoder_layer_dict[key]
        count += 1
    else:
        data = np.vstack( (data, decoder_layer_dict[key]))

        
# better format for visualization    
data = data.T



col_labels = [i for i in decoder_layer_dict.keys()]
row_labels = ["attn_q", "attn_k", "attn_v", "attn_o", "encdec_q", "encdec_k", "encdec_v", "encdec_o", "Wi", "Wo"]

fig, ax = plt.subplots(figsize=(20, 60))


from matplotlib.colors import LinearSegmentedColormap

# Create a custom colormap going from blue to red without white
colors = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
cmap_name = 'blue_to_red'
n_bins = 100  # Number of bins for the colormap
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


# Plot the heatmap
im = ax.imshow(data, cmap=cm)

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

ax.set_title(f"Heatmap of {model_id} model (# of params above the 80% threshold)")
fig.tight_layout(pad = 2.0)
plt.savefig("T5/T5_decoder_heatmap.png", bbox_inches="tight")



#################################################
  ####### Now plot things layer by layer #######
#################################################
attention_params = 589824
mlp_params = 2359296

# 4 atten params, 2 mlp params per layer
total_params = 8*attention_params + 2*mlp_params


layer_aggregated_dict = {}

for key in decoder_layer_dict.keys():
    
    params_in_layer = 0
    
    for i in range(len(decoder_layer_dict[key])):
        if i <8:
            params = attention_params * decoder_layer_dict[key][i]
        else:
            params = mlp_params * decoder_layer_dict[key][i]
        params_in_layer += params
        
    print( np.mean(decoder_layer_dict[key]))
    
    layer_aggregated_dict[key] = params_in_layer / total_params
    
    

    
# Convert the dictionary to a NumPy array
data = list(layer_aggregated_dict.values())
data = np.array(data)

print(data)

# Reshape the array to a 2D array for plotting
data = data.reshape(1, -1)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 1))

# Plot the heatmap
im = ax.imshow(data, cmap=cm)

# Add a colorbar
cbar = fig.colorbar(im, ax=ax)

# Set the x-ticks to the dictionary keys
ax.set_xticks(np.arange(len(layer_aggregated_dict)))
ax.set_xticklabels(list(layer_aggregated_dict.keys()), rotation=90)


# Remove the y-ticks
ax.set_yticks([])

# Set the title and adjust the layout
ax.set_title('Heatmap of percentage weights, layer by layer (# of params above the 80% threshold)')
plt.tight_layout()

# Show the plot
plt.savefig("T5/T5_decoder_layer_heatmap.png", bbox_inches="tight")