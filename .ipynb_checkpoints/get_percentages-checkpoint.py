from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import torch
import argparse
import os

parser = argparse.ArgumentParser(description="Grab statistics")

# Add arguments
parser.add_argument("--model_option", type=int, help="Which model to run")


# Parse arguments
args = parser.parse_args()
model_id = args.model_option



finetuned_model_names = [ "../mario/WizardLM-13B-V1.2", "../mario/WizardMath-13B-V1.0", "../mario/llama-2-13b-code-alpaca"]

my_model = finetuned_model_names[model_id]


if model_id == 0:
    save_path = "LM"
if model_id == 1:
    save_path = "MATH"
if model_id == 2:
    save_path = "CODE"
    
    
print(save_path)

os.makedirs(save_path, exist_ok=True)

# start by loading in the fine-tuned model
finetuned_model = AutoModelForCausalLM.from_pretrained(my_model, device_map="cpu")



# iterate the model layer by layer to create one large tensor of all of the values
model_param_tensor = torch.tensor([]) # will hold all params
num_params = 0
for layer_name, weight_value in finetuned_model.named_parameters():
    
    #count the num elements in the current layer
    num_params += weight_value.numel()
    
    #flatten weight value, concat to tensor holding prev seen weight params (ADDING ABS VALUE)
    model_param_tensor = torch.cat((model_param_tensor, weight_value.flatten().abs()), dim = 0)
    
    


# Grab the top 30% of all positive values from the tensor
top_values, top_indices = model_param_tensor.abs().topk(k = int(num_params*0.3))

# Grab the element the is equivalent to the top 30%. All values above this will not be pruned.
threshold_val = top_values[-1].item()


#define an empty dictionary and populate it with 
percentage_dict = {}
magnitude_dict = {}
range_dict = {}
total_params = {}
mean_dict = {}
#iterate the model again
for layer_name, weight_value in finetuned_model.named_parameters():
    
    # Get the norm
    magnitude_dict[layer_name] = torch.norm(weight_value).item()
    
    # Range of the layer
    range_dict[layer_name] = [torch.min(weight_value).item(), torch.max(weight_value).item()]
    
    #The mean of the layer
    mean_dict[layer_name] = torch.mean(weight_value).item()
    
    # get percentage of each layer that's greater than the threshold: # above threshold/total values
    percentage_dict[layer_name] = (weight_value.abs() > threshold_val).sum().item() / weight_value.numel()
    
    # get the total number of elements
    total_params[layer_name] = weight_value.numel()

    
    
    
torch.save(percentage_dict, os.path.join(save_path, "percentage_dict.pt"))
torch.save(magnitude_dict, os.path.join(save_path, "magnitude_dict.pt"))
torch.save(range_dict, os.path.join(save_path, "range_dict.pt"))
torch.save(total_params, os.path.join(save_path, "total_params.pt"))
torch.save(mean_dict, os.path.join(save_path, "mean_dict.pt"))

print("done")