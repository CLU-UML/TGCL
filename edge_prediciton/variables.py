import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dir_model = "../saved_models"
dir_logs = "../logs"
dir_data = "../data"



