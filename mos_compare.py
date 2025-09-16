import pandas as pd
from predict_score import predict_score
import torch
from IQANet import IQANet_DDF_Hyper
from tqdm import tqdm
from time import sleep


if __name__ == '__main__':
    checkpoint_name = 'CSIQ_26_epochs_best'
    dataset_name = 'KADID10K'
    checkpoint_path = f'./checkpoints/{checkpoint_name}.pth.tar'  # Update with your checkpoint path

    # Load the model state
    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    
    # Create IQANet_DDF_Hyper model
    model = IQANet_DDF_Hyper(128, 24, 192, 64).cuda()
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Read the CSV file containing image paths
    df = pd.read_csv(f'/dynamic_iqa_ex/Datasets/{dataset_name}/dmos.csv')  # Update with your CSV file path
    
    with open(f'{checkpoint_name}_{dataset_name}.csv', 'w') as file:
        file.write('image_name,predict_score,mos\n')

    for _, line in tqdm(df.iterrows()):
        image_path = f'./Datasets/{dataset_name}/images/{line["dist_img"]}'
        mos = (line["dmos"] - 1) / 4
        
        score = predict_score(image_path, model)

        with open(f'{checkpoint_name}_{dataset_name}.csv', 'a') as file:
            file.write(f'{image_path},{score},{mos}\n')