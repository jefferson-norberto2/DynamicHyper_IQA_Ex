import pandas as pd
from predict_score import predict_score
import torch
from IQANet import IQANet_DDF_Hyper
from tqdm import tqdm


if __name__ == '__main__':
    checkpoint_name = 'koniq_sigmoid_70_epochs_best'
    dataset_name = 'HRIQ'
    checkpoint_path = f'./checkpoints/{checkpoint_name}.pth.tar'  # Update with your checkpoint path

    state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
    
    model = IQANet_DDF_Hyper(128, 24, 192, 64).cuda()
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    df = pd.read_csv(f'/dynamic_iqa_ex/Datasets/{dataset_name}/hiq_mos_file.csv') 
    
    with open(f'{checkpoint_name}_{dataset_name}.csv', 'w') as file:
        file.write('image_name,predict_score,mos\n')

    for _, line in tqdm(df.iterrows()):
        image_path = f'./Datasets/{dataset_name}/512x384/{line["image_name"]}'
        mos = (line["mos"] - 1) / 4
        
        score = predict_score(image_path, model)

        with open(f'{checkpoint_name}_{dataset_name}.csv', 'a') as file:
            file.write(f'{image_path},{score},{mos}\n')

