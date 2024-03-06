import time
import argparse

from data_processing import *

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--binary')
parser.add_argument('--destdir')
parser.add_argument('--dataset', choices=['MERL', 'EPFL'], default='MERL')

args = parser.parse_args()


def eval_model(model, dataloader, path_=None, name=''):
    for step, (model_input, gt) in enumerate(dataloader):
        start = time.time()

        model.eval()
        model_input = {key: value.to(device) for key, value in model_input.items()}
        mat_name = dataloader.dataset.fnames[model_input['idx']].split('/')[-1].split('.')[0]
        model_output = model(model_input)
        torch.save(model_output['hypo_params'], args.destdir + mat_name + '.pt')
        end = time.time()
        print(end - start)
    return -1


if args.dataset == 'MERL':
    dataset = MerlDataset(args.binary)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

elif args.dataset == 'EPFL':
    dataset = EPFL(args.binary)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=1)


model = torch.load(args.model, map_location=torch.device('cpu'))
eval_model(model, dataloader)
