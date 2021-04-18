import os
import time
import datetime
import argparse
import functools
import torch
import torch.nn as nn

import models
import davis

def build_model(args):
    model = models.SlotAttentionAutoEncoder(resolution = args.resolution, num_slots = args.num_slots, num_iterations = args.num_iterations, hidden_dim = args.hidden_dim).to(args.device)
        
    if args.checkpoint or args.checkpoint_tensorflow:
        model_state_dict = torch.load(args.checkpoint, map_location = 'cpu')['model_state_dict'] if args.checkpoint else rename_and_transpose_tfcheckpoint(torch.load(args.checkpoint_tensorflow, map_location = 'cpu')) 
        model_state_dict = {'.'.join(k.split('.')[1:]) if k.startswith('module.') else k : v for k, v in model_state_dict.items()}
        status = model.load_state_dict(model_state_dict, strict = False)
        assert not status.missing_keys or set(status.missing_keys) == set(['encoder_pos.grid', 'decoder_pos.grid'])
    
    if args.data_parallel:
        model = nn.DataParallel(model)
    model = model.to(args.device).eval()

    return model

def build_dataset(args, filter = None):
    assert os.path.exists(args.dataset_root_dir), f'provided dataset path [{args.dataset_root_dir}] does not exist'
    
    if args.dataset == 'DAVIS':
        dataset = davis.DAVIS(args.dataset_root_dir, args.split_name, root_flow = args.dataset_root_dir_flow, resolution = args.dataset_resolution, year = args.dataset_year, dt = args.dt, filter = filter)
        batch_frontend = None # models.ClevrImagePreprocessor(resolution = args.resolution, crop = args.crop)
        collate_fn = torch.utils.data.dataloader.default_collate

    return dataset, collate_fn, batch_frontend

def main(args):
    os.makedirs(args.model_dir, exist_ok = True)
 
    dataset, collate_fn, batch_frontend = build_dataset(args, filter = lambda scene_objects: len(scene_objects) <= 6)
    
    model = build_model(args)

    criterion = nn.MSELoss()

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, collate_fn = collate_fn, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

    start = time.time()
    iteration = 0
    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0

        for i, (images, extra) in enumerate(data_loader):
            learning_rate = (args.learning_rate * (iteration / args.warmup_steps) if iteration < args.warmup_steps else args.learning_rate) * (args.decay_rate ** (iteration / args.decay_steps))

            optimizer.param_groups[0]['lr'] = learning_rate
            
            images = batch_frontend(images.to(args.device))
            recon_combined, recons, masks, slots, attn = model(images)
            loss = criterion(recon_combined, images)
            loss_item = float(loss)
            total_loss += loss_item

            del recons, masks, slots

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch:', epoch, '#', iteration, '|', i, '/', len(data_loader), 'Loss:', loss_item)
            iteration += 1

        total_loss /= len(data_loader)

        print ('Epoch:', epoch, 'Loss:', total_loss, 'Time:', datetime.timedelta(seconds = time.time() - start))

        if not epoch % args.checkpoint_epoch_interval:
            model_state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(dict(model_state_dict = model_state_dict), os.path.join(args.model_dir, args.checkpoint_pattern.format(epoch = epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='./checkpoints', type=str, help='where to save models' )
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--num-slots', default=7, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num-iterations', default=3, type=int, help='Number of attention iterations.')
    parser.add_argument('--num-train-steps', default=500000, type=int, help='Number of training steps.')
    parser.add_argument('--learning-rate', default=0.0004, type=float, help='Learning rate.')
    parser.add_argument('--warmup-steps', default=10000, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay-rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--decay-steps', default=100000, type=int, help='Number of steps for the learning rate decay.')

    parser.add_argument('--data-parallel', action = 'store_true') 
    parser.add_argument('--num-workers', default=16, type=int, help='number of workers for loading data')
    parser.add_argument('--num-epochs', default=10000, type=int, help='number of workers for loading data')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 128))
    parser.add_argument('--crop', type = int, nargs = 4, default = (29, 221, 64, 256))
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-epoch-interval', type = int, default = 10)
    parser.add_argument('--checkpoint-pattern', default = 'ckpt_{epoch:04d}.pt')
    parser.add_argument('--dataset', default = 'CLEVR', choices = ['CLEVR', 'COCO'])
    parser.add_argument('--dataset-root-dir', default = './CLEVR_v1.0')
    parser.add_argument('--split-name', default = 'train', choices = ['train', 'val'])
    args = parser.parse_args()

    main(args)
