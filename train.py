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
        
    if args.checkpoint:
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
        dataset = davis.DAVIS(args.dataset_root_dir, args.dataset_split_name, root_flow = args.dataset_root_dir_flow, resolution = args.dataset_resolution, year = args.dataset_year, dt = args.dataset_dt, filter = filter)
        batch_frontend = models.FlowPreprocessor(resolution = args.resolution, crop = args.crop)
        collate_fn = lambda batch, default_collate = torch.utils.data.dataloader.default_collate: ( [b[0] for b in batch], default_collate([b[1] for b in batch]), [b[2] for b in batch], default_collate([b[3] for b in batch]) )

    return dataset, collate_fn, batch_frontend

def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok = True)
 
    dataset, collate_fn, batch_frontend = build_dataset(args, filter = lambda scene_objects: len(scene_objects) <= 6)
    
    model = build_model(args)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size = args.batch_size, num_workers = args.num_workers, collate_fn = collate_fn, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    loss_scale_consistency = args.gamma_regularization
    loss_scale_entropy = args.gamma_regularization

    start = time.time()
    iteration = 0
    for epoch in range(args.num_epochs):
        model.train()

        total_loss = 0

        breakpoint()
        for i, (frames_paths, frames, flow_paths, flow) in enumerate(data_loader):
            learning_rate = (args.learning_rate * (iteration / args.warmup_steps) if iteration < args.warmup_steps else args.learning_rate) * (args.decay_rate ** int(iteration / args.decay_steps))
            loss_scale_consistency = args.loss_scale_consistency * (args.gamma_regularization ** int(iteration / args.decay_steps))
            loss_scale_entropy = args.loss_scale_entropy * (args.gamma_regularization ** int(iteration / args.decay_steps))
            optimizer.param_groups[0]['lr'] = learning_rate
            
            images = batch_frontend(images.to(args.device))
            recon_combined, recons, masks, slots, attn = model(images)
            loss = models.criterion(recon_combined, masks, images, loss_scale_reconstruction = args.loss_scale_reconstruction, loss_scale_consistency = loss_scale_consistency, loss_scale_entropy = loss_scale_entropy)
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
            torch.save(dict(model_state_dict = model_state_dict), os.path.join(args.checkpoint_dir, args.checkpoint_pattern.format(epoch = epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--batch-size', default=64, type=int, help='Batch size for the model.')
    parser.add_argument('--num-iterations', default=5, type=int, help='Number of attention iterations.')
    parser.add_argument('--num-train-steps', default=300_000, type=int, help='Number of training steps.')
    parser.add_argument('--learning-rate', default=0.0005, type=float, help='Learning rate.')
    parser.add_argument('--warmup-steps', default=200, type=int, help='Number of warmup steps for the learning rate.')
    parser.add_argument('--decay-rate', default=0.5, type=float, help='Rate for the learning rate decay.')
    parser.add_argument('--gamma-regularization', type = float, default = 1.0)
    parser.add_argument('--decay-steps', default=80_000, type=int, help='Number of steps for the learning rate decay.')
    parser.add_argument('--num-epochs', default=10000, type=int, help='number of workers for loading data')

    parser.add_argument('--resolution', type = int, nargs = 2, default = (128, 224))
    parser.add_argument('--num-slots', default=2, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--num-slots', default=2, type=int, help='Number of slots in Slot Attention.')
    parser.add_argument('--hidden-dim', default=64, type=int, help='hidden dimension size')
    parser.add_argument('--loss-scale-reconstruction', type = float, default = 1e+2)
    parser.add_argument('--loss-scale-consistency', type = float, default = 1e-2)
    parser.add_argument('--loss-scale-entropy', type = float, default = 1e-2)

    parser.add_argument('--data-parallel', action = 'store_true') 
    parser.add_argument('--device', default = 'cuda', choices = ['cuda', 'cpu'])
    parser.add_argument('--num-workers', default=16, type=int, help='number of workers for loading data')
    
    parser.add_argument('--checkpoint')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', type=str, help='where to save models' )
    parser.add_argument('--checkpoint-epoch-interval', type = int, default = 10)
    parser.add_argument('--checkpoint-pattern', default = 'ckpt_{epoch:04d}.pt')
    
    parser.add_argument('--dataset', default = 'DAVIS', choices = ['DAVIS'])
    parser.add_argument('--dataset-root-dir', default = 'data/common/DAVIS')
    parser.add_argument('--dataset-root-dir-flow', default = 'data/common/DAVISflow')
    parser.add_argument('--dataset-year', type = int, default = 2016)
    parser.add_argument('--dataset-resolution', default = '480p')
    parser.add_argument('--dataset-split-name', default = 'train', choices = ['train', 'val'])
    parser.add_argument('--dataset-dt', type = int, action = 'append', default = [0, -2, -1, 1, 2])
    args = parser.parse_args()

    main(args)
