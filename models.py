import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

def criterion(recon_combined, masks, images, loss_scale_reconstruction = 1, loss_scale_consistency = 1, loss_scale_entropy = 1, eps = 1e-15):
    loss_reconstruction = F.mse_loss(recon_combined, images)
    
    # TODO: reduce only over spatial extent, and check the paper
    # TODO: entropy over both flow fields?

    masks_t_n1, masks_t_n2 = masks.chunk(2, dim = 0)
    loss_consistency = torch.stack([F.mse_loss(masks_t_n1[:, 1], masks_t_n2[:, 1]), F.mse_loss(masks_t_n1[:, 1], masks_t_n2[:, 0])]).min()

    loss_entropy = - (masks * (masks + eps).log()).mean()

    return loss_scale_reconstruction * loss_reconstruction + loss_scale_consistency * loss_consistency + loss_scale_entropy * loss_entropy

class FlowPreprocessor(nn.Module):
    def __init__(self, rgb_mean = 0.5, rgb_std = 0.5):
        super().__init__()
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        
    def forward(self, img, normalize = True, interpolate_mode = 'bilinear'):
        assert img.is_floating_point()
        img = (img - self.rgb_mean) / self.rgb_std if normalize else img

        img = img.clamp(-1 if normalize else 0, 1)

        return img

class SlotAttention(nn.Module):
    def __init__(self, num_iter, num_slots, input_size, slot_size, mlp_hidden_size, epsilon=1e-8, gain = 1, temperature_factor = 1):
        super().__init__()
        self.temperature_factor = temperature_factor
        self.num_iter = num_iter
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon
        self.input_size = input_size

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots  = nn.LayerNorm(slot_size)
        self.norm_mlp    = nn.LayerNorm(slot_size)

        self.slots_mu        = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, 1, self.slot_size)))
        
        self.project_q = nn.Linear(slot_size, slot_size, bias = False)
        self.project_k = nn.Linear(input_size, slot_size, bias = False)
        self.project_v = nn.Linear(input_size, slot_size, bias = False)
        
        nn.init.xavier_uniform_(self.project_q.weight, gain = gain)
        nn.init.xavier_uniform_(self.project_k.weight, gain = gain)
        nn.init.xavier_uniform_(self.project_v.weight, gain = gain)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(inplace = True),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

    def forward(self, inputs : 'BTC', num_iter = 0, slots : 'BSC' = None) -> '(BSC, BST, BST)':
        inputs = self.project_x(inputs)

        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)
       
        if slots is None:
            slots = self.slots_mu 

        for _ in range(num_iter or self.num_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)
            q *= self.slot_size ** -0.5
            
            attn_logits = torch.bmm(q, k.transpose(-1, -2))

            attn = F.softmax(attn_logits / self.temperature_factor, dim = 1)
            attn_ = attn + self.epsilon

            bincount = attn_.sum(dim = -1, keepdim = True)
            updates = torch.bmm(attn_ / bincount, v)
            
            slots = self.gru(updates.flatten(end_dim = 1), slots_prev.flatten(end_dim = 1)).reshape_as(slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_logits, attn

class MotionGroupingEncoder(nn.Sequential):
    def __init__(self, hidden_dim = 64, kernel_size = 5, padding = 2):
        super().__init__(
            nn.Conv2d(3,          1 * hidden_dim, kernel_size = kernel_size, padding = padding), nn.InstanceNorm2d(1 * hidden_dim), nn.ReLU(inplace = True), # instance norm
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, 2 * hidden_dim, kernel_size = kernel_size, padding = padding), nn.InstanceNorm2d(2 * hidden_dim), nn.ReLU(inplace = True), # instance norm
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, 4 * hidden_dim, kernel_size = kernel_size, padding = padding), nn.InstanceNorm2d(4 * hidden_dim), nn.ReLU(inplace = True), # instance norm
            nn.MaxPool2d(2),
        )

class MotionGroupingDecoder(nn.Sequential):
    def __init__(self, hidden_dim = 64, kernel_size = 5, padding = 2, stride = 2):
        super().__init__(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = padding), nn.InstanceNorm2d(hidden_dim), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = padding), nn.InstanceNorm2d(hidden_dim), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size, stride = stride, padding = padding), nn.InstanceNorm2d(hidden_dim), nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size = kernel_size), nn.InstanceNorm2d(hidden_dim), nn.ReLU(inplace = True),
            nn.ConvTranspose2d(hidden_dim, 4, kernel_size = kernel_size)
        )

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dense = nn.Linear(4, hidden_dim)

    def forward(self, x):
        spatial_shape = x.shape[-3:-1]
        grid = torch.stack(torch.meshgrid(*[torch.linspace(0., 1., r, device = x.device) for r in spatial_shape]), dim = -1)
        grid = torch.cat([grid, 1 - grid], dim = -1)
        return x + self.dense(grid)

class MotionGroupingAutoEncoder(nn.Module):
    def __init__(self, resolution = (128, 128), num_slots = 8, num_iterations = 3, decoder_initial_size = (8, 8), hidden_dim = 64, interpolate_mode = 'bilinear'):
        super().__init__()
        self.interpolate_mode = interpolate_mode
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.decoder_initial_size = decoder_initial_size
        self.hidden_dim = hidden_dim
        
        self.encoder_cnn = MotionGroupingEncoder(hidden_dim = self.hidden_dim)
        self.encoder_pos = SoftPositionEmbed(self.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.slot_attention = SlotAttention(
            num_iter = self.num_iterations,
            num_slots = self.num_slots,
            input_size = self.hidden_dim,
            slot_size = self.hidden_dim,
            mlp_hidden_size = self.hidden_dim)
        
        self.decoder_pos = SoftPositionEmbed(self.hidden_dim)
        self.decoder_cnn = MotionGroupingDecoder(hidden_dim = self.hidden_dim)

    def forward(self, image):
        x = self.encoder_cnn(image).movedim(1, -1)
        x = self.encoder_pos(x)
        x = self.mlp(self.layer_norm(x))
        
        slots, attn_logits, attn = self.slot_attention(x.flatten(start_dim = 1, end_dim = 2))
        x = slots.reshape(-1, 1, 1, slots.shape[-1]).expand(-1, *self.decoder_initial_size, -1)
        x = self.decoder_pos(x)
        x = self.decoder_cnn(x.movedim(-1, 1))
        
        x = F.interpolate(x, image.shape[-2:], mode = self.interpolate_mode)

        recons, masks = x.unflatten(0, (len(image), len(x) // len(image))).split((3, 1), dim = 2)
        masks = masks.softmax(dim = 1)
        recon_combined = (recons * masks).sum(dim = 1)

        return recon_combined, recons, masks, slots, attn.unsqueeze(-2).unflatten(-1, x.shape[-2:])
