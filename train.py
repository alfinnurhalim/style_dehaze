import os
import cv2
import numpy as np

from tqdm import tqdm

from torch.utils.data import DataLoader

from model.utils.utils import get_config
from model.trainers.trainer import Trainer
from dataset.dataset_RESIDE import ImagePairDataset
from logger import TrainingLogger

code_name = 'Dataset_RESIDE'
root_dir = f'/content/drive/MyDrive/TESIS/context_sim2real/Dataset/{code_name}'
cfg_path = f'/content/drive/MyDrive/TESIS/context_sim2real/config/{code_name}.yaml'

train_dataset_mixed = ImagePairDataset(root_dir=root_dir, phase='train',steps=30, mix=False)
test_dataset_mixed = ImagePairDataset(root_dir=root_dir, phase='test', steps=10, mix=True)

print(f'\n\nTrain Dataset : {len(train_dataset_mixed)}')
print(f'Test Dataset : {len(test_dataset_mixed)}\n\n')
train_loader_mixed = DataLoader(train_dataset_mixed, batch_size=1, shuffle=True, num_workers=2)
test_loader_mixed = DataLoader(test_dataset_mixed, batch_size=1, shuffle=False, num_workers=2)

args = get_config(cfg_path)
keep_ratio = args['keep_ratio']
resume = arsg['resume']

for ratio in keep_ratio:

    print('Using ratio of', args['keep_ratio'])
    print(args['n_flow'], args['n_block'])

    trainer = Trainer(args)
    if resume is not None:
      trainer.load_model(resume)

    training_logger = TrainingLogger(trainer.log_path)

    for epoch in range(args['total_epoch']):
        progress_bar = tqdm(enumerate(train_loader_mixed),
                            total=len(train_loader_mixed),
                            desc=f"Epoch {epoch}")

        last_loss = None
        for batch_id, (source_image, style_image, gt_image) in progress_bar:
            loss_list = trainer.train(batch_id, source_image, style_image, gt_image, epoch)
            
            loss,loss_c,loss_s,loss_r,loss_smooth = loss_list if loss_list is not None else [0,0,0,0,0]

            last_loss = loss  # Update last_loss continuously
            progress_bar.set_postfix({'Loss': f'{loss:.4f}',
                                      'Loss_c': f'{loss_c:.4f}',
                                      'Loss_s': f'{loss_s:.4f}',
                                      'Loss_r': f'{loss_r:.4f}',
                                      'Loss_smooth': f'{loss_smooth:.4f}'})

        if last_loss is not None:
            training_logger.log_epoch_loss(epoch, last_loss)

        print(f"\nEpoch: {epoch} - Last Batch Loss: {last_loss:.4f}")

        if epoch%10 == 0:
          print('\n\nTesting......')
          for batch_id, (source_image, style_image, gt_image) in enumerate(test_loader_mixed):
              trainer.test(epoch, batch_id, source_image, style_image, gt_image)

        training_logger.update_graph()

        print('\n\n')
