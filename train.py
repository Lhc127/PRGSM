import time
from options.train_options import TrainOptions
from data.dataprocess import DataProcess
from models.model import create_model
import torchvision
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import os
from PIL import Image
import torch
import random

if __name__ == "__main__":

    opt = TrainOptions().parse()

    # 定义检查点目录
    checkpoint_dir = os.path.join(opt.checkpoints_dir, opt.name).replace('\\', '/')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # define the dataset
    dataset = DataProcess(opt.de_root, opt.input_mask_root, opt.ref_root, opt.isTrain)
    iterator_train = (
        data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers, drop_last=False,
                        pin_memory=True))

    # Calculate total number of batches per epoch
    total_batches = len(iterator_train)

    # Create model
    model = create_model(opt)
    total_steps = 0
    start_epoch = opt.epoch_count

    # ========== 修复：断点续训逻辑 ==========
    # 尝试加载最新的检查点（如果存在且要求继续训练）
    if opt.continue_train:
        if opt.checkpoint_path and os.path.exists(opt.checkpoint_path):
            # 加载指定的检查点
            checkpoint_path = opt.checkpoint_path
            print(f"Loading specified checkpoint: {checkpoint_path}")
        else:
            # 加载最新的检查点
            checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            print(f"Loading latest checkpoint: {checkpoint_path}")

        if os.path.exists(checkpoint_path):
            print(f"Found checkpoint: {checkpoint_path}")
            loaded_epoch, total_steps = model.load_checkpoint(checkpoint_path)
            # 修复：从已完成epoch的下一个开始
            start_epoch = loaded_epoch + 1
            print(
                f"Resuming training from epoch {start_epoch} (completed {loaded_epoch} epochs), total_steps {total_steps}")
        else:
            print("No checkpoint found, starting training from scratch")
            start_epoch = opt.epoch_count
    else:
        print("Starting new training session")
        start_epoch = opt.epoch_count
    # ========== 断点续训逻辑结束 ==========

    # Create the logs
    dir = os.path.join(opt.log_dir, opt.name).replace('\\', '/')
    if not os.path.exists(dir):
        os.mkdir(dir)
    writer = SummaryWriter(log_dir=dir, comment=opt.name)

    # Calculate total epochs (调整计算方式)
    total_epochs = opt.niter + opt.niter_decay - start_epoch + 1

    # 修复：正确计算学习率阶段
    if start_epoch <= opt.niter:
        fixed_epochs = opt.niter - start_epoch + 1
        decay_epochs = opt.niter_decay
    else:
        fixed_epochs = 0
        decay_epochs = (opt.niter + opt.niter_decay) - start_epoch + 1

    # 显示训练信息
    print(f"\n=== Training Information ===")
    print(f"Start epoch: {start_epoch}")
    print(f"Total epochs to train: {total_epochs}")
    print(f"Final epoch target: {opt.niter + opt.niter_decay}")
    print(f"Fixed learning rate epochs: {fixed_epochs}")
    print(f"Decaying learning rate epochs: {decay_epochs}")
    print(f"Total batches per epoch: {total_batches}")
    print(f"===========================\n")

    # 添加学习率监控
    print(f"Initial learning rate: {model.optimizer_model.param_groups[0]['lr']}")

    # Start Training
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        # Initialize batch counter for current epoch
        batch_idx = 0

        for detail, mask, reference in iterator_train:
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            # 修复：正确的batch计数
            batch_idx += 1

            model.set_input(detail, mask, reference)
            model.optimize_parameters()

            # 显示训练进度
            progress = (batch_idx / total_batches) * 100
            print(f'Epoch: {epoch}/{opt.niter + opt.niter_decay} [{batch_idx}/{total_batches}] '
                  f'Progress: {progress:.1f}% | Total Steps: {total_steps}')

            # 每100步监控学习率
            if total_steps % 100 == 0:
                current_lr = model.optimizer_model.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr}")

            # display the training processing
            if total_steps % opt.display_freq == 0:
                input, reference, output, GT = model.get_current_visuals()
                image_out = torch.cat([reference, input, output, GT], 0)
                grid = torchvision.utils.make_grid(image_out)
                writer.add_image('Epoch_(%d)_(%d)' % (epoch, total_steps + 1), grid, total_steps + 1)

            # display the training loss
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                writer.add_scalar('loss_L1', errors['loss_L1'], total_steps + 1)
                writer.add_scalar('Perceptual_loss', errors['Perceptual_loss'], total_steps + 1)
                writer.add_scalar('Style_loss', errors['Style_loss'], total_steps + 1)
                print(f'Iteration time: {t:.4f}s | Epoch: {epoch} | Total Steps: {total_steps}')

            # ========== 定期保存检查点 ==========
            if total_steps % opt.save_checkpoint_freq == 0 and total_steps > 0:
                latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
                model.save_checkpoint(epoch, total_steps, latest_checkpoint_path)
                print(f"Saved latest checkpoint at step {total_steps}")
            # ========== 定期保存检查点结束 ==========

        # 每个epoch结束后的信息
        epoch_time = time.time() - epoch_start_time
        print(f'End of epoch {epoch}/{opt.niter + opt.niter_decay} '
              f'| Time Taken: {epoch_time:.2f} sec '
              f'| Total Steps: {total_steps}')

        if epoch % opt.save_epoch_freq == 0:
            print(f'Saving the model at the end of epoch {epoch}, iters {total_steps}')
            model.save_networks(epoch)
            # ========== 保存epoch检查点 ==========
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            model.save_checkpoint(epoch, total_steps, epoch_checkpoint_path)
            # 同时更新最新检查点
            latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            model.save_checkpoint(epoch, total_steps, latest_checkpoint_path)
            print(f"Saved epoch checkpoint for epoch {epoch}")
            # ========== 保存epoch检查点结束 ==========

        # 显示总体训练进度
        overall_progress = ((epoch - start_epoch + 1) / total_epochs) * 100
        print(f'Overall Training Progress: {overall_progress:.1f}% '
              f'({epoch - start_epoch + 1}/{total_epochs} epochs completed)')

        model.update_learning_rate()

    # 训练完成
    print(f'Training completed! Total epochs: {total_epochs}, Total steps: {total_steps}')
    model.save_networks('latest')
    # ========== 保存最终检查点 ==========
    final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
    model.save_checkpoint(opt.niter + opt.niter_decay, total_steps, final_checkpoint_path)
    print(f"Saved final checkpoint")
    # ========== 保存最终检查点结束 ==========

    writer.close()