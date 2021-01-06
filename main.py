from trainers import *
import time
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")


def main():
    opts = BaseOptions()
    args = opts.parse()
    logger = Logger(args.save_path)
    opts.print_options(logger)#打印参数
    loader, gallery_loader, probe_loader = \
        get_reid_dataloaders(args.dataset_path, args.img_size,
                             args.crop_size, args.padding, args.batch_size)#根据yaml文件参数加载数据集

    if args.resume:
        trainer, start_epoch = load_checkpoint(args, logger)
    else:
        trainer = ReidTrainer(args, logger, loader)
        start_epoch = 0

    total_epoch = args.epochs

    start_time = time.time()
    epoch_time = AverageMeter()
    #AverageMeter可以记录当前的输出，累加到某个变量之中，然后根据需要可以打印出历史上的平均

    for epoch in range(start_epoch, total_epoch):
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (total_epoch - epoch))#秒数转换成字符串
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        logger.print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}'.format(time_string(), epoch, total_epoch, need_time))

        meters_trn = trainer.train_epoch(loader, epoch)#返回3个loss
        logger.print_log('  **Train**  ' + create_stat_string(meters_trn))#打印3个loss

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

    meters_val = trainer.eval_performance(loader, gallery_loader, probe_loader)#返回rank-k和mAP
    logger.print_log('  **Test**  ' + create_stat_string(meters_val))


if __name__ == '__main__':
    main()
