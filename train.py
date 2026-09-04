import time, os, torch, argparse, warnings, glob, random, numpy


from dataLoader import train_loader, val_loader
from utils.tools import *
from ASD import ASD


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    # This code is modified based on this [repository](https://github.com/TaoRuijie/TalkNet-ASD).
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Model Training")
    # Training details
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    parser.add_argument('--maxEpoch', type=int, default=30, help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize', type=int, default=2000, help='Dynamic batch size, default is 2000 frames')
    parser.add_argument('--nDataLoaderThread', type=int, default=64, help='Number of loader threads')
    # QMF-inspired frame-wise reliability fusion
    parser.add_argument('--fusionMode', type=str, default='qmf',
                        choices=['sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'],
                        help='sum baseline, M01, M02, or ranked asymmetric M03')
    parser.add_argument('--reliabilityHiddenDim', type=int, default=32,
                        help='Hidden dimension of each frame-wise reliability head')
    parser.add_argument('--reliabilityDropout', type=float, default=0.1,
                        help='Dropout probability in the reliability heads')
    parser.add_argument('--minReliability', type=float, default=0.1,
                        help='Lower bound for each modality reliability')
    parser.add_argument('--lambdaSync', type=float, default=0.1,
                        help='Weight of AV correspondence loss in M02/M03')
    parser.add_argument('--lambdaRank', type=float, default=0.1,
                        help='Weight of visual reliability ranking loss in M03')
    parser.add_argument('--rankMargin', type=float, default=0.1,
                        help='Required visual-score margin for M03 ranking')
    parser.add_argument('--rankMinLossGap', type=float, default=0.05,
                        help='Minimum visual-loss gap used to form M03 pairs')
    parser.add_argument('--energyTemperature', type=float, default=1.0,
                        help='Temperature used by visual energy confidence')
    parser.add_argument('--fusionTemperature', type=float, default=1.0,
                        help='Temperature used to normalize QMF fusion scores')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for model and data loading')
    # Data path
    parser.add_argument('--dataPathAVA', type=str, default="AVADataPath", help='Save path of AVA dataset')
    parser.add_argument('--savePath', type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val",
                        help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA', dest='downloadAVA', action='store_true',
                        help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation', dest='evaluation', action='store_true',
                        help='Only do evaluation by using pretrained model [pretrain_AVA_CVPR.model]')
    parser.add_argument('--pretrainModel', type=str, default='weight/pretrain_AVA_CVPR.model',
                        help='Checkpoint used together with --evaluation')
    args = parser.parse_args()
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loaderGenerator = torch.Generator()
    loaderGenerator.manual_seed(args.seed)
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    loader = train_loader(trialFileName=args.trainTrialAVA, \
                          audioPath=os.path.join(args.audioPathAVA, 'train'), \
                          visualPath=os.path.join(args.visualPathAVA, 'train'), \
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=True, num_workers=args.nDataLoaderThread,
                                              pin_memory=True, worker_init_fn=seed_worker,
                                              generator=loaderGenerator)

    loader = val_loader(trialFileName=args.evalTrialAVA, \
                        audioPath=os.path.join(args.audioPathAVA, args.evalDataType), \
                        visualPath=os.path.join(args.visualPathAVA, args.evalDataType), \
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=64, pin_memory=True,
                                            worker_init_fn=seed_worker,
                                            generator=loaderGenerator)

    if args.evaluation == True:
        s = ASD(**vars(args))
        s.loadParameters(args.pretrainModel)
        print("Model %s loaded from previous state!" % args.pretrainModel)
        mAP = s.evaluate_network(loader=valLoader, **vars(args))
        print("mAP %2.2f%%" % (mAP))
        quit()

    checkpointfiles = glob.glob(
        '%s/training_0*.checkpoint' % args.modelSavePath
    )
    checkpointfiles.sort()
    resumeBestmAP = None
    if len(checkpointfiles) >= 1:
        s = ASD(**vars(args))
        loadedEpoch, resumeBestmAP, loaderState = s.loadCheckpoint(
            checkpointfiles[-1]
        )
        if loaderState is not None:
            loaderGenerator.set_state(loaderState)
        epoch = loadedEpoch + 1
        print(
            "Training checkpoint %s loaded; resume from epoch %d with "
            "optimizer and scheduler state!" % (checkpointfiles[-1], epoch)
        )
    else:
        modelfiles = glob.glob('%s/model_0*.model' % args.modelSavePath)
        modelfiles.sort()
        if len(modelfiles) >= 1:
            print(
                "Model %s loaded from previous state; Adam state is "
                "unavailable." % modelfiles[-1]
            )
            epoch = int(
                os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]
            ) + 1
            s = ASD(epoch=epoch, **vars(args))
            s.loadParameters(modelfiles[-1])
        else:
            epoch = 1
            s = ASD(epoch=epoch, **vars(args))

    mAPs = [] if resumeBestmAP is None else [resumeBestmAP]
    scoreFile = open(args.scoreSavePath, "a+")

    if epoch > args.maxEpoch:
        print(
            "Latest checkpoint already reached maxEpoch=%d; nothing to train."
            % args.maxEpoch
        )
        scoreFile.close()
        return

    while (1):
        loss, lr = s.train_network(epoch=epoch, loader=trainLoader, **vars(args))

        if epoch % args.testInterval == 0:
            s.saveParameters(args.modelSavePath + "/model_%04d.model" % epoch)
            mAPs.append(s.evaluate_network(epoch=epoch, loader=valLoader, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"),
                  "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%" % (epoch, mAPs[-1], max(mAPs)))
            trainReliability = s.format_reliability_stats(
                s.last_train_reliability, 'Train'
            )
            evalReliability = s.format_reliability_stats(
                s.last_eval_reliability, 'Eval'
            )
            trainMemory = s.format_gpu_memory(s.last_train_gpu_memory, 'Train')
            evalMemory = s.format_gpu_memory(s.last_eval_gpu_memory, 'Eval')
            fusionDiagnostics = s.format_fusion_diagnostics()
            s.saveCheckpoint(
                args.modelSavePath + "/training_%04d.checkpoint" % epoch,
                epoch,
                max(mAPs),
                loader_generator_state=loaderGenerator.get_state(),
            )
            scoreFile.write(
                "%d epoch, LR %f, LOSS %f, LossSync %f, LossRank %f, "
                "mAP %2.2f%%, bestmAP %2.2f%%, %s, %s, "
                "Train/Eval VScoreLossCorr=%.6f/%.6f, %s, %s, %s\n"
                % (epoch, lr, loss, s.last_train_sync_loss or 0.0,
                   s.last_train_rank_loss or 0.0, mAPs[-1], max(mAPs),
                   trainReliability, evalReliability,
                   s.last_train_visual_loss_correlation or 0.0,
                   s.last_eval_visual_loss_correlation or 0.0,
                   fusionDiagnostics, trainMemory, evalMemory))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1


if __name__ == '__main__':
    main()
