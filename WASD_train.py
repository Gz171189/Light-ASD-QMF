"""Train Light-ASD / local QMF on WASD train and evaluate on WASD val.

This is a separate entry point: train.py remains AVA-only. The validation set
is used only for evaluation/model selection, never for gradients or adaptation.
"""

import argparse
from collections import defaultdict
import glob
import math
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np
import torch

from ASD import ASD
from dataLoader import load_audio, load_label
from WASD_test import (
    WASDValLoader,
    check_evaluator,
    evaluate_wasd,
    infer,
    load_model,
    load_track_visual,
    read_annotations,
    require_path,
    save_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train on WASD train; evaluate/select only on WASD val',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataPathWASD', required=True, help='WASD dataset root')
    parser.add_argument('--savePath', default='exps/wasd_train', help='Independent output directory')
    parser.add_argument('--wasdEvalDir', required=True,
                        help='Directory containing official WASD evaluator files')
    parser.add_argument('--pretrainModel', default='',
                        help='Optional initialization weights; omit to train from scratch')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrDecay', type=float, default=0.95)
    parser.add_argument('--maxEpoch', type=int, default=30)
    parser.add_argument('--testInterval', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=2000,
                        help='Maximum frames per dynamic mini-batch')
    parser.add_argument('--nDataLoaderThread', type=int, default=64,
                        help='DataLoader workers')
    parser.add_argument('--fusionMode', default='qmf_sync_rank',
                        choices=['sum', 'qmf', 'qmf_sync', 'qmf_sync_rank',
                                 'qmf_anchor'])
    parser.add_argument('--reliabilityHiddenDim', type=int, default=32)
    parser.add_argument('--reliabilityDropout', type=float, default=0.1)
    parser.add_argument('--minReliability', type=float, default=0.1)
    parser.add_argument('--lambdaSync', type=float, default=0.1)
    parser.add_argument('--lambdaRank', type=float, default=0.1)
    parser.add_argument('--lambdaAudioQuality', type=float, default=0.1)
    parser.add_argument('--rankMargin', type=float, default=0.1)
    parser.add_argument('--rankMinLossGap', type=float, default=0.05)
    parser.add_argument('--energyTemperature', type=float, default=1.0)
    parser.add_argument('--fusionTemperature', type=float, default=1.0)
    parser.add_argument('--syncShiftFrames', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.maxEpoch < 1 or args.testInterval < 1 or args.batchSize < 1:
        parser.error('maxEpoch, testInterval, and batchSize must be positive')
    if args.nDataLoaderThread < 0:
        parser.error('nDataLoaderThread must be >= 0')
    if args.lr <= 0 or not 0 < args.lrDecay <= 1:
        parser.error('lr must be positive and lrDecay must be in (0, 1]')
    if not 0 <= args.minReliability < 1:
        parser.error('minReliability must be in [0, 1)')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.energyTemperature, args.fusionTemperature)):
        parser.error('QMF temperatures must be finite and positive')
    if args.lambdaAudioQuality < 0:
        parser.error('lambdaAudioQuality must be non-negative')
    if args.syncShiftFrames < 1:
        parser.error('syncShiftFrames must be positive')
    return args


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def validate_paths(args):
    root = Path(args.dataPathWASD).expanduser().resolve()
    require_path(root, 'WASD directory', True)
    for split in ('train', 'val'):
        require_path(root / 'csv' / (split + '_loader.csv'),
                     'WASD {}_loader.csv'.format(split))
        require_path(root / 'csv' / (split + '_orig.csv'),
                     'WASD {}_orig.csv'.format(split))
        require_path(root / 'clips_audios' / split,
                     'WASD {} audio directory'.format(split), True)
        require_path(root / 'clips_videos' / split,
                     'WASD {} face directory'.format(split), True)
    check_evaluator(Path(args.wasdEvalDir).expanduser().resolve())
    if args.pretrainModel:
        require_path(Path(args.pretrainModel).expanduser().resolve(),
                     'Initialization checkpoint')
    return root


class WASDTrainLoader:
    """Dynamic same-length batches with local Light-ASD augmentations."""

    def __init__(self, tracks, batch_size):
        groups = defaultdict(list)
        for track in tracks:
            groups[int(track['data'][1])].append(track)
        self.mini_batches = []
        for length in sorted(groups, reverse=True):
            tracks_at_length = groups[length]
            tracks_per_batch = max(batch_size // length, 1)
            for start in range(0, len(tracks_at_length), tracks_per_batch):
                self.mini_batches.append(
                    tracks_at_length[start:start + tracks_per_batch])
        if not self.mini_batches:
            raise ValueError('WASD training loader is empty')

    def __len__(self):
        return len(self.mini_batches)

    def __getitem__(self, index):
        from scipy.io import wavfile

        tracks = self.mini_batches[index]
        num_frames = int(tracks[0]['data'][1])
        audio_set = {}
        for track in tracks:
            sample_rate, audio = wavfile.read(str(track['audio_path']))
            if sample_rate != 16000 or audio.ndim != 1 or audio.size == 0:
                raise ValueError('Expected nonempty mono 16 kHz WAV: {}'.format(
                    track['audio_path']))
            audio_set[track['data'][0]] = audio
        audio_features, visual_features, labels, audio_qualities = [], [], [], []
        for track in tracks:
            data = track['data']
            audio_feature, audio_quality = load_audio(
                data, str(track['audio_path'].parent), num_frames,
                audioAug=True, audioSet=audio_set, returnQuality=True)
            audio_features.append(audio_feature)
            audio_qualities.append(audio_quality)
            visual_features.append(load_track_visual(
                track, num_frames, visual_aug=True))
            labels.append(load_label(data, num_frames))
        return (torch.FloatTensor(np.array(audio_features)),
                torch.FloatTensor(np.array(visual_features)),
                torch.LongTensor(np.array(labels)),
                torch.FloatTensor(np.array(audio_qualities)))


def overall_map(log_path):
    text = Path(log_path).read_text(encoding='utf-8')
    matches = re.findall(
        r'^Overall Average Precision:\s*([0-9]+(?:\.[0-9]+)?)\s*$',
        text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError('Could not uniquely parse Overall AP from {}'.format(log_path))
    value = float(matches[0])
    if not 0 <= value <= 1:
        raise ValueError('Official Overall AP is outside [0, 1]: {}'.format(value))
    return value * 100.0


def save_training_checkpoint(model, path, epoch, best_map, loader_state):
    checkpoint = {
        'epoch': epoch,
        'best_mAP': best_map,
        'fusion_mode': getattr(model, 'fusion_mode', 'sum'),
        'state_dict': model.state_dict(),
        'optimizer': model.optim.state_dict(),
        'scheduler': model.scheduler.state_dict(),
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
        'loader_generator_state': loader_state,
    }
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, str(path))


def load_training_checkpoint(model, path, requested_mode):
    checkpoint = torch.load(str(path))
    runtime_mode = getattr(model, 'fusion_mode', 'sum')
    checkpoint_mode = checkpoint.get('fusion_mode', runtime_mode)
    if runtime_mode != requested_mode or checkpoint_mode != requested_mode:
        raise ValueError(
            'Training/runtime/checkpoint fusion modes differ: {}/{}/{}'
            .format(requested_mode, runtime_mode, checkpoint_mode))
    model.load_state_dict(checkpoint['state_dict'])
    model.optim.load_state_dict(checkpoint['optimizer'])
    model.scheduler.load_state_dict(checkpoint['scheduler'])
    if 'python_rng_state' in checkpoint:
        random.setstate(checkpoint['python_rng_state'])
    if 'numpy_rng_state' in checkpoint:
        np.random.set_state(checkpoint['numpy_rng_state'])
    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
    return (checkpoint['epoch'], checkpoint.get('best_mAP'),
            checkpoint.get('loader_generator_state'))


def validate_runtime_mode(model, requested_mode):
    runtime_mode = getattr(model, 'fusion_mode', None)
    if runtime_mode is None:
        if requested_mode != 'sum':
            raise ValueError('Official Light-ASD supports only --fusionMode sum')
    elif runtime_mode != requested_mode:
        raise ValueError('Imported ASD fusion_mode={} does not match requested mode={}'
                         .format(runtime_mode, requested_mode))


def model_for_training(args, model_dir):
    checkpoints = sorted(glob.glob(str(model_dir / 'training_0*.checkpoint')))
    if checkpoints:
        model = ASD(**vars(args))
        validate_runtime_mode(model, args.fusionMode)
        epoch, best_map, loader_state = load_training_checkpoint(
            model, checkpoints[-1], args.fusionMode)
        print('Resuming full training state from {}'.format(checkpoints[-1]))
        return model, epoch + 1, best_map, loader_state
    models = sorted(glob.glob(str(model_dir / 'model_0*.model')))
    if models:
        model = load_model(args, Path(models[-1]))
        model.requires_grad_(True)
        epoch = int(Path(models[-1]).stem.split('_')[-1]) + 1
        print('Resuming model weights from {}; optimizer state is unavailable.'.format(models[-1]))
        return model, epoch, None, None
    if args.pretrainModel:
        # Strictly validates original/QMF architecture, then re-enables training.
        model = load_model(args, Path(args.pretrainModel).expanduser().resolve())
        model.requires_grad_(True)
        print('Initializing WASD training from {}; optimizer starts fresh.'.format(
            args.pretrainModel))
        return model, 1, None, None
    model = ASD(**vars(args))
    validate_runtime_mode(model, args.fusionMode)
    return model, 1, None, None


def main():
    args = parse_args()
    root = validate_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    save_path = Path(args.savePath).expanduser().resolve()
    model_dir = save_path / 'model'
    model_dir.mkdir(parents=True, exist_ok=True)
    train_annotations = read_annotations(root / 'csv/train_orig.csv')
    val_annotations = read_annotations(root / 'csv/val_orig.csv')
    train_tracks = WASDValLoader(root, train_annotations, split='train')
    val_dataset = WASDValLoader(root, val_annotations, split='val')
    train_dataset = WASDTrainLoader(train_tracks.tracks, args.batchSize)
    loader_options = dict(
        batch_size=1, shuffle=True, num_workers=args.nDataLoaderThread,
        pin_memory=True, worker_init_fn=seed_worker, generator=generator)
    if args.nDataLoaderThread:
        loader_options['prefetch_factor'] = 1
    train_loader = torch.utils.data.DataLoader(train_dataset, **loader_options)
    print('WASD train: {} entities / {} frames / {} dynamic batches'.format(
        len(train_tracks), len(train_annotations), len(train_dataset)))
    print('WASD val: {} entities / {} frames'.format(
        len(val_dataset), len(val_annotations)))

    model, epoch, best_map, loader_state = model_for_training(args, model_dir)
    if loader_state is not None:
        generator.set_state(loader_state)
    if epoch > args.maxEpoch:
        print('Latest checkpoint already reached maxEpoch={}; nothing to train.'
              .format(args.maxEpoch))
        return
    score_path = save_path / 'score.txt'
    score_file = score_path.open('a', encoding='utf-8')
    try:
        while epoch <= args.maxEpoch:
            loss, learning_rate = model.train_network(
                epoch=epoch, loader=train_loader, **vars(args))
            if epoch % args.testInterval == 0:
                model_path = model_dir / ('model_{:04d}.model'.format(epoch))
                model.saveParameters(str(model_path))
                scores = infer(model, val_dataset, args.nDataLoaderThread,
                               len(val_annotations))
                prediction_path = save_path / 'val_res.csv'
                save_predictions(val_annotations, scores, prediction_path)
                evaluate_wasd(args.wasdEvalDir, root / 'csv/val_orig.csv',
                              prediction_path)
                current_map = overall_map(save_path / 'wasd_eval.txt')
                previous_best = float('-inf') if best_map is None else best_map
                if current_map > previous_best:
                    best_map = current_map
                    model.saveParameters(str(model_dir / 'best.model'))
                save_training_checkpoint(
                    model,
                    model_dir / 'training_{:04d}.checkpoint'.format(epoch),
                    epoch, best_map, generator.get_state())
                if hasattr(model, 'format_reliability_stats'):
                    train_reliability = model.format_reliability_stats(
                        model.last_train_reliability, 'Train')
                    train_memory = model.format_gpu_memory(
                        model.last_train_gpu_memory, 'Train')
                else:
                    train_reliability = 'Train reliability: unavailable (original sum model)'
                    train_memory = 'Train GPU memory: unavailable (original wrapper)'
                line = ('{} epoch, LR {:.6f}, LOSS {:.6f}, LossSync {:.6f}, '
                        'LossRank {:.6f}, LossAQ {:.6f}, WASD Overall mAP '
                        '{:.2f}%, bestmAP {:.2f}%, {}, {}\n'.format(
                            epoch, learning_rate, loss,
                            model.last_train_sync_loss or 0.0,
                            model.last_train_rank_loss or 0.0,
                            model.last_train_audio_quality_loss or 0.0,
                            current_map, best_map,
                            train_reliability, train_memory))
                print(time.strftime('%Y-%m-%d %H:%M:%S'), line.rstrip())
                score_file.write(line)
                score_file.flush()
            epoch += 1
    finally:
        score_file.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit('WASD training error: {}'.format(error))
