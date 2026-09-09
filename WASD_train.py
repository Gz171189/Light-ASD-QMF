"""Train Light-ASD / local QMF on WASD train and evaluate on WASD val.

This is a separate entry point: train.py remains AVA-only. The validation set
is used only for evaluation/model selection, never for gradients or adaptation.
"""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
import tempfile

import numpy as np
import torch

from ASD import ASD
from dataLoader import load_audio, load_label
from utils.checkpoint_config import (
    MODEL_CONFIG_FIELDS, load_checkpoint_payload, model_config_kwargs,
    resolve_checkpoint_config,
)
from WASD_test import (
    WASDValLoader,
    check_evaluator,
    evaluate_wasd,
    format_wasd_percentage,
    infer,
    load_model,
    load_track_visual,
    read_annotations,
    require_path,
    save_predictions,
)


TRAIN_DEFAULTS = dict(lr=0.001, lrDecay=0.95, lambdaSync=0.1, lambdaRank=0.1,
                      rankMargin=0.1, rankMinLossGap=0.05, batchSize=2000,
                      nDataLoaderThread=64, seed=0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Train on WASD train; evaluate/select only on WASD val',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--dataPathWASD', required=True, help='WASD dataset root')
    parser.add_argument('--savePath', required=True, help='New output directory, also for resume')
    parser.add_argument('--wasdEvalDir', required=True,
                        help='Directory containing official WASD evaluator files')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--resume', default='', help='Full checkpoint to resume at completed epoch + 1')
    source.add_argument('--pretrainModel', default='',
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
                        choices=['sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'])
    parser.add_argument('--reliabilityHiddenDim', type=int, default=32)
    parser.add_argument('--reliabilityDropout', type=float, default=0.1)
    parser.add_argument('--minReliability', type=float, default=0.1)
    parser.add_argument('--lambdaSync', type=float, default=0.1)
    parser.add_argument('--lambdaRank', type=float, default=0.1)
    parser.add_argument('--rankMargin', type=float, default=0.1)
    parser.add_argument('--rankMinLossGap', type=float, default=0.05)
    parser.add_argument('--energyTemperature', type=float, default=1.0)
    parser.add_argument('--fusionTemperature', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    # None means omitted, allowing saved config to take precedence on resume.
    parser.set_defaults(**{cli: None for cli, _ in MODEL_CONFIG_FIELDS.values()},
                        **{key: None for key in TRAIN_DEFAULTS})
    args = parser.parse_args(argv)
    return args


def resolve_training_args(args):
    source = args.resume or args.pretrainModel
    payload = load_checkpoint_payload(Path(source).expanduser().resolve()) if source else None
    if payload is not None:
        _, config = resolve_checkpoint_config(payload, vars(args))
        vars(args).update(model_config_kwargs(config))
    else:
        for key, (cli, default) in MODEL_CONFIG_FIELDS.items():
            if getattr(args, cli) is None:
                setattr(args, cli, 'qmf_sync_rank' if cli == 'fusionMode' else default)
    saved = payload.get('wasd_train_config', {}) if args.resume else {}
    if not isinstance(saved, dict):
        raise ValueError('Invalid wasd_train_config')
    for key, default in TRAIN_DEFAULTS.items():
        requested = getattr(args, key)
        if key in saved and requested is not None and requested != saved[key]:
            raise ValueError('Resume {}={} conflicts with saved {}; use initialization for a new experiment'
                             .format(key, requested, saved[key]))
        setattr(args, key, saved.get(key, default) if requested is None else requested)
    if args.resume and not saved:
        print('Legacy checkpoint has no WASD training settings; supply original loss/batch/worker/seed '
              'settings for faithful continuation. Otherwise original defaults apply.')
    if args.maxEpoch < 1 or args.testInterval < 1 or args.batchSize < 1:
        raise ValueError('maxEpoch, testInterval, and batchSize must be positive')
    if args.nDataLoaderThread < 0:
        raise ValueError('nDataLoaderThread must be >= 0')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.lrDecay <= 1:
        raise ValueError('lr must be positive and lrDecay must be in (0, 1]')
    if not 0 <= args.minReliability < 1:
        raise ValueError('minReliability must be in [0, 1)')
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.energyTemperature, args.fusionTemperature)):
        raise ValueError('QMF temperatures must be finite and positive')
    for key in ('lambdaSync', 'lambdaRank', 'rankMargin', 'rankMinLossGap'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            raise ValueError('{} must be finite and nonnegative'.format(key))
    return payload


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
    for source in (args.pretrainModel, args.resume):
        if source:
            require_path(Path(source).expanduser().resolve(), 'Checkpoint')
    return root


class WASDTrainLoader:
    """Dynamic same-length batches with original Light-ASD augmentations.

    Items: [B,4T,13], [B,T,112,112], [B,T]. Outer DataLoader batch_size=1
    adds the singleton consumed by unchanged ASD.train_network's feature[0].
    """

    def __init__(self, tracks, batch_size):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
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
        audio_features, visual_features, labels = [], [], []
        for track in tracks:
            data = track['data']
            audio_feature = load_audio(
                data, str(track['audio_path'].parent), num_frames,
                audioAug=True, audioSet=audio_set)
            audio_features.append(audio_feature)
            visual_features.append(load_track_visual(
                track, num_frames, visual_aug=True))
            labels.append(load_label(data, num_frames))
        return (torch.FloatTensor(np.array(audio_features)),
                torch.FloatTensor(np.array(visual_features)),
                torch.LongTensor(np.array(labels)))


def overall_map(log_path):
    text = Path(log_path).read_text(encoding='utf-8')
    matches = re.findall(
        r'^Overall Average Precision:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)\s*$',
        text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError('Could not uniquely parse Overall AP from {}'.format(log_path))
    value = float(matches[0])
    if not 0 <= value <= 1:
        raise ValueError('Official Overall AP is outside [0, 1]: {}'.format(value))
    return value * 100.0


def save_training_checkpoint(model, path, epoch, best_map, loader_state, args=None):
    """Original checkpoint protocol plus WASD run metadata; stage before replacing."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix='wasd_checkpoint_', dir=str(path.parent)) as temporary:
        staged = Path(temporary) / 'training.checkpoint'
        model.saveCheckpoint(str(staged), epoch, best_map, loader_generator_state=loader_state)
        if args is not None:
            payload = load_checkpoint_payload(staged)
            payload['wasd_train_config'] = {key: getattr(args, key) for key in TRAIN_DEFAULTS}
            torch.save(payload, str(staged))
        os.replace(str(staged), str(path))


def validate_runtime_mode(model, requested_mode):
    runtime_mode = getattr(model, 'fusion_mode', None)
    if runtime_mode is None:
        if requested_mode != 'sum':
            raise ValueError('Official Light-ASD supports only --fusionMode sum')
    elif runtime_mode != requested_mode:
        raise ValueError('Imported ASD fusion_mode={} does not match requested mode={}'
                         .format(runtime_mode, requested_mode))


def model_for_training(args, model_dir=None):
    if args.resume:
        path = Path(args.resume).expanduser().resolve()
        payload = load_checkpoint_payload(path)
        required = {'state_dict', 'optimizer', 'scheduler', 'epoch', 'best_mAP'}
        if not required.issubset(payload):
            raise ValueError('--resume requires a full checkpoint; missing {}'.format(sorted(required - set(payload))))
        if not isinstance(payload['epoch'], int) or payload['epoch'] < 0:
            raise ValueError('Invalid completed epoch in checkpoint')
        if payload['best_mAP'] is not None and (not math.isfinite(payload['best_mAP']) or
                                               not 0 <= payload['best_mAP'] <= 100):
            raise ValueError('Invalid historical best_mAP')
        # Strict names/shapes/config checks precede restoration of optimizer state.
        model = load_model(args, path)
        epoch, best_map, loader_state = model.loadCheckpoint(str(path))
        model.requires_grad_(True)
        print('Restored full state from {}; next epoch {}'.format(path, epoch + 1))
        return model, epoch + 1, best_map, loader_state
    if args.pretrainModel:
        model = load_model(args, Path(args.pretrainModel).expanduser().resolve())
        model.requires_grad_(True)
        print('Initialized compatible weights; optimizer/scheduler start fresh at epoch 1')
        return model, 1, None, None
    model = ASD(**vars(args))
    validate_runtime_mode(model, args.fusionMode)
    return model, 1, None, None


def main():
    args = parse_args()
    root = validate_paths(args)
    resolve_training_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the original ASD wrapper')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    save_path = Path(args.savePath).expanduser().resolve()
    save_path.mkdir(parents=True, exist_ok=False)
    model_dir = save_path / 'model'
    model_dir.mkdir()
    (save_path / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
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
            current_map = None
            model_path = model_dir / ('model_{:04d}.model'.format(epoch))
            model.saveParameters(str(model_path))
            if epoch % args.testInterval == 0 or epoch == args.maxEpoch:
                scores = infer(model, val_dataset, args.nDataLoaderThread,
                               len(val_annotations))
                eval_path = save_path / 'val_{:04d}'.format(epoch)
                eval_path.mkdir()
                prediction_path = eval_path / 'val_res.csv'
                save_predictions(val_annotations, scores, prediction_path)
                evaluate_wasd(args.wasdEvalDir, root / 'csv/val_orig.csv',
                              prediction_path)
                current_map = overall_map(eval_path / 'wasd_eval_raw.txt')
                previous_best = float('-inf') if best_map is None else best_map
                if current_map > previous_best:
                    best_map = current_map
                    save_training_checkpoint(model, model_dir / 'best.checkpoint',
                                             epoch, best_map, generator.get_state(), args)
                if hasattr(model, 'format_reliability_stats'):
                    train_reliability = model.format_reliability_stats(
                        model.last_train_reliability, 'Train')
                    train_memory = model.format_gpu_memory(
                        model.last_train_gpu_memory, 'Train')
                else:
                    train_reliability = 'Train reliability: unavailable (original sum model)'
                    train_memory = 'Train GPU memory: unavailable (original wrapper)'
                line = ('{} epoch, LR {:.6f}, LOSS {:.6f}, LossSync {:.6f}, '
                        'LossRank {:.6f}, WASD Overall mAP '
                        '{}, bestmAP {}, {}, {}, Train VScoreLossCorr={}, {}\n'.format(
                            epoch, learning_rate, loss,
                            model.last_train_sync_loss or 0.0,
                            model.last_train_rank_loss or 0.0,
                            format_wasd_percentage(current_map, scale=1),
                            format_wasd_percentage(best_map, scale=1),
                            train_reliability, train_memory,
                            model.last_train_visual_loss_correlation,
                            model.format_fusion_diagnostics()))
                print(time.strftime('%Y-%m-%d %H:%M:%S'), line.rstrip())
                score_file.write(line)
                score_file.flush()
            else:
                score_file.write('{} epoch, LR {:.6f}, LOSS {:.6f}, LossSync {}, LossRank {}, '
                                 'validation not scheduled\n'.format(
                                     epoch, learning_rate, loss, model.last_train_sync_loss,
                                     model.last_train_rank_loss))
                score_file.flush()
            save_training_checkpoint(model, model_dir / 'training_{:04d}.checkpoint'.format(epoch),
                                     epoch, best_map, generator.get_state(), args)
            epoch += 1
    finally:
        score_file.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit('WASD training error: {}'.format(error))
