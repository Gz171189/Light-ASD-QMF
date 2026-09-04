"""Inference only: an AVA-trained local ASD checkpoint -> WASD validation.

The existing training/model/loss files are deliberately not modified. Audio
preprocessing comes from dataLoader.load_audio; visual preprocessing follows
dataLoader.load_visual with only the WASD path/basename handling changed.

Format references (checked 2026-08-31):
https://github.com/Tiago-Roxo/WASD/blob/main/create_dataset.py
https://github.com/Tiago-Roxo/WASD/blob/main/eval/WASD_evaluation.py
The latter expects HEADERLESS, ordered 8-column GT / 9-column predictions.
We export a temporary GT view without changing the original val_orig.csv.
Legacy division headings are translated using the mapping explicitly confirmed
by the user, in a temporary copy only. Modern headings use the original file.
Unknown/mixed headings are rejected; no category membership is inferred.
"""

import argparse
import ast
import csv
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile


GT_COLUMNS = [
    'video_id', 'frame_timestamp', 'entity_box_x1', 'entity_box_y1',
    'entity_box_x2', 'entity_box_y2', 'label', 'entity_id',
]
PRED_COLUMNS = GT_COLUMNS + ['score']
CATEGORIES = ('OC', 'SI', 'FO', 'HVN', 'SS')
LEGACY_CATEGORIES = {
    'Interview': 'OC', 'Debate': 'SI', 'Podcast': 'FO',
    'React': 'HVN', 'Police': 'SS',
}
# Byte matching preserves every non-heading byte, including ID order, spaces,
# blank lines, CRLF/LF endings and an optional UTF-8 BOM.
DIVISION_HEADING = re.compile(rb'^([ \t]*#[ \t]*)(.*?)([ \t]*(?:\r\n|\r|\n)?)$')


def parse_args():
    parser = argparse.ArgumentParser(
        description='AVA-trained Light-ASD / QMF -> WASD validation (no training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog='Only use AVA training settings/checkpoint selection; never tune on WASD.',
    )
    parser.add_argument('--dataPathWASD', required=True, help='WASD dataset root')
    parser.add_argument('--pretrainModel', required=True,
                        help='Your AVA-trained .model or training .checkpoint')
    parser.add_argument('--savePath', default='exps/wasd', help='Output directory')
    parser.add_argument('--nDataLoaderThread', type=int, default=64,
                        help='DataLoader workers (0 is also supported)')
    parser.add_argument('--wasdEvalDir', help='Directory with official evaluator and division file')
    # .model files contain tensors, not constructor settings. Infer architecture
    # and hidden width from tensors, but expose the non-state inference settings.
    parser.add_argument('--fusionMode', default='auto',
                        choices=['auto', 'sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'],
                        help='Auto-detect from checkpoint; explicit modes must match')
    parser.add_argument('--minReliability', type=float, default=0.1,
                        help='Must match AVA training; not stored in checkpoint')
    parser.add_argument('--energyTemperature', type=float, default=1.0,
                        help='Must match AVA training; not stored in checkpoint')
    parser.add_argument('--fusionTemperature', type=float, default=1.0,
                        help='Must match AVA training; not stored in checkpoint')
    args = parser.parse_args()
    if args.nDataLoaderThread < 0:
        parser.error('--nDataLoaderThread must be >= 0')
    if not 0 <= args.minReliability < 1:
        parser.error('--minReliability must be in [0, 1)')
    if any(not math.isfinite(x) or x <= 0 for x in
           (args.energyTemperature, args.fusionTemperature)):
        parser.error('QMF temperatures must be finite and positive')
    return args


def require_path(path, description, directory=False):
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError('{} not found: {}'.format(description, path))


def validate_paths(args):
    root = Path(args.dataPathWASD).expanduser().resolve()
    checkpoint = Path(args.pretrainModel).expanduser().resolve()
    require_path(root, 'WASD directory', directory=True)
    require_path(root / 'csv/val_loader.csv', 'WASD val_loader.csv')
    require_path(root / 'csv/val_orig.csv', 'WASD val_orig.csv')
    require_path(root / 'clips_audios/val', 'WASD validation audio directory', True)
    require_path(root / 'clips_videos/val', 'WASD validation face directory', True)
    require_path(checkpoint, 'AVA-trained checkpoint')
    return root, checkpoint


def read_annotations(path):
    import numpy as np
    import pandas as pd

    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        first = next(csv.reader(handle), [])
    options = dict(dtype={'video_id': str, 'entity_id': str}, keep_default_na=False)
    if 'video_id' in first and 'entity_id' in first:
        frame = pd.read_csv(path, **options)
    elif len(first) == len(GT_COLUMNS):
        frame = pd.read_csv(path, header=None, names=GT_COLUMNS, **options)
    else:
        raise ValueError('Unrecognized val_orig.csv format: {}'.format(path))
    missing = set(GT_COLUMNS) - set(frame.columns)
    if missing or frame.empty:
        raise ValueError('Empty/invalid val_orig.csv; missing columns: {}'.format(sorted(missing)))
    for column in GT_COLUMNS[1:6]:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
        if not np.isfinite(frame[column]).all():
            raise ValueError('Non-finite annotation values in {}'.format(column))
    if (frame['frame_timestamp'] < 0).any():
        raise ValueError('Negative frame timestamps in val_orig.csv')
    if frame.duplicated(['entity_id', 'frame_timestamp']).any():
        raise ValueError('Duplicate (entity_id, frame_timestamp) in val_orig.csv')
    valid_labels = {'SPEAKING_AUDIBLE', 'SPEAKING_NOT_AUDIBLE', 'NOT_SPEAKING'}
    if not set(frame['label']).issubset(valid_labels):
        raise ValueError('val_orig.csv label must use official speaking label strings')
    return frame


def resolve_entity_media(parent, entity, suffix='', directory=False):
    """Keep annotation IDs intact; tolerate ':' -> '_' in extracted filenames.

    Native WASD IDs can contain colons. Some archives/extraction tools replace
    them with underscores in media paths (observed on the user's server).
    Prefer the exact name, and only try this specific alias when it is absent.
    """
    candidates = [parent / (entity + suffix)]
    if ':' in entity:
        candidates.append(parent / (entity.replace(':', '_') + suffix))
    for path in candidates:
        if path.is_dir() if directory else path.is_file():
            return path
    raise FileNotFoundError('WASD {} for {!r} not found; checked: {}'.format(
        'face directory' if directory else 'audio', entity,
        ', '.join(str(path) for path in candidates)))


class WASDValLoader:
    """One complete entity per item, as in the local validation loader.

    Use val_orig's entity -> full video_id mapping, never entity_id[:11].
    Track rows are sorted by numeric timestamp and mapped back to original CSV
    row positions, so neither entity order nor annotation row order is assumed.
    """

    def __init__(self, root, annotations):
        import numpy as np

        self.root = Path(root)
        self.tracks = []
        groups = annotations.groupby('entity_id', sort=False).indices
        seen = set()
        media_owners = {}
        trial = self.root / 'csv/val_loader.csv'
        with trial.open(encoding='utf-8-sig') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                data = line.rstrip('\r\n').split('\t')
                try:
                    entity, num_frames, fps = data[0], int(data[1]), float(data[2])
                    labels = ast.literal_eval(data[3])
                except (IndexError, ValueError, SyntaxError) as exc:
                    raise ValueError(
                        '{}:{}: expected tab-separated entity, frames, fps, labels'
                        .format(trial, line_number)
                    ) from exc
                if (num_frames <= 0 or not math.isfinite(fps) or fps <= 0 or
                        not isinstance(labels, list) or len(labels) != num_frames or
                        any(label not in (0, 1) for label in labels)):
                    raise ValueError('Invalid frame count/fps/binary labels for {}'.format(entity))
                if entity in seen or entity not in groups:
                    raise ValueError('Duplicate or unannotated loader entity: {}'.format(entity))
                seen.add(entity)
                rows = groups[entity]
                group = annotations.iloc[rows]
                videos = group['video_id'].unique()
                if len(videos) != 1:
                    raise ValueError('Entity maps to multiple video IDs: {}'.format(entity))
                video = videos[0]
                for identifier in (entity, video):
                    # Colons are legal in WASD IDs and POSIX filenames. Reject
                    # traversal/separators/NUL, not the dataset's ID syntax.
                    if (not identifier or identifier in ('.', '..') or
                            Path(identifier).anchor or any(
                                c in identifier for c in ('/', '\\', '\x00'))):
                        raise ValueError('Invalid WASD path identifier: {!r}'.format(identifier))
                if len(rows) != num_frames:
                    raise ValueError(
                        'Frame count mismatch for {}: loader={}, val_orig={}'
                        .format(entity, num_frames, len(rows))
                    )
                timestamps = group['frame_timestamp'].to_numpy(dtype=float)
                order = np.argsort(timestamps, kind='stable')
                audio_path = resolve_entity_media(
                    self.root / 'clips_audios/val' / video, entity, suffix='.wav')
                face_dir = resolve_entity_media(
                    self.root / 'clips_videos/val' / video, entity, directory=True)
                for media_path in (audio_path, face_dir):
                    resolved = media_path.resolve()
                    if resolved in media_owners:
                        raise ValueError('Ambiguous WASD media path: {} is shared by '
                                         'entities {!r} and {!r}'.format(
                                             media_path, media_owners[resolved], entity))
                    media_owners[resolved] = entity
                self.tracks.append(dict(data=data, rows=rows[order],
                                        timestamps=timestamps[order],
                                        audio_path=audio_path, face_dir=face_dir))
        missing = set(groups) - seen
        total = sum(len(track['rows']) for track in self.tracks)
        if missing or total != len(annotations):
            raise ValueError(
                'WASD coverage mismatch: loader frames={}, annotations={}, '
                'missing entities (first 5)={}'.format(total, len(annotations), sorted(missing)[:5])
            )

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, index):
        # Lazy imports also allow --help on a machine without torch/CUDA/OpenCV.
        import cv2
        import numpy as np
        import torch
        from scipy.io import wavfile
        from dataLoader import load_audio, load_label

        track = self.tracks[index]
        data = track['data']
        num_frames = int(data[1])
        sample_rate, audio = wavfile.read(str(track['audio_path']))
        if sample_rate != 16000 or audio.ndim != 1 or audio.size == 0:
            raise ValueError('Expected nonempty mono 16 kHz WAV: {}'.format(track['audio_path']))
        # Exact local MFCC parameters, fps alignment, wrap padding, and truncation.
        audio_features = load_audio(data, str(track['audio_path'].parent), num_frames,
                                    audioAug=False, audioSet={data[0]: audio})
        try:
            face_files = sorted(track['face_dir'].glob('*.jpg'), key=lambda p: float(p.stem))
        except ValueError as exc:
            raise ValueError('Non-numeric JPG timestamp in {}'.format(track['face_dir'])) from exc
        if len(face_files) < num_frames:
            raise ValueError('Missing face frames for {}: expected {}, found {}'.format(
                data[0], num_frames, len(face_files)))
        face_files = face_files[:num_frames]  # Same truncation as load_visual.
        # create_dataset.py names crops with %.2f timestamps. Check alignment
        # before assigning scores; equal counts alone cannot detect shifted frames.
        actual_times = np.array([float(path.stem) for path in face_files])
        if not np.allclose(actual_times, track['timestamps'], rtol=0, atol=0.005001):
            raise ValueError('Face timestamps do not match val_orig.csv for {}'.format(data[0]))
        faces = []
        for path in face_files:
            face = cv2.imread(str(path))
            if face is None:
                raise ValueError('Unreadable WASD face: {}'.format(path))
            # Exactly local load_visual(..., visualAug=False): BGR -> gray -> 112.
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            faces.append(cv2.resize(face, (112, 112)))
        return (torch.FloatTensor(np.array([audio_features])),
                torch.FloatTensor(np.array([faces])),
                torch.LongTensor(np.array([load_label(data, num_frames)])), index)


def checkpoint_config(state):
    """Detect only architecture information actually present in local tensors."""
    prefix = 'model.reliabilityFusion.'
    if prefix + 'visual_quality_head.0.weight' in state:
        mode, key = 'qmf_sync_rank', 'sync_head.0.weight'
    elif prefix + 'sync_head.0.weight' in state:
        mode, key = 'qmf_sync', 'sync_head.0.weight'
    elif prefix + 'audio_reliability.0.weight' in state:
        mode, key = 'qmf', 'audio_reliability.0.weight'
    elif any(name.startswith(prefix) for name in state):
        raise ValueError('Unrecognized QMF checkpoint architecture')
    else:
        return 'sum', 32
    return mode, int(state[prefix + key].shape[0])


def load_model(args, checkpoint):
    import torch
    from ASD import ASD

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the existing ASD wrapper; run on the GPU server.')
    # Local saveParameters saves a plain state_dict; saveCheckpoint nests it.
    # Use only trusted, user-trained checkpoints, as in the existing loader.
    payload = torch.load(str(checkpoint), map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError('Expected a local ASD state_dict or training checkpoint')
    nested = 'state_dict' in payload
    state = payload['state_dict'] if nested else payload
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint contains no model state_dict')
    state = {name.replace('module.', ''): value for name, value in state.items()}
    mode, hidden_dim = checkpoint_config(state)
    if nested and payload.get('fusion_mode', mode) != mode:
        raise ValueError('Checkpoint fusion_mode metadata disagrees with its tensors')
    if args.fusionMode not in ('auto', mode):
        raise ValueError('Requested fusionMode={} but checkpoint is {}'.format(args.fusionMode, mode))
    model = ASD(fusionMode=mode, reliabilityHiddenDim=hidden_dim,
                minReliability=args.minReliability, energyTemperature=args.energyTemperature,
                fusionTemperature=args.fusionTemperature)
    # Official ASD accepts **kwargs but ignores QMF settings and has no
    # fusion_mode. Never silently run QMF weights through its sum-only network.
    runtime_mode = getattr(model, 'fusion_mode', None)
    if runtime_mode is None and mode != 'sum':
        raise ValueError('A {} checkpoint requires the matching QMF implementation; '
                         'the imported ASD wrapper is original Light-ASD.'.format(mode))
    if runtime_mode is not None and runtime_mode != mode:
        raise ValueError('ASD fusion_mode={} does not match checkpoint mode={}'
                         .format(runtime_mode, mode))
    # loadParameters intentionally tolerates missing/mismatched tensors for
    # training initialization. Evaluation must reject them, not use random heads.
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    wrong = [name for name in set(expected) & set(state)
             if not torch.is_tensor(state[name]) or state[name].shape != expected[name].shape]
    if missing or extra or wrong:
        raise ValueError('Checkpoint incompatible with current ASD: missing={}, extra={}, '
                         'wrong shapes={}'.format(missing[:8], extra[:8], wrong[:8]))
    # Keep the project's loading API, also for nested checkpoints, without
    # restoring optimizer/scheduler/RNG state or changing the source checkpoint.
    with tempfile.TemporaryDirectory(prefix='wasd_weights_') as temporary:
        weights_path = Path(temporary) / 'model.model'
        torch.save(state, str(weights_path))
        model.loadParameters(str(weights_path))
    model.eval()
    model.requires_grad_(False)
    print('Loaded {} | fusionMode={} | reliabilityHiddenDim={}'.format(checkpoint, mode, hidden_dim))
    print('Forward implementation: {}'.format(
        'official original Light-ASD' if runtime_mode is None else 'local Light-ASD / QMF'))
    if mode != 'sum':
        print('Inference settings: minReliability={}, energyTemperature={}, fusionTemperature={}. '
              'These scalars are not saved by current checkpoints; they must match AVA training.'
              .format(args.minReliability, args.energyTemperature, args.fusionTemperature))
    return model


def forward_validation_backend(model, audio_embed, visual_embed):
    """Follow the imported ASD wrapper's original or QMF validation API."""
    mode = getattr(model, 'fusion_mode', None)
    if mode is None:
        # Official Light-ASD: no lossV.logits(), QMF kwargs, or reliability tuple.
        # Return the full [T, 128] tensor; indexing [0] here would lose frames.
        return model.model.forward_audio_visual_backend(audio_embed, visual_embed)
    if mode not in ('sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'):
        raise ValueError('Unsupported ASD fusion_mode: {}'.format(mode))
    # Keep the local validation flow for ALL its modes, including local sum.
    outs_v = model.model.forward_visual_backend(visual_embed)
    visual_logits = model.lossV.logits(outs_v).reshape(
        audio_embed.shape[0], audio_embed.shape[1], 2)
    backend_args = dict(visual_logits=visual_logits, return_reliability=True)
    if mode == 'qmf_sync_rank':
        backend_args['return_fusion_aux'] = True
    return model.model.forward_audio_visual_backend(
        audio_embed, visual_embed, **backend_args)[0]


def infer(model, dataset, workers, annotation_count):
    import numpy as np
    import torch
    import tqdm

    options = dict(batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    if workers:
        options['prefetch_factor'] = 1  # Bound prefetched full tracks with 64 workers.
    loader = torch.utils.data.DataLoader(dataset, **options)
    scores = np.full(annotation_count, np.nan, dtype=np.float32)
    count = 0
    model.eval()
    with torch.no_grad():
        for audio_feature, visual_feature, labels, track_index in tqdm.tqdm(loader, desc='WASD val'):
            # Same scoring forward as ASD.evaluate_network, including M02/M03's
            # visual classifier input. No training, augmentation, or adaptation.
            audio_embed = model.model.forward_audio_frontend(audio_feature[0].cuda())
            visual_embed = model.model.forward_visual_frontend(visual_feature[0].cuda())
            outs_av = forward_validation_backend(model, audio_embed, visual_embed)
            labels = labels[0].reshape(-1).cuda()
            # Calling lossAV without labels returns RAW LOGITS in this project!
            # The validation branch returns softmax probabilities; labels are
            # used only to compute the discarded loss/accuracy, never to update.
            _, pred_score, _, _ = model.lossAV.forward(outs_av, labels)
            track_scores = pred_score[:, 1].detach().cpu().numpy()
            rows = dataset.tracks[int(track_index.item())]['rows']
            if len(track_scores) != len(rows) or not np.isfinite(track_scores).all():
                raise ValueError('Invalid prediction count/values for track {}'.format(track_index.item()))
            if np.isfinite(scores[rows]).any():
                raise ValueError('Duplicate prediction assignment for annotation rows')
            scores[rows] = track_scores
            count += len(track_scores)
    if count != annotation_count or not np.isfinite(scores).all():
        raise ValueError('Prediction count mismatch: {} scores for {} annotations; '
                         'val_res.csv was not written.'.format(count, annotation_count))
    return scores


def save_predictions(annotations, scores, output):
    import numpy as np

    scores = np.asarray(scores)
    if (scores.shape != (len(annotations),) or not np.isfinite(scores).all() or
            ((scores < 0) | (scores > 1)).any()):
        raise ValueError('Prediction count/values invalid; val_res.csv was not written')
    result = annotations.loc[:, GT_COLUMNS].copy()
    result['label'] = 'SPEAKING_AUDIBLE'
    result['score'] = scores
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # No label_id/instance_id, and no header (official WASD load_csv uses header=None).
    with tempfile.TemporaryDirectory(prefix='wasd_csv_', dir=str(output.parent)) as temporary:
        staged = Path(temporary) / 'val_res.csv'
        result.to_csv(staged, columns=PRED_COLUMNS, index=False, header=False)
        os.replace(str(staged), str(output))
    print('Saved {} scores: {}'.format(len(scores), output))


def check_evaluator(eval_dir):
    """Return a heading-only legacy copy as bytes, or None for modern headings."""
    require_path(eval_dir, 'WASD evaluator directory', True)
    require_path(eval_dir / 'WASD_evaluation.py', 'Official WASD_evaluation.py')
    division = eval_dir / 'dataset_division.txt'
    require_path(division, 'Official dataset_division.txt')
    source = division.read_bytes()
    bom = b'\xef\xbb\xbf' if source.startswith(b'\xef\xbb\xbf') else b''
    lines = source[len(bom):].splitlines(keepends=True)
    sections = []
    for index, line in enumerate(lines):
        match = DIVISION_HEADING.fullmatch(line)
        if match:
            try:
                heading = match.group(2).decode('utf-8')
            except UnicodeDecodeError as exc:
                raise ValueError('Invalid UTF-8 section heading in {}'.format(division)) from exc
            sections.append((index, match, heading))
        elif b'#' in line or (line.strip() and not sections):
            raise ValueError('Invalid dataset_division.txt format at {}:{}'
                             .format(division, index + 1))
    headings = [heading for _, _, heading in sections]
    if len(headings) == len(CATEGORIES) and set(headings) == set(CATEGORIES):
        return None  # Already fixed upstream: use the original file and cwd.
    if len(headings) != len(LEGACY_CATEGORIES) or set(headings) != set(LEGACY_CATEGORIES):
        raise ValueError(
            'Unknown, mixed, missing or duplicate section headings in {}: {}. '
            'Expected exactly OC/SI/FO/HVN/SS or Interview/Debate/Podcast/React/Police; '
            'no mapping will be guessed.'.format(division, headings))
    for index, match, heading in sections:
        # Replace ONLY the section title, never any video IDs or their order.
        lines[index] = (match.group(1) + LEGACY_CATEGORIES[heading].encode('ascii')
                        + match.group(3))
    return bom + b''.join(lines)


def evaluate_wasd(eval_dir, original_csv, prediction_csv):
    """Also callable on saved predictions, without loading torch or doing inference."""
    eval_dir = Path(eval_dir).expanduser().resolve()
    prediction_csv = Path(prediction_csv).expanduser().resolve()
    log_path = prediction_csv.parent / 'wasd_eval.txt'
    with log_path.open('w', encoding='utf-8') as log:
        def report(text):
            print(text, end='', flush=True)
            log.write(text)
            log.flush()

        try:
            compatible_division = check_evaluator(eval_dir)
            require_path(prediction_csv, 'WASD predictions')
            annotations = read_annotations(original_csv)
            # Official WASD_evaluation.py reads 8 positional GT columns, while
            # prepared val_orig.csv has a header and extra training columns.
            with tempfile.TemporaryDirectory(prefix='wasd_eval_') as temporary:
                temporary = Path(temporary)
                evaluation_cwd = eval_dir
                if compatible_division is not None:
                    (temporary / 'dataset_division.txt').write_bytes(compatible_division)
                    evaluation_cwd = temporary
                    report('Using temporary legacy heading compatibility: {}. '
                           'Original dataset_division.txt is unchanged.\n'.format(
                               ', '.join('{} -> {}'.format(old, new)
                                         for old, new in LEGACY_CATEGORIES.items())))
                groundtruth = temporary / 'groundtruth.csv'
                annotations.to_csv(groundtruth, columns=GT_COLUMNS, index=False, header=False)
                command = [sys.executable, '-u', '-O', str(eval_dir / 'WASD_evaluation.py'),
                           '-g', str(groundtruth), '-p', str(prediction_csv)]
                report('Official WASD evaluator (cwd={}):\n{}\n'.format(evaluation_cwd, shlex.join(command)))
                with subprocess.Popen(command, cwd=str(evaluation_cwd), stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                      errors='replace', bufsize=1) as process:
                    for line in process.stdout:
                        report(line)
                    returncode = process.wait()
                if returncode:
                    raise RuntimeError('Official WASD evaluator exited with code {}'.format(returncode))
            report('Official evaluation completed. Log: {}\n'.format(log_path))
        except (OSError, ValueError, RuntimeError) as exc:
            report('Evaluation failed: {}\nPredictions retained: {}\n'.format(exc, prediction_csv))
            raise RuntimeError('WASD evaluation failed; see {}'.format(log_path)) from exc


def print_manual_evaluation(original_csv, prediction_csv):
    # Reuse the exact official subprocess/GT adapter without another GPU run.
    code = ('from WASD_test import evaluate_wasd; evaluate_wasd({}, {}, {})'.format(
        repr('/root/WASD/eval'), repr(str(original_csv)), repr(str(prediction_csv))))
    print('To evaluate later from this project directory (no inference), set the '
          'official eval directory in this command:\n' + shlex.join([sys.executable, '-c', code]))


def main():
    args = parse_args()  # --help requires only the Python standard library.
    root, checkpoint = validate_paths(args)
    output = Path(args.savePath).expanduser().resolve() / 'val_res.csv'
    if args.wasdEvalDir:
        # Reject unknown layouts before spending time on GPU inference.
        check_evaluator(Path(args.wasdEvalDir).expanduser().resolve())
    annotations = read_annotations(root / 'csv/val_orig.csv')
    dataset = WASDValLoader(root, annotations)
    print('Validated {} entities / {} annotation rows.'.format(len(dataset), len(annotations)))
    model = load_model(args, checkpoint)
    scores = infer(model, dataset, args.nDataLoaderThread, len(annotations))
    save_predictions(annotations, scores, output)
    if args.wasdEvalDir:
        evaluate_wasd(args.wasdEvalDir, root / 'csv/val_orig.csv', output)
    else:
        print('No --wasdEvalDir provided; inference completed without evaluation.')
        print_manual_evaluation(root / 'csv/val_orig.csv', output)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        sys.exit('WASD test error: {}'.format(error))
