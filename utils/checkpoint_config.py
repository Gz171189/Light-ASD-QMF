"""Inference-only constructor metadata; no optimizer/loss/training settings."""

import math


# (ASD/CLI name, legacy default). Keep the ASD_Model constructor unchanged.
MODEL_CONFIG_FIELDS = {
    'fusion_mode': ('fusionMode', 'qmf'),
    'min_reliability': ('minReliability', 0.1),
    'energy_temperature': ('energyTemperature', 1.0),
    'fusion_temperature': ('fusionTemperature', 1.0),
    'reliability_hidden_dim': ('reliabilityHiddenDim', 32),
    'reliability_dropout': ('reliabilityDropout', 0.1),
}


def model_config_from_kwargs(kwargs):
    return {key: kwargs.get(cli, default)
            for key, (cli, default) in MODEL_CONFIG_FIELDS.items()}


def model_config_kwargs(config):
    return {cli: config[key] for key, (cli, _) in MODEL_CONFIG_FIELDS.items()}


def checkpoint_config(state):
    """Detect architecture from the existing ASD tensor names, including DDP."""
    state = {name.replace('module.', ''): value for name, value in state.items()}
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
        return 'sum', 32  # sum has no reliability head; width is unused.
    if prefix + key not in state or state[prefix + key].ndim != 2:
        raise ValueError('Invalid QMF hidden-dimension tensor: ' + prefix + key)
    return mode, int(state[prefix + key].shape[0])


def validate_model_config(config):
    if config['fusion_mode'] not in ('sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'):
        raise ValueError('Invalid fusion_mode: {}'.format(config['fusion_mode']))
    if not 0 <= config['min_reliability'] < 1:
        raise ValueError('minReliability must be in [0, 1)')
    for key in ('energy_temperature', 'fusion_temperature'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError('{} must be finite and positive'.format(key))
    hidden = config['reliability_hidden_dim']
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
        raise ValueError('reliabilityHiddenDim must be a positive integer')
    if not 0 <= config['reliability_dropout'] <= 1:
        raise ValueError('reliabilityDropout must be in [0, 1]')


def resolve_checkpoint_config(payload, requested=None, report=print):
    """Return normalized state and config before constructing an inference model.

    requested contains only explicit CLI values (None/auto mean unspecified).
    New checkpoints store model_config; old top-level fields remain readable.
    Architecture metadata and explicit settings must agree with the tensors.
    """
    if not isinstance(payload, dict):
        raise ValueError('Expected an ASD state_dict or training checkpoint')
    nested = 'state_dict' in payload
    state = payload['state_dict'] if nested else payload
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint contains no model state_dict')
    state = {name.replace('module.', ''): value for name, value in state.items()}
    mode, hidden = checkpoint_config(state)
    saved = {}
    if nested:
        metadata = payload.get('model_config', {})
        if not isinstance(metadata, dict):
            raise ValueError('Checkpoint model_config must be a dictionary')
        for key in MODEL_CONFIG_FIELDS:
            if key in payload:
                saved[key] = payload[key]
            if key in metadata:
                if key in saved and saved[key] != metadata[key]:
                    raise ValueError('Conflicting checkpoint metadata: {}'.format(key))
                saved[key] = metadata[key]

    inferred = {'fusion_mode': mode}
    if mode != 'sum':
        inferred['reliability_hidden_dim'] = hidden
    for key, value in inferred.items():
        if key in saved and saved[key] != value:
            raise ValueError('Checkpoint {} metadata={} disagrees with its tensors={}'
                             .format(key, saved[key], value))

    config = {key: default for key, (_, default) in MODEL_CONFIG_FIELDS.items()}
    config.update(inferred)
    config.update(saved)
    requested = requested or {}
    for key, (cli, _) in MODEL_CONFIG_FIELDS.items():
        value = requested.get(cli)
        if value is None or (key == 'fusion_mode' and value == 'auto'):
            continue
        if (key in saved or key in inferred) and value != config[key]:
            raise ValueError(
                'Checkpoint was trained with {}={}, but command line requested '
                '{}={}. Refusing to run inconsistent inference.'
                .format(cli, config[key], cli, value))
        config[key] = value
    validate_model_config(config)

    missing = [key for key in MODEL_CONFIG_FIELDS if key not in saved and key not in inferred]
    if missing:
        # Print, rather than warnings.warn: train.py suppresses Python warnings.
        kind = 'Legacy training checkpoint' if nested else 'Legacy .model'
        report('WARNING: {} detected.\nThe following settings are not stored in '
               'this checkpoint (using command line/defaults):\n{}\n'
               'Make sure these values match training.'.format(
                   kind, '\n'.join('{}={}'.format(MODEL_CONFIG_FIELDS[key][0], config[key])
                                   for key in missing)))
    report('Resolved model config: ' + ', '.join(
        '{}={} ({})'.format(cli, config[key], 'checkpoint' if key in saved else
                           'tensors' if key in inferred else 'command line/default')
        for key, (cli, _) in MODEL_CONFIG_FIELDS.items()))
    return state, config


def load_checkpoint_payload(path):
    """Load trusted local checkpoints, including training RNG state on Torch 2.6+."""
    import torch
    return torch.load(str(path), map_location='cpu', weights_only=False)
