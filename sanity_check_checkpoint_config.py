"""Dataset-free CPU checks: python sanity_check_checkpoint_config.py."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from ASD import ASD
from model.Model import ASD_Model
from WASD_test import load_model, parse_args
from utils.checkpoint_config import (
    MODEL_CONFIG_FIELDS, checkpoint_config, load_checkpoint_payload,
    model_config_kwargs, resolve_checkpoint_config,
)


CONFIG = dict(fusion_mode='qmf_sync_rank', min_reliability=0.2,
              fusion_temperature=0.5, energy_temperature=0.8,
              reliability_hidden_dim=48, reliability_dropout=0.15)


def cpu_asd(config):
    # Exercise the actual wrapper/save API without changing its CUDA behavior.
    with patch.object(torch.nn.Module, 'cuda', lambda self: self):
        return ASD(**model_config_kwargs(config))


def resolve(payload, requested=None):
    return resolve_checkpoint_config(payload, requested, report=lambda message: None)


class CheckpointConfigChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        torch.set_num_threads(1)
        cls.model = cpu_asd(CONFIG).eval().requires_grad_(False)
        # Non-zero learned heads expose temperature/floor restoration errors;
        # zero-initialized heads alone would mask those errors.
        with torch.no_grad():
            fusion = cls.model.model.reliabilityFusion
            for head in (fusion.sync_head, fusion.visual_quality_head):
                head[-1].weight.normal_(0, 0.2)
                head[-1].bias.fill_(0.3)
        cls.temporary = tempfile.TemporaryDirectory(prefix='asd_config_check_')
        cls.path = Path(cls.temporary.name) / 'training_0001.checkpoint'
        with patch.object(torch.cuda, 'is_available', return_value=False):
            cls.model.saveCheckpoint(str(cls.path), epoch=1, best_mAP=0.5)
        cls.payload = load_checkpoint_payload(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_1_roundtrip_and_wasd_loader(self):
        self.assertEqual(self.payload['model_config'], CONFIG)
        self.assertEqual(set(self.payload['model_config']), set(MODEL_CONFIG_FIELDS))
        _, config = resolve(self.payload)
        self.assertEqual(config, CONFIG)
        args = parse_args(['--dataPathWASD', 'unused', '--pretrainModel', str(self.path)])
        with patch.object(torch.nn.Module, 'cuda', lambda self: self), \
                patch.object(torch.cuda, 'is_available', return_value=True):
            restored = load_model(args, self.path)
        self.assertEqual(restored.model_config, CONFIG)
        self.assertEqual(restored.model.reliabilityFusion.sync_head[2].p, 0.15)
        audio, visual = torch.randn(2, 5, 128), torch.randn(2, 5, 128)
        logits = torch.randn(2, 5, 2)
        with torch.no_grad():
            before = self.model.model.forward_audio_visual_backend(audio, visual, logits)
            after = restored.model.forward_audio_visual_backend(audio, visual, logits)
        self.assertTrue(torch.equal(before, after))

    def test_2_explicit_conflicts_and_matching_values(self):
        _, config = resolve(self.payload, model_config_kwargs(CONFIG))
        self.assertEqual(config, CONFIG)
        conflicts = dict(fusionMode='qmf_sync', minReliability=0.1,
                         fusionTemperature=1.0, energyTemperature=1.0,
                         reliabilityHiddenDim=32, reliabilityDropout=0.1)
        for cli, value in conflicts.items():
            with self.subTest(cli=cli), self.assertRaisesRegex(ValueError, cli):
                resolve(self.payload, {cli: value})
        payload = dict(self.payload, model_config=dict(CONFIG, fusion_temperature=1.0))
        with self.assertRaisesRegex(ValueError, 'fusionTemperature=1.0.*fusionTemperature=0.5'):
            resolve(payload, {'fusionTemperature': 0.5})
        # Explicitly requesting a parser's old default must still conflict.
        args = parse_args(['--dataPathWASD', 'unused', '--pretrainModel', 'unused',
                           '--minReliability=0.1'])
        self.assertIsNone(args.fusionTemperature)
        with self.assertRaisesRegex(ValueError, 'minReliability'):
            resolve(self.payload, vars(args))

    def test_3_legacy_models_and_training_checkpoints(self):
        for mode in ('sum', 'qmf', 'qmf_sync', 'qmf_sync_rank'):
            with self.subTest(mode=mode):
                config = dict(CONFIG, fusion_mode=mode)
                model = cpu_asd(config)
                path = Path(self.temporary.name) / (mode + '.model')
                model.saveParameters(str(path))
                state = load_checkpoint_payload(path)
                # Keep AVA's existing permissive initialization API compatible.
                ava_model = cpu_asd(config)
                ava_model.loadParameters(str(path))
                self.assertTrue(all(torch.equal(value, ava_model.state_dict()[key])
                                    for key, value in state.items()))
                expected_hidden = 32 if mode == 'sum' else 48
                self.assertEqual(checkpoint_config(state), (mode, expected_hidden))
                # Plain, wrapped legacy and DDP weights all follow the fallback.
                for payload in (state, {'state_dict': state, 'fusion_mode': mode},
                                {'module.' + key: value for key, value in state.items()}):
                    messages = []
                    normalized, restored = resolve_checkpoint_config(
                        payload, report=messages.append)
                    self.assertEqual(restored['fusion_mode'], mode)
                    self.assertEqual(restored['reliability_hidden_dim'], expected_hidden)
                    self.assertEqual(restored['min_reliability'], 0.1)
                    self.assertIn('Make sure these values match training.', messages[0])
                    self.assertIn('reliabilityDropout=0.1', messages[0])
                    cpu_asd(restored).load_state_dict(normalized, strict=True)
                _, restored = resolve(state, {'minReliability': 0.2,
                                              'fusionTemperature': 0.5})
                self.assertEqual(restored['min_reliability'], 0.2)
                self.assertEqual(restored['fusion_temperature'], 0.5)

    def test_4_state_dict_names_shapes_and_values_unchanged(self):
        state = self.model.state_dict()
        saved = self.payload['state_dict']
        self.assertEqual(set(state), set(saved))
        reference = ASD_Model(**CONFIG).state_dict()
        self.assertEqual(set(reference), {key[len('model.'):] for key in saved
                                          if key.startswith('model.')})
        for key in state:
            self.assertEqual(state[key].shape, saved[key].shape)
            self.assertTrue(torch.equal(state[key], saved[key]))
        for key, value in reference.items():
            self.assertEqual(value.shape, saved['model.' + key].shape)

    def test_5_sum_equals_direct_add_then_gru(self):
        model = ASD_Model(fusion_mode='sum').eval()
        audio, visual = torch.randn(2, 7, 128), torch.randn(2, 7, 128)
        with torch.no_grad():
            actual = model.forward_audio_visual_backend(audio, visual)
            direct = model.GRU(audio + visual).reshape(-1, 128)
        self.assertTrue(torch.allclose(actual, direct, rtol=0, atol=0))

    def test_6_bad_or_partial_metadata(self):
        for key, value in (('fusion_mode', 'sum'), ('reliability_hidden_dim', 32)):
            payload = dict(self.payload, model_config=dict(CONFIG, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve(payload)
        partial = {'state_dict': self.payload['state_dict'],
                   'fusion_mode': 'qmf_sync_rank', 'fusion_temperature': 0.5}
        _, restored = resolve(partial)
        self.assertEqual(restored['reliability_hidden_dim'], 48)
        self.assertEqual(restored['fusion_temperature'], 0.5)
        self.assertEqual(restored['energy_temperature'], 1.0)
        for key, value in (('energy_temperature', float('nan')),
                           ('fusion_temperature', 0), ('reliability_dropout', 1.5)):
            payload = dict(self.payload, model_config=dict(CONFIG, **{key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve(payload)


if __name__ == '__main__':
    unittest.main(verbosity=2)
