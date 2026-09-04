import torch
import torch.nn as nn

import sys, time, numpy, os, subprocess, pandas, tqdm, random
from subprocess import PIPE

from loss import lossAV, lossV
from model.Model import ASD_Model


class _ReliabilityStats(object):
    """Accumulate frame-wise reliability moments without storing predictions."""

    def __init__(self):
        self.count = 0
        self.value_sum = None
        self.square_sum = None
        self.minimum = None
        self.maximum = None

    def update(self, reliability):
        values = reliability.detach().reshape(-1, 2).float()
        if values.numel() == 0:
            return

        batch_sum = values.sum(dim=0)
        batch_square_sum = (values * values).sum(dim=0)
        batch_minimum = values.min(dim=0)[0]
        batch_maximum = values.max(dim=0)[0]
        if self.value_sum is None:
            self.value_sum = batch_sum
            self.square_sum = batch_square_sum
            self.minimum = batch_minimum
            self.maximum = batch_maximum
        else:
            self.value_sum += batch_sum
            self.square_sum += batch_square_sum
            self.minimum = torch.minimum(self.minimum, batch_minimum)
            self.maximum = torch.maximum(self.maximum, batch_maximum)
        self.count += values.shape[0]

    def summary(self):
        if self.count == 0:
            return None
        mean = self.value_sum / self.count
        variance = torch.clamp(self.square_sum / self.count - mean * mean, min=0.0)
        std = torch.sqrt(variance)
        result = torch.stack((mean, std, self.minimum, self.maximum), dim=1)
        return result.cpu().tolist()


class _CorrelationStats(object):
    """Streaming Pearson correlation for visual score and visual loss."""

    def __init__(self):
        self.count = 0
        self.x_sum = 0.0
        self.y_sum = 0.0
        self.xx_sum = 0.0
        self.yy_sum = 0.0
        self.xy_sum = 0.0

    def update(self, x, y):
        x = x.detach().reshape(-1).double()
        y = y.detach().reshape(-1).double()
        if x.numel() != y.numel():
            raise ValueError("Correlation inputs must have the same size.")
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.x_sum += x.sum().item()
        self.y_sum += y.sum().item()
        self.xx_sum += (x * x).sum().item()
        self.yy_sum += (y * y).sum().item()
        self.xy_sum += (x * y).sum().item()

    def summary(self):
        if self.count < 2:
            return None
        count = float(self.count)
        covariance = self.xy_sum - self.x_sum * self.y_sum / count
        x_variance = self.xx_sum - self.x_sum * self.x_sum / count
        y_variance = self.yy_sum - self.y_sum * self.y_sum / count
        denominator = max(x_variance * y_variance, 0.0) ** 0.5
        if denominator <= 1e-12:
            return 0.0
        return covariance / denominator


class ASD(nn.Module):
    def __init__(self, lr = 0.001, lrDecay = 0.95, **kwargs):
        super(ASD, self).__init__()        
        self.model = ASD_Model(
            fusion_mode=kwargs.get('fusionMode', 'qmf'),
            reliability_hidden_dim=kwargs.get('reliabilityHiddenDim', 32),
            reliability_dropout=kwargs.get('reliabilityDropout', 0.1),
            min_reliability=kwargs.get('minReliability', 0.1),
            energy_temperature=kwargs.get('energyTemperature', 1.0),
            fusion_temperature=kwargs.get('fusionTemperature', 1.0),
        ).cuda()
        self.lossAV = lossAV().cuda()
        self.lossV = lossV().cuda()
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        self.last_train_reliability = None
        self.last_eval_reliability = None
        self.last_train_gpu_memory = None
        self.last_eval_gpu_memory = None
        self.last_train_sync_loss = None
        self.last_train_rank_loss = None
        self.last_train_visual_loss_correlation = None
        self.last_eval_visual_loss_correlation = None
        self.fusion_mode = kwargs.get('fusionMode', 'qmf')
        self.lambda_sync = kwargs.get('lambdaSync', 0.1)
        self.lambda_rank = kwargs.get('lambdaRank', 0.1)
        self.rank_margin = kwargs.get('rankMargin', 0.1)
        self.rank_min_loss_gap = kwargs.get('rankMinLossGap', 0.05)
        self.reliability_names = (
            ('WA', 'WV') if self.fusion_mode in (
                'qmf_sync', 'qmf_sync_rank'
            )
            else ('RA', 'RV')
        )
        print(time.strftime("%m-%d %H:%M:%S") + " Model para number = %.2f"%(sum(param.numel() for param in self.model.parameters()) / 1000 / 1000))

    def format_reliability_stats(self, stats, prefix):
        if stats is None:
            return "%s reliability: unavailable" % prefix
        audio, visual = stats
        audio_name, visual_name = self.reliability_names
        return (
            "%s %s(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f, "
            "%s(mean/std/min/max)=%.4f/%.4f/%.4f/%.4f"
            % (prefix, audio_name, *audio, visual_name, *visual)
        )

    @staticmethod
    def _peak_gpu_memory():
        if not torch.cuda.is_available():
            return None
        mb = 1024.0 * 1024.0
        return {
            'allocated_mb': torch.cuda.max_memory_allocated() / mb,
            'reserved_mb': torch.cuda.max_memory_reserved() / mb,
        }

    @staticmethod
    def format_gpu_memory(memory, prefix):
        if memory is None:
            return "%s GPU memory: unavailable" % prefix
        return "%s GPU peak allocated/reserved=%.1f/%.1f MB" % (
            prefix, memory['allocated_mb'], memory['reserved_mb']
        )

    def format_fusion_diagnostics(self):
        if self.fusion_mode == 'qmf_sync':
            fusion = self.model.reliabilityFusion
            return (
                "VisualEnergy(scale/bias)=%.6f/%.6f"
                % (
                    fusion.visual_energy_scale.detach().item(),
                    fusion.visual_energy_bias.detach().item(),
                )
            )
        if self.fusion_mode == 'qmf_sync_rank':
            last_layer = self.model.reliabilityFusion.visual_quality_head[-1]
            return (
                "VisualQualityHead(weight_norm/bias)=%.6f/%.6f"
                % (
                    last_layer.weight.detach().norm().item(),
                    last_layer.bias.detach().item(),
                )
            )
        else:
            return "Fusion diagnostics: unavailable"

    def train_network(self, loader, epoch, **kwargs):
        self.train()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.scheduler.step(epoch - 1)  # StepLR
        index, top1 = 0, 0
        lossV, lossAV, lossSync, lossRank, loss = 0, 0, 0, 0, 0
        reliability_stats = _ReliabilityStats()
        visual_loss_correlation = _CorrelationStats()
        lr = self.optim.param_groups[0]['lr']
        r = 1.3 - 0.02 * (epoch - 1)
        for num, (audioFeature, visualFeature, labels) in enumerate(loader, start=1):
            self.zero_grad()

            audioEmbed = self.model.forward_audio_frontend(audioFeature[0].cuda())
            visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
            outsV = self.model.forward_visual_backend(visualEmbed)
            visualLogits = self.lossV.logits(outsV)
            frameVisualLogits = visualLogits.reshape(
                audioEmbed.shape[0], audioEmbed.shape[1], 2
            )

            if self.fusion_mode == 'qmf_sync_rank':
                outsAV, reliability, fusionAux = (
                    self.model.forward_audio_visual_backend(
                        audioEmbed,
                        visualEmbed,
                        visual_logits=frameVisualLogits,
                        return_reliability=True,
                        return_fusion_aux=True,
                    )
                )
            else:
                outsAV, reliability = self.model.forward_audio_visual_backend(
                    audioEmbed,
                    visualEmbed,
                    visual_logits=frameVisualLogits,
                    return_reliability=True,
                )
            reliability_stats.update(reliability)

            labels = labels[0].reshape((-1)).cuda() # Loss
            nlossAV, _, _, prec = self.lossAV.forward(outsAV, labels, r)
            frameVisualLosses = self.lossV.frame_losses_from_logits(
                visualLogits, labels, r
            )
            nlossV = frameVisualLosses.mean()
            if self.fusion_mode in ('qmf_sync', 'qmf_sync_rank'):
                nlossSync = self.model.reliabilityFusion.synchronization_loss(
                    audioEmbed, visualEmbed, labels
                )
            else:
                nlossSync = reliability.new_zeros(())
            if self.fusion_mode == 'qmf_sync_rank':
                visualScores = fusionAux['visual_score'].reshape(-1)
                nlossRank = self.model.reliabilityFusion.visual_ranking_loss(
                    visualScores,
                    frameVisualLosses,
                    margin=self.rank_margin,
                    min_loss_gap=self.rank_min_loss_gap,
                )
                visual_loss_correlation.update(
                    visualScores, frameVisualLosses
                )
            else:
                nlossRank = reliability.new_zeros(())
            nloss = (
                nlossAV
                + 0.5 * nlossV
                + self.lambda_sync * nlossSync
                + self.lambda_rank * nlossRank
            )

            lossV += nlossV.detach().cpu().numpy()
            lossAV += nlossAV.detach().cpu().numpy()
            lossSync += nlossSync.detach().cpu().numpy()
            lossRank += nlossRank.detach().cpu().numpy()
            loss += nloss.detach().cpu().numpy()
            top1 += prec
            nloss.backward()
            self.optim.step()
            index += len(labels)
            sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
            " [%2d] r: %2f, Lr: %5f, Training: %.2f%%, "    %(epoch, r, lr, 100 * (num / loader.__len__())) + \
            " LossV: %.5f, LossAV: %.5f, LossSync: %.5f, "
            "LossRank: %.5f, Loss: %.5f, ACC: %2.2f%% \r"
            % (lossV/(num), lossAV/(num), lossSync/(num),
               lossRank/(num), loss/(num), 100 * (top1/index)))
            sys.stderr.flush()  

        sys.stdout.write("\n")      
        self.last_train_reliability = reliability_stats.summary()
        self.last_train_gpu_memory = self._peak_gpu_memory()
        self.last_train_sync_loss = lossSync / num
        self.last_train_rank_loss = lossRank / num
        self.last_train_visual_loss_correlation = (
            visual_loss_correlation.summary()
        )
        print(self.format_reliability_stats(self.last_train_reliability, 'Train'))
        if self.fusion_mode in ('qmf_sync', 'qmf_sync_rank'):
            print(self.format_fusion_diagnostics())
        if self.fusion_mode == 'qmf_sync_rank':
            print(
                "Train VScoreLossCorr=%.6f"
                % (self.last_train_visual_loss_correlation or 0.0)
            )
        print(self.format_gpu_memory(self.last_train_gpu_memory, 'Train'))

        return loss/num, lr

    def evaluate_network(self, loader, evalCsvSave, evalOrig, **kwargs):
        self.eval()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        predScores = []
        reliability_stats = _ReliabilityStats()
        visual_loss_correlation = _CorrelationStats()
        for audioFeature, visualFeature, labels in tqdm.tqdm(loader):
            with torch.no_grad():                
                audioEmbed  = self.model.forward_audio_frontend(audioFeature[0].cuda())
                visualEmbed = self.model.forward_visual_frontend(visualFeature[0].cuda())
                outsV = self.model.forward_visual_backend(visualEmbed)
                visualLogits = self.lossV.logits(outsV).reshape(
                    audioEmbed.shape[0], audioEmbed.shape[1], 2
                )
                if self.fusion_mode == 'qmf_sync_rank':
                    outsAV, reliability, fusionAux = (
                        self.model.forward_audio_visual_backend(
                            audioEmbed,
                            visualEmbed,
                            visual_logits=visualLogits,
                            return_reliability=True,
                            return_fusion_aux=True,
                        )
                    )
                else:
                    outsAV, reliability = (
                        self.model.forward_audio_visual_backend(
                            audioEmbed,
                            visualEmbed,
                            visual_logits=visualLogits,
                            return_reliability=True,
                        )
                    )
                reliability_stats.update(reliability)
                labels = labels[0].reshape((-1)).cuda()             
                if self.fusion_mode == 'qmf_sync_rank':
                    frameVisualLosses = self.lossV.frame_losses_from_logits(
                        visualLogits.reshape(-1, 2), labels
                    )
                    visual_loss_correlation.update(
                        fusionAux['visual_score'], frameVisualLosses
                    )
                _, predScore, _, _ = self.lossAV.forward(outsAV, labels)    
                predScore = predScore[:,1].detach().cpu().numpy()
                predScores.extend(predScore)
                # break
        self.last_eval_reliability = reliability_stats.summary()
        self.last_eval_visual_loss_correlation = (
            visual_loss_correlation.summary()
        )
        self.last_eval_gpu_memory = self._peak_gpu_memory()
        print(self.format_reliability_stats(self.last_eval_reliability, 'Eval'))
        if self.fusion_mode == 'qmf_sync_rank':
            print(
                "Eval VScoreLossCorr=%.6f"
                % (self.last_eval_visual_loss_correlation or 0.0)
            )
        print(self.format_gpu_memory(self.last_eval_gpu_memory, 'Eval'))
        evalLines = open(evalOrig).read().splitlines()[1:]
        labels = []
        labels = pandas.Series( ['SPEAKING_AUDIBLE' for line in evalLines])
        scores = pandas.Series(predScores)
        evalRes = pandas.read_csv(evalOrig)
        evalRes['score'] = scores
        evalRes['label'] = labels
        evalRes.drop(['label_id'], axis=1,inplace=True)
        evalRes.drop(['instance_id'], axis=1,inplace=True)
        evalRes.to_csv(evalCsvSave, index=False)
        cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s "%(evalOrig, evalCsvSave)
        mAP_result = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        
        # 检查命令是否成功执行
        if mAP_result.returncode != 0:
            print("Error running evaluation script:")
            print("stderr:", mAP_result.stderr.decode('utf-8'))
            return 0.0
            
        # 尝试解析输出
        output_str = mAP_result.stdout.decode('utf-8')
        try:
            mAP = float(output_str.split(' ')[2][:5])
        except (IndexError, ValueError) as e:
            print("Error parsing evaluation result:")
            print("Output was:", output_str)
            print("Error:", str(e))
            return 0.0
            
        return mAP

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def saveCheckpoint(self, path, epoch, best_mAP,
                       loader_generator_state=None):
        checkpoint = {
            'epoch': epoch,
            'best_mAP': best_mAP,
            'fusion_mode': self.fusion_mode,
            'state_dict': self.state_dict(),
            'optimizer': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'python_rng_state': random.getstate(),
            'numpy_rng_state': numpy.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'loader_generator_state': loader_generator_state,
        }
        if torch.cuda.is_available():
            checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, path)

    def loadCheckpoint(self, path):
        checkpoint = torch.load(path)
        checkpoint_mode = checkpoint.get('fusion_mode')
        if checkpoint_mode is not None and checkpoint_mode != self.fusion_mode:
            raise ValueError(
                "Checkpoint fusion mode %s does not match requested mode %s."
                % (checkpoint_mode, self.fusion_mode)
            )
        self.load_state_dict(checkpoint['state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            numpy.random.set_state(checkpoint['numpy_rng_state'])
        if 'torch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['torch_rng_state'])
        if torch.cuda.is_available() and 'cuda_rng_state_all' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        return (
            checkpoint['epoch'],
            checkpoint.get('best_mAP'),
            checkpoint.get('loader_generator_state'),
        )

    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name;
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)
