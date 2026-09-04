import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Classifier import BGRU
from model.Encoder import visual_encoder, audio_encoder


class FrameWiseReliabilityFusion(nn.Module):
    """QMF-inspired, frame-wise reliability weighting for two modalities.

    The two reliability heads operate independently on audio and visual
    embeddings.  Their output is constrained to ``[min_reliability, 1]``.
    A fixed scale makes the zero-initialized heads exactly reproduce the
    original Light-ASD fusion (``audio + visual``), which is useful when
    loading a baseline checkpoint.
    """

    def __init__(self, feature_dim=128, hidden_dim=32, dropout=0.1,
                 min_reliability=0.1):
        super(FrameWiseReliabilityFusion, self).__init__()
        if not 0.0 <= min_reliability < 1.0:
            raise ValueError("min_reliability must be in [0, 1).")

        self.min_reliability = float(min_reliability)
        self.audio_reliability = self._make_head(feature_dim, hidden_dim, dropout)
        self.visual_reliability = self._make_head(feature_dim, hidden_dim, dropout)
        self._init_as_baseline()

    @staticmethod
    def _make_head(feature_dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _init_as_baseline(self):
        # sigmoid(0) = 0.5.  The scale used in forward then gives a unit
        # gain to both modalities, so a baseline checkpoint is not perturbed.
        for head in (self.audio_reliability, self.visual_reliability):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(self, audio_embedding, visual_embedding):
        if audio_embedding.shape != visual_embedding.shape:
            raise ValueError(
                "Audio and visual embeddings must have the same [B, T, C] "
                "shape, got %s and %s."
                % (tuple(audio_embedding.shape), tuple(visual_embedding.shape))
            )
        if audio_embedding.dim() != 3:
            raise ValueError(
                "Frame-wise fusion expects [B, T, C] embeddings, got %s."
                % (tuple(audio_embedding.shape),)
            )

        audio_reliability = torch.sigmoid(self.audio_reliability(audio_embedding))
        visual_reliability = torch.sigmoid(self.visual_reliability(visual_embedding))

        reliability_floor = self.min_reliability
        audio_reliability = reliability_floor + (1.0 - reliability_floor) * audio_reliability
        visual_reliability = reliability_floor + (1.0 - reliability_floor) * visual_reliability

        # At initialization each reliability is (1 + floor) / 2.  This scale
        # therefore makes both gains exactly one and preserves x_audio+x_visual.
        baseline_scale = 2.0 / (1.0 + reliability_floor)
        fused = baseline_scale * (
            audio_reliability * audio_embedding
            + visual_reliability * visual_embedding
        )
        reliability = torch.cat((audio_reliability, visual_reliability), dim=-1)
        return fused, reliability


class FrameWiseAsymmetricQMFFusion(nn.Module):
    """ASD-adapted QMF fusion without an invalid audio-only ASD head.

    Visual quality is derived from the energy of the valid visual auxiliary
    classifier.  The audio-side score is candidate-conditioned and predicts
    audio-visual correspondence from both embeddings.  Scores are normalized
    into two weights whose sum is always two; zero initialization therefore
    reproduces the original ``audio + visual`` fusion exactly.
    """

    def __init__(self, feature_dim=128, hidden_dim=32, dropout=0.1,
                 min_weight=0.1, energy_temperature=1.0,
                 fusion_temperature=1.0):
        super(FrameWiseAsymmetricQMFFusion, self).__init__()
        if not 0.0 <= min_weight < 1.0:
            raise ValueError("min_weight must be in [0, 1).")
        if energy_temperature <= 0.0 or fusion_temperature <= 0.0:
            raise ValueError("QMF temperatures must be positive.")

        self.min_weight = float(min_weight)
        self.energy_temperature = float(energy_temperature)
        self.fusion_temperature = float(fusion_temperature)
        self.sync_head = nn.Sequential(
            nn.Linear(feature_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # The affine calibration starts at zero so visual energy and the
        # zero-initialized sync score are equal before any optimization.
        self.visual_energy_scale = nn.Parameter(torch.zeros(1))
        self.visual_energy_bias = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.sync_head[-1].weight)
        nn.init.zeros_(self.sync_head[-1].bias)

    @staticmethod
    def _validate_embeddings(audio_embedding, visual_embedding):
        if audio_embedding.shape != visual_embedding.shape:
            raise ValueError(
                "Audio and visual embeddings must have the same [B, T, C] "
                "shape, got %s and %s."
                % (tuple(audio_embedding.shape), tuple(visual_embedding.shape))
            )
        if audio_embedding.dim() != 3:
            raise ValueError(
                "Frame-wise fusion expects [B, T, C] embeddings, got %s."
                % (tuple(audio_embedding.shape),)
            )

    def synchronization_logits(self, audio_embedding, visual_embedding):
        self._validate_embeddings(audio_embedding, visual_embedding)
        evidence = torch.cat((
            audio_embedding,
            visual_embedding,
            torch.abs(audio_embedding - visual_embedding),
            audio_embedding * visual_embedding,
        ), dim=-1)
        return self.sync_head(evidence)

    def synchronization_loss(self, audio_embedding, visual_embedding, labels):
        """Contrast aligned and mismatched pairs on true speaking frames.

        Pure audio cannot predict the current candidate's ASD label.  This
        loss therefore supervises a cross-modal correspondence head instead
        of introducing an audio-only classification loss.
        """
        batch_size, num_frames = audio_embedding.shape[:2]
        frame_labels = labels.reshape(batch_size, num_frames).bool()
        if not torch.any(frame_labels):
            return audio_embedding.sum() * 0.0

        positive_logits = self.synchronization_logits(
            audio_embedding, visual_embedding
        ).squeeze(-1)
        if batch_size > 1:
            negative_audio = torch.roll(audio_embedding, shifts=1, dims=0)
        elif num_frames > 1:
            negative_audio = torch.roll(
                audio_embedding, shifts=max(num_frames // 2, 1), dims=1
            )
        else:
            # There is no meaningful mismatched pair for a single frame.
            return F.binary_cross_entropy_with_logits(
                positive_logits[frame_labels],
                torch.ones_like(positive_logits[frame_labels]),
            )
        negative_logits = self.synchronization_logits(
            negative_audio, visual_embedding
        ).squeeze(-1)

        positive_loss = F.binary_cross_entropy_with_logits(
            positive_logits[frame_labels],
            torch.ones_like(positive_logits[frame_labels]),
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negative_logits[frame_labels],
            torch.zeros_like(negative_logits[frame_labels]),
        )
        return 0.5 * (positive_loss + negative_loss)

    def forward(self, audio_embedding, visual_embedding, visual_logits):
        self._validate_embeddings(audio_embedding, visual_embedding)
        expected_logits_shape = (*audio_embedding.shape[:-1], 2)
        if tuple(visual_logits.shape) != expected_logits_shape:
            raise ValueError(
                "Visual logits must have shape %s, got %s."
                % (expected_logits_shape, tuple(visual_logits.shape))
            )

        sync_score = self.synchronization_logits(
            audio_embedding, visual_embedding
        )
        temperature = self.energy_temperature
        visual_confidence = temperature * torch.logsumexp(
            visual_logits / temperature, dim=-1, keepdim=True
        )
        visual_score = (
            self.visual_energy_scale * visual_confidence
            + self.visual_energy_bias
        )

        scores = torch.cat((sync_score, visual_score), dim=-1)
        relative_confidence = torch.softmax(
            scores / self.fusion_temperature, dim=-1
        )
        weights = (
            self.min_weight
            + 2.0 * (1.0 - self.min_weight) * relative_confidence
        )
        fused = (
            weights[..., 0:1] * audio_embedding
            + weights[..., 1:2] * visual_embedding
        )
        return fused, weights


class FrameWiseRankedAsymmetricQMFFusion(FrameWiseAsymmetricQMFFusion):
    """M03: asymmetric QMF with rank-supervised visual quality.

    The visual quality head consumes the visual embedding and a shift-invariant
    centered-logit energy.  Its final layer is zero initialized, as is the AV
    synchronization head, so the initial fusion is exactly ``audio + visual``.
    A frame-wise ranking loss (computed from visual auxiliary losses during
    training) supplies the missing quality direction without an audio-only ASD
    classifier or a signed visual-energy scale.
    """

    def __init__(self, feature_dim=128, hidden_dim=32, dropout=0.1,
                 min_weight=0.1, energy_temperature=1.0,
                 fusion_temperature=1.0):
        super(FrameWiseRankedAsymmetricQMFFusion, self).__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            min_weight=min_weight,
            energy_temperature=energy_temperature,
            fusion_temperature=fusion_temperature,
        )
        # M03 replaces M02's unconstrained signed affine energy calibration.
        del self.visual_energy_scale
        del self.visual_energy_bias
        self.visual_quality_head = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.visual_quality_head[-1].weight)
        nn.init.zeros_(self.visual_quality_head[-1].bias)

    def centered_visual_energy(self, visual_logits):
        """Return non-negative, common-logit-shift-invariant confidence."""
        centered_logits = visual_logits - visual_logits.mean(
            dim=-1, keepdim=True
        )
        temperature = self.energy_temperature
        num_classes = visual_logits.shape[-1]
        return (
            temperature * torch.logsumexp(
                centered_logits / temperature, dim=-1, keepdim=True
            )
            - temperature * torch.log(
                visual_logits.new_tensor(float(num_classes))
            )
        )

    def visual_quality_score(self, visual_embedding, visual_logits):
        expected_logits_shape = (*visual_embedding.shape[:-1], 2)
        if tuple(visual_logits.shape) != expected_logits_shape:
            raise ValueError(
                "Visual logits must have shape %s, got %s."
                % (expected_logits_shape, tuple(visual_logits.shape))
            )
        centered_energy = self.centered_visual_energy(visual_logits)
        quality_evidence = torch.cat(
            (visual_embedding, centered_energy), dim=-1
        )
        return self.visual_quality_head(quality_evidence)

    @staticmethod
    def visual_ranking_loss(visual_scores, frame_losses, margin=0.1,
                            min_loss_gap=0.05):
        """Rank low-loss visual frames above high-loss frames in O(N) memory.

        Frames are sorted by detached visual auxiliary loss.  The better half
        is paired with the worse half, avoiding an O(N^2) pairwise matrix for
        Light-ASD's dynamic batches of up to thousands of frames.
        """
        scores = visual_scores.reshape(-1)
        losses = frame_losses.detach().reshape(-1)
        if scores.numel() != losses.numel():
            raise ValueError(
                "Visual scores and frame losses must contain the same number "
                "of frames, got %d and %d."
                % (scores.numel(), losses.numel())
            )
        pair_count = scores.numel() // 2
        if pair_count == 0:
            return scores.sum() * 0.0

        order = torch.argsort(losses)
        better = order[:pair_count]
        worse = order[-pair_count:]
        valid = (losses[worse] - losses[better]) >= min_loss_gap
        if not torch.any(valid):
            return scores.sum() * 0.0
        score_difference = scores[better[valid]] - scores[worse[valid]]
        return F.relu(margin - score_difference).mean()

    def forward(self, audio_embedding, visual_embedding, visual_logits):
        self._validate_embeddings(audio_embedding, visual_embedding)
        sync_score = self.synchronization_logits(
            audio_embedding, visual_embedding
        )
        visual_score = self.visual_quality_score(
            visual_embedding, visual_logits
        )
        scores = torch.cat((sync_score, visual_score), dim=-1)
        relative_confidence = torch.softmax(
            scores / self.fusion_temperature, dim=-1
        )
        weights = (
            self.min_weight
            + 2.0 * (1.0 - self.min_weight) * relative_confidence
        )
        fused = (
            weights[..., 0:1] * audio_embedding
            + weights[..., 1:2] * visual_embedding
        )
        return fused, weights, visual_score


class ASD_Model(nn.Module):
    def __init__(self, fusion_mode="qmf", reliability_hidden_dim=32,
                 reliability_dropout=0.1, min_reliability=0.1,
                 energy_temperature=1.0, fusion_temperature=1.0):
        super(ASD_Model, self).__init__()

        if fusion_mode not in (
            "sum", "qmf", "qmf_sync", "qmf_sync_rank"
        ):
            raise ValueError(
                "fusion_mode must be 'sum', 'qmf', 'qmf_sync', or "
                "'qmf_sync_rank'."
            )
        self.fusion_mode = fusion_mode
        self.visualEncoder  = visual_encoder()
        self.audioEncoder  = audio_encoder()
        if self.fusion_mode == "qmf":
            self.reliabilityFusion = FrameWiseReliabilityFusion(
                feature_dim=128,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
                min_reliability=min_reliability,
            )
        elif self.fusion_mode == "qmf_sync":
            self.reliabilityFusion = FrameWiseAsymmetricQMFFusion(
                feature_dim=128,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
                min_weight=min_reliability,
                energy_temperature=energy_temperature,
                fusion_temperature=fusion_temperature,
            )
        elif self.fusion_mode == "qmf_sync_rank":
            self.reliabilityFusion = FrameWiseRankedAsymmetricQMFFusion(
                feature_dim=128,
                hidden_dim=reliability_hidden_dim,
                dropout=reliability_dropout,
                min_weight=min_reliability,
                energy_temperature=energy_temperature,
                fusion_temperature=fusion_temperature,
            )
        self.GRU = BGRU(128)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualEncoder(x)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(1).transpose(2, 3)     
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2, visual_logits=None,
                                     return_reliability=False,
                                     return_fusion_aux=False):
        fusion_aux = None
        if self.fusion_mode == "qmf":
            x, reliability = self.reliabilityFusion(x1, x2)
        elif self.fusion_mode == "qmf_sync":
            if visual_logits is None:
                raise ValueError(
                    "qmf_sync requires frame-wise visual classifier logits."
                )
            x, reliability = self.reliabilityFusion(
                x1, x2, visual_logits
            )
        elif self.fusion_mode == "qmf_sync_rank":
            if visual_logits is None:
                raise ValueError(
                    "qmf_sync_rank requires frame-wise visual classifier "
                    "logits."
                )
            x, reliability, visual_score = self.reliabilityFusion(
                x1, x2, visual_logits
            )
            fusion_aux = {'visual_score': visual_score}
        else:
            if x1.shape != x2.shape:
                raise ValueError(
                    "Audio and visual embeddings must have the same shape, "
                    "got %s and %s." % (tuple(x1.shape), tuple(x2.shape))
                )
            x = x1 + x2
            reliability = torch.ones(
                (*x1.shape[:-1], 2), device=x1.device, dtype=x1.dtype
            )
        x = self.GRU(x)   
        x = torch.reshape(x, (-1, 128))
        if return_fusion_aux:
            if not return_reliability:
                raise ValueError(
                    "return_fusion_aux requires return_reliability=True."
                )
            return x, reliability, fusion_aux
        if return_reliability:
            return x, reliability
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward(self, audioFeature, visualFeature, visual_logits=None,
                return_reliability=False):
        audioEmbed = self.forward_audio_frontend(audioFeature)
        visualEmbed = self.forward_visual_frontend(visualFeature)
        backend_output = self.forward_audio_visual_backend(
            audioEmbed,
            visualEmbed,
            visual_logits=visual_logits,
            return_reliability=return_reliability,
        )
        if return_reliability:
            outsAV, reliability = backend_output
        else:
            outsAV = backend_output
        outsV = self.forward_visual_backend(visualEmbed)

        if return_reliability:
            return outsAV, outsV, reliability
        return outsAV, outsV
