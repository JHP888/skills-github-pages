import math
import torch
from torch import nn
from einops import repeat

from .basic_layers import Transformer
from .bert import BertTextEncoder


class MLPConverter(nn.Module):
    """
    Simple feature converter used in TCMR:
    C_{s->d}(H_s)
    """
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class ScalarGate(nn.Module):
    """
    g_i^m = sigmoid(W_g^m * p_i^m + b_g^m)
    outputs a vector gate with size [B, D]
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(1, dim)

    def forward(self, quality_score):
        return torch.sigmoid(self.gate(quality_score))


class TCMRAggregator(nn.Module):
    """
    Aggregate reconstructed text from visual/audio when text is missing.
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

    def forward(self, feats, masks):
        """
        feats: list of [B, D]
        masks: list of [B, 1] in {0,1}
        """
        score_list = []
        for feat, mask in zip(feats, masks):
            s = self.score(feat)  # [B,1]
            s = s.masked_fill(mask <= 0, -1e4)
            score_list.append(s)

        scores = torch.cat(score_list, dim=1)          # [B, num_src]
        attn = torch.softmax(scores, dim=1)            # [B, num_src]
        stacked = torch.stack(feats, dim=1)            # [B, num_src, D]
        out = torch.sum(stacked * attn.unsqueeze(-1), dim=1)
        return out, attn


class LNLN(nn.Module):
    """
    Kept class name LNLN for compatibility with your current training code,
    but the inside is refactored toward TCDF:
      1) Feature Encoding
      2) DDL
      3) TCMR
      4) QAGF
    """
    def __init__(self, args):
        super(LNLN, self).__init__()
        self.args = args

        feat_cfg = args['model']['feature_extractor']
        pred_cfg = args['model'].get('prediction', {})
        ddl_cfg = args['model'].get('ddl', {})
        tcmr_cfg = args['model'].get('tcmr', {})
        qagf_cfg = args['model'].get('qagf', {})

        self.token_length = feat_cfg['token_length']
        self.hidden_dims = feat_cfg['hidden_dims']
        self.common_dim = self.hidden_dims[0]
        self.queue_size = ddl_cfg.get('queue_size', 128)
        self.queue_momentum = ddl_cfg.get('momentum', 0.99)
        self.queue_warmup = ddl_cfg.get('warmup_count', 64)
        self.eps = 1e-6

        # -------------------------
        # 1) Feature Encoding
        # -------------------------
        self.bertmodel = BertTextEncoder(
            use_finetune=True,
            transformers='bert',
            pretrained=feat_cfg['bert_pretrained']
        )

        self.proj_l = nn.Sequential(
            nn.Linear(feat_cfg['input_dims'][0], feat_cfg['hidden_dims'][0]),
            Transformer(
                num_frames=feat_cfg['input_length'][0],
                save_hidden=False,
                token_len=feat_cfg['token_length'][0],
                dim=feat_cfg['hidden_dims'][0],
                depth=feat_cfg['depth'],
                heads=feat_cfg['heads'],
                mlp_dim=feat_cfg['hidden_dims'][0]
            )
        )

        self.proj_v = nn.Sequential(
            nn.Linear(feat_cfg['input_dims'][1], feat_cfg['hidden_dims'][1]),
            Transformer(
                num_frames=feat_cfg['input_length'][1],
                save_hidden=False,
                token_len=feat_cfg['token_length'][1],
                dim=feat_cfg['hidden_dims'][1],
                depth=feat_cfg['depth'],
                heads=feat_cfg['heads'],
                mlp_dim=feat_cfg['hidden_dims'][1]
            )
        )

        self.proj_a = nn.Sequential(
            nn.Linear(feat_cfg['input_dims'][2], feat_cfg['hidden_dims'][2]),
            Transformer(
                num_frames=feat_cfg['input_length'][2],
                save_hidden=False,
                token_len=feat_cfg['token_length'][2],
                dim=feat_cfg['hidden_dims'][2],
                depth=feat_cfg['depth'],
                heads=feat_cfg['heads'],
                mlp_dim=feat_cfg['hidden_dims'][2]
            )
        )

        self.align_l = nn.Identity() if self.hidden_dims[0] == self.common_dim else nn.Linear(self.hidden_dims[0], self.common_dim)
        self.align_v = nn.Identity() if self.hidden_dims[1] == self.common_dim else nn.Linear(self.hidden_dims[1], self.common_dim)
        self.align_a = nn.Identity() if self.hidden_dims[2] == self.common_dim else nn.Linear(self.hidden_dims[2], self.common_dim)

        # -------------------------
        # 2) DDL buffers
        # -------------------------
        self.register_buffer("queue_t", torch.zeros(self.queue_size, self.common_dim))
        self.register_buffer("queue_v", torch.zeros(self.queue_size, self.common_dim))
        self.register_buffer("queue_a", torch.zeros(self.queue_size, self.common_dim))

        self.register_buffer("queue_t_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_v_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_a_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_t_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_v_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_a_count", torch.zeros(1, dtype=torch.long))

        # -------------------------
        # 3) TCMR
        # -------------------------
        conv_hidden = tcmr_cfg.get('hidden_dim', self.common_dim * 2)
        conv_dropout = tcmr_cfg.get('dropout', 0.1)

        self.c_v2t = MLPConverter(self.common_dim, conv_hidden, conv_dropout)
        self.c_t2v = MLPConverter(self.common_dim, conv_hidden, conv_dropout)
        self.c_a2t = MLPConverter(self.common_dim, conv_hidden, conv_dropout)
        self.c_t2a = MLPConverter(self.common_dim, conv_hidden, conv_dropout)

        self.text_aggregator = TCMRAggregator(self.common_dim)

        self.token_refiner_t = Transformer(
            num_frames=feat_cfg['token_length'][0],
            save_hidden=False,
            token_len=None,
            dim=self.common_dim,
            depth=1,
            heads=feat_cfg['heads'],
            mlp_dim=self.common_dim
        )
        self.token_refiner_v = Transformer(
            num_frames=feat_cfg['token_length'][1],
            save_hidden=False,
            token_len=None,
            dim=self.common_dim,
            depth=1,
            heads=feat_cfg['heads'],
            mlp_dim=self.common_dim
        )
        self.token_refiner_a = Transformer(
            num_frames=feat_cfg['token_length'][2],
            save_hidden=False,
            token_len=None,
            dim=self.common_dim,
            depth=1,
            heads=feat_cfg['heads'],
            mlp_dim=self.common_dim
        )

        # -------------------------
        # 4) QAGF
        # -------------------------
        self.gate_t = ScalarGate(self.common_dim)
        self.gate_v = ScalarGate(self.common_dim)
        self.gate_a = ScalarGate(self.common_dim)

        fusion_hidden = qagf_cfg.get('fusion_hidden_dim', self.common_dim * 2)
        fusion_dropout = qagf_cfg.get('dropout', 0.1)

        self.fusion_network = nn.Sequential(
            nn.Linear(self.common_dim * 3, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden, self.common_dim),
            nn.ReLU(inplace=True)
        )

        pred_out_dim = pred_cfg.get('out_dim', args['model'].get('output_dim', 1))
        self.pred_head = nn.Linear(self.common_dim, pred_out_dim)

    def _sequence_to_vector(self, x):
        return x.mean(dim=1)

    def _vector_to_tokens(self, x, token_len):
        return x.unsqueeze(1).repeat(1, token_len, 1)

    def _infer_mask_from_tensor(self, x):
        if x is None:
            return None
        dims = tuple(range(1, x.dim()))
        valid = (x.abs().sum(dim=dims) > 0).float().unsqueeze(1)
        return valid

    def _get_queue_and_ptr(self, modality):
        if modality == 't':
            return self.queue_t, self.queue_t_ptr, self.queue_t_count
        if modality == 'v':
            return self.queue_v, self.queue_v_ptr, self.queue_v_count
        if modality == 'a':
            return self.queue_a, self.queue_a_ptr, self.queue_a_count
        raise ValueError(f"Unsupported modality: {modality}")

    def _get_queue_stats(self, modality):
        queue, _, count = self._get_queue_and_ptr(modality)
        valid_count = int(count.item())
        if valid_count <= 1:
            mu = torch.zeros(self.common_dim, device=queue.device, dtype=queue.dtype)
            sigma = torch.ones(self.common_dim, device=queue.device, dtype=queue.dtype)
            return mu, sigma

        cur_queue = queue[:valid_count]
        mu = cur_queue.mean(dim=0)
        sigma = cur_queue.std(dim=0, unbiased=False).clamp_min(self.eps)
        return mu, sigma

    def _gaussian_quality(self, feat, mu, sigma):
        sigma_norm = torch.norm(sigma, p=2).clamp_min(self.eps)
        diff = feat - mu.unsqueeze(0)
        exponent = -(diff.pow(2).sum(dim=-1, keepdim=True)) / (2.0 * sigma_norm.pow(2))
        coeff = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_norm)
        quality = coeff * torch.exp(exponent)
        return quality

    @torch.no_grad()
    def _update_queue(self, modality, feats, quality_scores=None):
        queue, ptr, count = self._get_queue_and_ptr(modality)
        device = feats.device

        mu, sigma = self._get_queue_stats(modality)
        if quality_scores is None:
            quality_scores = self._gaussian_quality(feats, mu, sigma)

        valid_count = int(count.item())
        if valid_count > 1:
            existing_q = self._gaussian_quality(queue[:valid_count].to(device), mu.to(device), sigma.to(device))
            avg_quality = existing_q.mean()
            keep_mask = (quality_scores.squeeze(-1) > avg_quality).bool()
        else:
            keep_mask = torch.ones(feats.size(0), dtype=torch.bool, device=device)

        selected = feats[keep_mask].detach()
        if selected.numel() == 0:
            return

        for i in range(selected.size(0)):
            queue[ptr.item()] = selected[i].to(queue.device)
            ptr[0] = (ptr.item() + 1) % self.queue_size
            count[0] = min(count.item() + 1, self.queue_size)

    def _encode_modalities(self, vision, audio, language):
        h_l = self.proj_l(self.bertmodel(language))[:, :self.token_length[0]]
        h_v = self.proj_v(vision)[:, :self.token_length[1]]
        h_a = self.proj_a(audio)[:, :self.token_length[2]]

        h_l = self.align_l(h_l)
        h_v = self.align_v(h_v)
        h_a = self.align_a(h_a)

        z_l = self._sequence_to_vector(h_l)
        z_v = self._sequence_to_vector(h_v)
        z_a = self._sequence_to_vector(h_a)

        return h_v, h_a, h_l, z_v, z_a, z_l

    def _build_reconstructed_vectors(self, z_v, z_a, z_l, mask_v, mask_a, mask_t):
        v2t = self.c_v2t(z_v)
        a2t = self.c_a2t(z_a)

        rec_t_from_v = v2t
        rec_t_from_a = a2t
        rec_t_agg, text_attn = self.text_aggregator(
            feats=[rec_t_from_v, rec_t_from_a],
            masks=[mask_v, mask_a]
        )

        final_text_center = mask_t * z_l + (1.0 - mask_t) * rec_t_agg

        rec_v = self.c_t2v(final_text_center)
        rec_a = self.c_t2a(final_text_center)

        final_v = mask_v * z_v + (1.0 - mask_v) * rec_v
        final_a = mask_a * z_a + (1.0 - mask_a) * rec_a
        final_t = final_text_center

        rec_dict = {
            'rec_t_from_v': rec_t_from_v,
            'rec_t_from_a': rec_t_from_a,
            'rec_t': rec_t_agg,
            'rec_v': rec_v,
            'rec_a': rec_a,
            'text_attn': text_attn
        }

        final_dict = {
            'final_t': final_t,
            'final_v': final_v,
            'final_a': final_a
        }

        return rec_dict, final_dict

    def _build_reconstructed_tokens(self, final_dict):
        rec_t_tokens = self._vector_to_tokens(final_dict['final_t'], self.token_length[0])
        rec_v_tokens = self._vector_to_tokens(final_dict['final_v'], self.token_length[1])
        rec_a_tokens = self._vector_to_tokens(final_dict['final_a'], self.token_length[2])

        rec_t_tokens = self.token_refiner_t(rec_t_tokens)
        rec_v_tokens = self.token_refiner_v(rec_v_tokens)
        rec_a_tokens = self.token_refiner_a(rec_a_tokens)

        return rec_v_tokens, rec_a_tokens, rec_t_tokens

    def _compute_quality_scores(self, final_dict):
        """
        Warmup DDL queue before using Gaussian-quality scores.
        Before enough samples are accumulated, use uniform quality = 1.
        """
        min_count = min(
            int(self.queue_t_count.item()),
            int(self.queue_v_count.item()),
            int(self.queue_a_count.item())
        )

        if min_count < self.queue_warmup:
            p_t = torch.ones(final_dict['final_t'].size(0), 1, device=final_dict['final_t'].device, dtype=final_dict['final_t'].dtype)
            p_v = torch.ones(final_dict['final_v'].size(0), 1, device=final_dict['final_v'].device, dtype=final_dict['final_v'].dtype)
            p_a = torch.ones(final_dict['final_a'].size(0), 1, device=final_dict['final_a'].device, dtype=final_dict['final_a'].dtype)

            mu_t = torch.zeros(self.common_dim, device=final_dict['final_t'].device, dtype=final_dict['final_t'].dtype)
            mu_v = torch.zeros(self.common_dim, device=final_dict['final_v'].device, dtype=final_dict['final_v'].dtype)
            mu_a = torch.zeros(self.common_dim, device=final_dict['final_a'].device, dtype=final_dict['final_a'].dtype)

            sigma_t = torch.ones(self.common_dim, device=final_dict['final_t'].device, dtype=final_dict['final_t'].dtype)
            sigma_v = torch.ones(self.common_dim, device=final_dict['final_v'].device, dtype=final_dict['final_v'].dtype)
            sigma_a = torch.ones(self.common_dim, device=final_dict['final_a'].device, dtype=final_dict['final_a'].dtype)
        else:
            mu_t, sigma_t = self._get_queue_stats('t')
            mu_v, sigma_v = self._get_queue_stats('v')
            mu_a, sigma_a = self._get_queue_stats('a')

            p_t = self._gaussian_quality(final_dict['final_t'], mu_t, sigma_t)
            p_v = self._gaussian_quality(final_dict['final_v'], mu_v, sigma_v)
            p_a = self._gaussian_quality(final_dict['final_a'], mu_a, sigma_a)

        return {
            'mu_t': mu_t, 'sigma_t': sigma_t, 'p_t': p_t,
            'mu_v': mu_v, 'sigma_v': sigma_v, 'p_v': p_v,
            'mu_a': mu_a, 'sigma_a': sigma_a, 'p_a': p_a,
        }

    def forward(self, complete_input, incomplete_input, modality_mask=None, update_queue=True):
        vision, audio, language = complete_input
        vision_m, audio_m, language_m = incomplete_input

        if modality_mask is None:
            mask_v = self._infer_mask_from_tensor(vision_m)
            mask_a = self._infer_mask_from_tensor(audio_m)
            mask_t = self._infer_mask_from_tensor(language_m)
        else:
            mask_v = modality_mask['vision']
            mask_a = modality_mask['audio']
            mask_t = modality_mask['language']

            if mask_v.dim() == 1:
                mask_v = mask_v.unsqueeze(1)
            if mask_a.dim() == 1:
                mask_a = mask_a.unsqueeze(1)
            if mask_t.dim() == 1:
                mask_t = mask_t.unsqueeze(1)

            mask_v = mask_v.float()
            mask_a = mask_a.float()
            mask_t = mask_t.float()

        h_v_m, h_a_m, h_l_m, z_v_m, z_a_m, z_l_m = self._encode_modalities(
            vision_m, audio_m, language_m
        )

        rec_dict, final_dict = self._build_reconstructed_vectors(
            z_v=z_v_m,
            z_a=z_a_m,
            z_l=z_l_m,
            mask_v=mask_v,
            mask_a=mask_a,
            mask_t=mask_t
        )

        rec_v_tokens, rec_a_tokens, rec_t_tokens = self._build_reconstructed_tokens(final_dict)

        quality_pack = self._compute_quality_scores(final_dict)
        mu_t, sigma_t, p_t = quality_pack['mu_t'], quality_pack['sigma_t'], quality_pack['p_t']
        mu_v, sigma_v, p_v = quality_pack['mu_v'], quality_pack['sigma_v'], quality_pack['p_v']
        mu_a, sigma_a, p_a = quality_pack['mu_a'], quality_pack['sigma_a'], quality_pack['p_a']

        g_t = self.gate_t(p_t)
        g_v = self.gate_v(p_v)
        g_a = self.gate_a(p_a)

        hat_t = final_dict['final_t'] * g_t
        hat_v = final_dict['final_v'] * g_v
        hat_a = final_dict['final_a'] * g_a

        fusion_input = torch.cat([hat_v, hat_t, hat_a], dim=-1)
        fused_feat = self.fusion_network(fusion_input)
        output = self.pred_head(fused_feat)

        rec_feats = None
        complete_feats = None
        complete_feature_dict = None

        if (vision is not None) and (audio is not None) and (language is not None):
            h_v_c, h_a_c, h_l_c, z_v_c, z_a_c, z_l_c = self._encode_modalities(
                vision, audio, language
            )
            complete_feats = torch.cat([h_a_c, h_v_c, h_l_c], dim=1)
            rec_feats = torch.cat([rec_a_tokens, rec_v_tokens, rec_t_tokens], dim=1)

            complete_feature_dict = {
                'h_v': h_v_c,
                'h_a': h_a_c,
                'h_l': h_l_c,
                'z_v': z_v_c,
                'z_a': z_a_c,
                'z_l': z_l_c,
            }

        if self.training and update_queue:
            self._update_queue('t', final_dict['final_t'], p_t)
            self._update_queue('v', final_dict['final_v'], p_v)
            self._update_queue('a', final_dict['final_a'], p_a)

        return {
            'sentiment_preds': output,
            'rec_feats': rec_feats,
            'complete_feats': complete_feats,

            'modality_masks': {
                'vision': mask_v,
                'audio': mask_a,
                'language': mask_t,
            },

            'encoded_incomplete': {
                'h_v': h_v_m,
                'h_a': h_a_m,
                'h_l': h_l_m,
                'z_v': z_v_m,
                'z_a': z_a_m,
                'z_l': z_l_m,
            },

            'encoded_complete': complete_feature_dict,

            'reconstructed': rec_dict,
            'final_features': final_dict,

            'reconstructed_tokens': {
                'rec_v_tokens': rec_v_tokens,
                'rec_a_tokens': rec_a_tokens,
                'rec_t_tokens': rec_t_tokens,
            },

            'ddl_stats': {
                'mu_t': mu_t,
                'sigma_t': sigma_t,
                'mu_v': mu_v,
                'sigma_v': sigma_v,
                'mu_a': mu_a,
                'sigma_a': sigma_a,
            },

            'quality_scores': {
                'p_t': p_t,
                'p_v': p_v,
                'p_a': p_a,
            },
            'gates': {
                'g_t': g_t,
                'g_v': g_v,
                'g_a': g_a,
            },
            'gated_features': {
                'hat_t': hat_t,
                'hat_v': hat_v,
                'hat_a': hat_a,
            },
            'fused_feat': fused_feat,
        }


def build_model(args):
    return LNLN(args)
