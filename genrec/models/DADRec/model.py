# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from genrec.model import AbstractModel
from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer
from .ablate_decode import decode_ablate_confidence


def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or "layernorm").lower() == "rmsnorm":
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        # Initialize weights
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, x, attention_mask=None, key_value=None, past_key_value=None, use_cache=False, is_decoder_self_attn=False):
        B, T, C = x.size()

        if key_value is not None:
            # Cross attention: Q from x, K,V from key_value
            q = self.qkv(x)[:, :, :self.emb_dim]  # Only take Q part
            k, v = key_value.chunk(2, dim=-1)  # key_value should be [B, T_enc, 2*emb_dim]
            T_kv = k.size(1)
        else:
            # Self attention
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T

        # Handle past key-value cache for incremental decoding
        if past_key_value is not None and use_cache and is_decoder_self_attn:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
            T_kv = k.size(1)

        # 保存拼接后的完整k和v用于cache（在reshape之前）
        k_for_cache = k
        v_for_cache = v

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T_kv, head_dim)

        # Scaled dot-product attention
        scale = 1.0 / (self.head_dim ** 0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, n_head, T, T_kv)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: (B, T, T_kv) or (B, 1, T, T_kv)
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # Add head dimension
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = torch.matmul(att, v)  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, emb_dim)

        # Output projection
        y = self.resid_dropout(self.proj(y))

        # Prepare cache for next iteration - 保存原始的3维k和v
        present_key_value = (k_for_cache, v_for_cache) if use_cache else None

        return y, present_key_value


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, attention_mask=None):
        # 自注意力 + 残差连接（非decoder自注意力）
        attn_output, _ = self.attn(self.ln_1(x), attention_mask=attention_mask, is_decoder_self_attn=False)
        x = x + attn_output
        
        # 前馈网络 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):

    def __init__(self, emb_dim, n_head, n_inner, attn_drop=0.1, resid_drop=0.1, 
                 act='gelu', norm_type='layernorm', norm_eps=1e-6):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_3 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self, x, encoder_hidden=None, attention_mask=None, 
                past_key_value=None, use_cache=False, cross_key_value=None):
        # 修改：去除因果掩码，因为diffusion模型不需要严格的序列顺序
        # 自注意力（不使用因果掩码）
        self_past_kv = None
        cross_past_kv = None
        if past_key_value is not None:
            if len(past_key_value) >= 1:
                self_past_kv = past_key_value[0]
            if len(past_key_value) >= 2:
                cross_past_kv = past_key_value[1]
        
        attn_output, present_key_value = self.self_attn(
            self.ln_1(x), 
            attention_mask=None,  # 不使用因果掩码
            past_key_value=self_past_kv,
            use_cache=use_cache,
            is_decoder_self_attn=True
        )
        x = x + attn_output

        # 交叉注意力
        if encoder_hidden is not None:
            if cross_key_value is not None:
                # 🚀 使用预计算的KV，避免重复计算
                encoder_kv = cross_key_value
            else:
                # 兼容旧逻辑：重新计算（仅用于非优化路径）
                encoder_kv = torch.cat([encoder_hidden, encoder_hidden], dim=-1)  # Concat K and V
            
            cross_attn_output, cross_present = self.cross_attn(
                self.ln_2(x),
                key_value=encoder_kv,
                past_key_value=cross_past_kv,
                use_cache=use_cache
            )
            x = x + cross_attn_output
            
            if use_cache:
                present_key_value = (present_key_value, cross_present)
        
        # 前馈网络
        x = x + self.mlp(self.ln_3(x))
        
        return_dict = {}
        return_dict['hidden_states'] = x
        if use_cache:
            return_dict['present_key_value'] = present_key_value
        
        return return_dict


class ModelOutput:

    def __init__(self):
        self.loss = None
        self.logits = None
        self.hidden_states = None
        self.past_key_values = None
        self.drift_gate_probs = None
        self.drift_probs = None
        self.drift_scores = None
        self.drift_gate_loss = None
        self.cadd_aux_loss = None


class DriftExpertAdapter(nn.Module):
    def __init__(self, emb_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, emb_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class DADRec(AbstractModel):

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        super().__init__(config, dataset, tokenizer)
        
        self.config = config
        self.tokenizer = tokenizer
        self.n_digit = config['n_digit']
        self.codebook_size = config['codebook_size']
        self.vocab_size = tokenizer.vocab_size
        
        # Model dimensions
        self.n_embd = config['n_embd']
        self.n_head = config['n_head']
        self.n_inner = config['n_inner']
        self.dropout = config['dropout']
        
        # Encoder layers
        self.encoder_n_layer = config['encoder_n_layer']
        self.decoder_n_layer = config['decoder_n_layer']
        
        # Normalization configuration
        self.norm_type = (config.get('norm_type', 'layernorm') or 'layernorm').lower()
        self.norm_eps  = float(config.get('norm_eps', 1e-6 if self.norm_type=='rmsnorm' else 1e-5))
        
        # ==== 读取新策略 ====
        self.masking_strategy = config.get('masking_strategy', 'random')  # random | sequential
        
        if self.masking_strategy == 'sequential':
            # 连贯多视图
            seq_cfg = config.get('sequential_steps', 'auto')
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            assert 1 <= self.seq_steps <= self.n_digit, \
                f"sequential_steps must be 1~{self.n_digit}, got {self.seq_steps}"
            
            # 新增：多路径支持
            self.sequential_paths = config.get('sequential_paths', 1)
            assert self.sequential_paths >= 1, \
                f"sequential_paths must be >= 1, got {self.sequential_paths}"
            
            self.augment_factor = self.seq_steps * self.sequential_paths  # 更新计算方式
            print(f"[MODEL] ▶ use SEQUENTIAL views: steps={self.seq_steps}, "
                  f"paths={self.sequential_paths}, augment_factor={self.augment_factor}")
            # 移除不必要的mask_probs设置，节省内存
            self.mask_probs = None
        elif self.masking_strategy == 'guided':
            # 置信度引导的连贯多视图（每个batch由模型决定揭示顺序）
            guided_cfg = config.get('guided_steps', 'auto')
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            # 限制最多 4 步（你现在 n_digit=4，因此刚好 4）
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = config.get('guided_conf_metric', 'msp')
            assert self.guided_conf_metric in ('msp', 'entropy'), \
                f"guided_conf_metric must be one of ['msp','entropy'], got {self.guided_conf_metric}"
            # 新增：选择揭示“最有把握(most)”或“最不把握(least)”的位置
            self.guided_select = config.get('guided_select', 'most')
            assert self.guided_select in ('most', 'least'), \
                f"guided_select must be one of ['most','least'], got {self.guided_select}"
            self.augment_factor = self.guided_steps
            print(f"[MODEL] ▶ GUIDED: steps={self.guided_steps}, metric={self.guided_conf_metric}, "
                  f"select={self.guided_select}, augment_factor={self.augment_factor}")
            self.mask_probs = None
        else:
            # 旧的随机掩码分支（保持原逻辑）
            # Diffusion specific parameters - 多概率掩码配置
            # 新增：支持按区间随机采样单一掩码概率，并可通过augment_factor重复该概率
            self.mask_prob_random = bool(config.get('mask_prob_random', False))
            if self.mask_prob_random:
                low = float(config.get('mask_prob_random_min', 0.0))
                high = float(config.get('mask_prob_random_max', 1.0))
                if not (0.0 <= low <= high <= 1.0):
                    raise ValueError(
                        f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                    )
                sampled_prob = float(np.random.uniform(low, high))
                # 按需求：开启随机掩码概率时不做多视图扩增
                self.augment_factor = 1
                self.mask_probs = [sampled_prob]
                self.sampled_mask_prob = sampled_prob
                print(
                    f"[MODEL] Using RANDOMLY-SAMPLED masking prob: {sampled_prob:.4f} (range [{low}, {high}]); disable multi-view (augment_factor=1)"
                )
            elif 'mask_probs' in config and config['mask_probs'] is not None:
                # 新方式：直接指定多个掩码概率
                mask_probs_raw = config['mask_probs']
                
                if isinstance(mask_probs_raw, str):
                    # 字符串格式："1.0,0.75,0.5,0.25"
                    self.mask_probs = [float(p.strip()) for p in mask_probs_raw.split(',')]
                elif isinstance(mask_probs_raw, (list, tuple)):
                    # 列表或元组格式：[1.0, 0.75, 0.5, 0.25]
                    self.mask_probs = [float(p) for p in mask_probs_raw]
                elif isinstance(mask_probs_raw, (int, float)):
                    # 单个数值，转换为单元素列表
                    self.mask_probs = [float(mask_probs_raw)]
                else:
                    # 其他类型，尝试转换为字符串再解析
                    try:
                        mask_probs_str = str(mask_probs_raw)
                        self.mask_probs = [float(p.strip()) for p in mask_probs_str.split(',')]
                    except (ValueError, AttributeError):
                        raise ValueError(f"Cannot parse mask_probs: {mask_probs_raw} (type: {type(mask_probs_raw)}). "
                                       "Expected string like '1.0,0.75,0.5,0.25' or list like [1.0, 0.75, 0.5, 0.25]")
                
                self.augment_factor = len(self.mask_probs)  # 自动设置增强倍数
                print(f"[MODEL] Using multi-probability masking: {self.mask_probs}")
            else:
                # 旧方式：单一掩码概率 + 增强倍数
                mask_prob = config.get('mask_prob', 0.5)
                self.augment_factor = config.get('augment_factor', 4)
                self.mask_probs = [float(mask_prob)] * self.augment_factor  # 重复相同概率
                print(f"[MODEL] Using single-probability masking: {mask_prob} x {self.augment_factor}")
        
        # 验证掩码概率的有效性（仅对random策略有效）
        if self.masking_strategy == 'random' and self.mask_probs is not None:
            for i, prob in enumerate(self.mask_probs):
                if not (0.0 <= prob <= 1.0):
                    raise ValueError(f"mask_probs[{i}] = {prob} is not in valid range [0.0, 1.0]")
        
        # Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.n_embd)
        
        # 添加与RPG_ED一致的item_mlp：将n_digit个SID token压缩为1个token
        self.item_mlp = nn.Sequential(
            nn.Linear(self.n_digit * self.n_embd, self.n_embd),  # n_digit×d → d
            nn.ReLU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        
        # 新增：掩码嵌入表，用于表示被掩码的位置
        self.mask_emb_table = nn.Embedding(self.n_digit, self.n_embd)

        # CADD-lite: use a continuous semantic hint instead of a purely static mask state.
        cadd_cfg = config.get('cadd', {}) or {}
        self.cadd_enabled = bool(cadd_cfg.get('enabled', False))
        self.cadd_hint_scale = float(cadd_cfg.get('hint_scale', 0.5))
        self.cadd_aux_loss_weight = float(cadd_cfg.get('aux_loss_weight', 0.05))
        self.cadd_aux_target = str(cadd_cfg.get('aux_target', 'sid_embedding')).lower()
        self.cadd_semantic_loss = str(cadd_cfg.get('semantic_loss', 'cosine')).lower()
        self.cadd_semantic_pool = str(cadd_cfg.get('semantic_pool', 'masked_mean')).lower()
        self.cadd_code_prob_temperature = float(cadd_cfg.get('code_prob_temperature', 1.0))
        self.cadd_hint_injection = str(cadd_cfg.get('hint_injection', 'continuous')).lower()
        self.cadd_opq_aux_targets = (
            'opq_subvector', 'opq_subvectors', 'pre_opq_subvector', 'pre_opq_subvectors'
        )
        self.cadd_preopq_aux_targets = ('pre_opq', 'pre_opq_semantic', 'sent_emb', 'semantic')
        self.cadd_semantic_dim = int(cadd_cfg.get(
            'semantic_dim',
            config.get('sent_emb_pca', config.get('sent_emb_dim', self.n_embd))
        ))
        self.cadd_opq_subvector_dim = int(cadd_cfg.get(
            'opq_subvector_dim',
            max(1, self.cadd_semantic_dim // self.n_digit)
        ))
        if self.cadd_aux_target not in ('sid_embedding',) + self.cadd_preopq_aux_targets + self.cadd_opq_aux_targets:
            raise ValueError(f"Unsupported cadd.aux_target: {self.cadd_aux_target}")
        if self.cadd_semantic_loss not in ('cosine', 'mse'):
            raise ValueError(f"Unsupported cadd.semantic_loss: {self.cadd_semantic_loss}")
        if self.cadd_semantic_pool not in ('masked_mean', 'mean'):
            raise ValueError(f"Unsupported cadd.semantic_pool: {self.cadd_semantic_pool}")
        if self.cadd_code_prob_temperature <= 0.0:
            raise ValueError("cadd.code_prob_temperature must be > 0")
        if self.cadd_hint_injection not in ('continuous', 'soft_quantized_sid'):
            raise ValueError(f"Unsupported cadd.hint_injection: {self.cadd_hint_injection}")
        if self.cadd_hint_injection == 'soft_quantized_sid' and self.cadd_aux_target not in self.cadd_opq_aux_targets:
            raise ValueError("cadd.hint_injection=soft_quantized_sid requires cadd.aux_target=opq_subvector")
        self.cadd_hint_dropout = nn.Dropout(float(cadd_cfg.get('hint_dropout', self.dropout)))
        self.cadd_digit_emb = nn.Embedding(self.n_digit, self.n_embd)
        self.cadd_hint_mlp = nn.Sequential(
            nn.Linear(3 * self.n_embd, self.n_embd),
            nn.GELU(),
            nn.Linear(self.n_embd, self.n_embd)
        )
        self.cadd_sem_prior_mlp = None
        self.cadd_sem_to_hidden = None
        self.cadd_hint_from_sem_mlp = None
        self.cadd_opq_subvec_mlp = None
        self.cadd_opq_subvec_to_hidden = None
        if self.cadd_aux_target in self.cadd_opq_aux_targets:
            self.cadd_opq_subvec_mlp = nn.Sequential(
                nn.Linear(3 * self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, self.cadd_opq_subvector_dim)
            )
            self.cadd_opq_subvec_to_hidden = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.cadd_opq_subvector_dim),
                    nn.Linear(self.cadd_opq_subvector_dim, self.n_embd),
                )
                for _ in range(self.n_digit)
            ])
        elif self.cadd_aux_target in self.cadd_preopq_aux_targets:
            self.cadd_sem_prior_mlp = nn.Sequential(
                nn.Linear(2 * self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, self.cadd_semantic_dim)
            )
            self.cadd_sem_to_hidden = nn.Sequential(
                nn.LayerNorm(self.cadd_semantic_dim),
                nn.Linear(self.cadd_semantic_dim, self.n_embd),
                nn.GELU()
            )
            self.cadd_hint_from_sem_mlp = nn.Sequential(
                nn.Linear(4 * self.n_embd, self.n_embd),
                nn.GELU(),
                nn.Linear(self.n_embd, self.n_embd)
            )

        sid_code_centroids = getattr(tokenizer, 'sid_code_centroids', None)
        if sid_code_centroids is not None:
            sid_code_centroids = torch.as_tensor(sid_code_centroids, dtype=torch.float32)
            centroid_dim = self.cadd_opq_subvector_dim if self.cadd_aux_target in self.cadd_opq_aux_targets else self.cadd_semantic_dim
            expected_shape = (self.n_digit, self.codebook_size, centroid_dim)
            if tuple(sid_code_centroids.shape) != expected_shape:
                raise ValueError(
                    f"SID code centroids shape {tuple(sid_code_centroids.shape)} != {expected_shape}"
                )
            self.register_buffer("cadd_code_centroids", sid_code_centroids, persistent=True)
        else:
            self.cadd_code_centroids = None
        if self.cadd_hint_injection == 'soft_quantized_sid' and self.cadd_code_centroids is None:
            raise ValueError("cadd.hint_injection=soft_quantized_sid requires fixed OPQ/PQ SID code centroids")

        # Drift-conditioned lightweight MoE. Routing is computed from CADD hint SID-code probabilities.
        drift_moe_cfg = config.get('drift_moe', {}) or {}
        self.drift_moe_enabled = bool(drift_moe_cfg.get('enabled', False))
        self.drift_moe_n_experts = int(drift_moe_cfg.get('n_experts', 4))
        if self.drift_moe_n_experts != 4:
            raise ValueError(f"drift_moe.n_experts must be 4 for four drift buckets, got {self.drift_moe_n_experts}")
        self.drift_moe_bottleneck = int(drift_moe_cfg.get('bottleneck', 32))
        self.drift_moe_gate_loss_weight = float(
            drift_moe_cfg.get('drift_loss_weight', drift_moe_cfg.get('gate_loss_weight', 0.05))
        )
        self.drift_moe_recency_gamma = float(drift_moe_cfg.get('recency_gamma', 0.9))
        self.drift_moe_digit_weights = drift_moe_cfg.get('digit_weights', None)
        self.drift_moe_hint_code_temperature = float(drift_moe_cfg.get('hint_code_temperature', 1.0))
        self.drift_moe_bucket_temperature = float(drift_moe_cfg.get('bucket_temperature', 0.03))
        self.drift_moe_bucket_strategy = str(
            drift_moe_cfg.get('bucket_strategy', 'train_quantile_full')
        ).lower()
        self.drift_moe_full_drift_eps = float(drift_moe_cfg.get('full_drift_eps', 1e-12))
        self.drift_moe_bucket_fit_batch_size = int(drift_moe_cfg.get('bucket_fit_batch_size', 8192))
        self.drift_moe_novel_topk = int(drift_moe_cfg.get('novel_topk', drift_moe_cfg.get('code_sim_topk', 8)))
        self.drift_moe_evidence_temperature = float(drift_moe_cfg.get('evidence_temperature', 1.0))
        if self.drift_moe_hint_code_temperature <= 0.0:
            raise ValueError("drift_moe.hint_code_temperature must be > 0")
        if self.drift_moe_bucket_temperature <= 0.0:
            raise ValueError("drift_moe.bucket_temperature must be > 0")
        if self.drift_moe_bucket_fit_batch_size <= 0:
            raise ValueError("drift_moe.bucket_fit_batch_size must be > 0")
        if self.drift_moe_novel_topk <= 0:
            raise ValueError("drift_moe.novel_topk must be > 0")
        if self.drift_moe_evidence_temperature <= 0.0:
            raise ValueError("drift_moe.evidence_temperature must be > 0")
        self.register_buffer(
            "drift_moe_bucket_edges",
            torch.tensor([1.0 / 3.0, 2.0 / 3.0, 1.0], dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "drift_moe_bucket_centers",
            torch.tensor([1.0 / 6.0, 0.5, 5.0 / 6.0, 1.0], dtype=torch.float32),
            persistent=True,
        )
        self.drift_moe_buckets_fitted = False
        self.drift_moe_alpha = nn.Parameter(torch.tensor(float(drift_moe_cfg.get('alpha_init', 0.0))))
        self.drift_experts = nn.ModuleList([
            DriftExpertAdapter(
                emb_dim=self.n_embd,
                bottleneck=self.drift_moe_bottleneck,
                dropout=float(drift_moe_cfg.get('expert_dropout', self.dropout)),
            )
            for _ in range(self.drift_moe_n_experts)
        ])
        
        # 位置编码：只为encoder添加绝对位置编码（与RPG_ED一致）
        self.max_history_len = config.get('max_history_len', 50)  # 从config读取，默认50
        self.pos_emb_enc = nn.Embedding(self.max_history_len, self.n_embd)
        # 移除decoder位置编码，decoder只使用掩码
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.encoder_n_layer)
        ])
        
        # Decoder blocks  
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self.n_embd, self.n_head, self.n_inner,
                config['attn_pdrop'], config['resid_pdrop'],
                act='gelu',
                norm_type=self.norm_type, norm_eps=self.norm_eps
            )
            for _ in range(self.decoder_n_layer)
        ])
        
        # Layer normalization
        self.ln_f = make_norm(self.norm_type, self.n_embd, self.norm_eps)
        
        # -- 1.1 删除旧的独立 heads，改为共享 embedding dot-product --
        share_out = self.config.get('share_decoder_output_embedding', True)
        if share_out:
            # 直接 weight-tying，不新增参数
            self.output_adapter = nn.Identity()
            print(f"[DADRec] Using shared embedding dot-product output layer")
        else:
            # 若以后要回滚到独立 head，用这一行
            self.output_adapter = nn.Linear(self.n_embd, self.n_embd, bias=False)
            print(f"[DADRec] Using independent Linear output adapter")
        # -------------------------------------------------------------
        
        # Dropout
        self.drop = nn.Dropout(self.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)

        # 当启用 ablation 时，自动注入 confidence_s1/s2/s3 模式以确保评估阶段会跑三种
        ab_cfg = self.config.get('ablate_decode', {}) or {}
        if bool(ab_cfg.get('enabled', False)):
            modes = list(self.config.get('beam_search_modes', []) or [])
            to_add = ['confidence_s1', 'confidence_s2', 'confidence_s3']
            if 'confidence' in modes:
                base = modes.index('confidence')
                for i, m in enumerate(to_add, 1):
                    if m not in modes:
                        modes.insert(base + i, m)
            else:
                for m in reversed(to_add):
                    if m not in modes:
                        modes.insert(0, m)
            self.config['beam_search_modes'] = modes

    def resample_mask_prob_if_needed(self):
        """
        当采用 random + mask_prob_random=true 时，在训练的每个 epoch 开始调用，
        重新从设定区间采样一次掩码概率，并更新当前 epoch 使用的掩码率与loss缩放。
        """
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            low = float(getattr(self, 'mask_prob_random_min', 0.0)) if hasattr(self, 'mask_prob_random_min') else float(self.config.get('mask_prob_random_min', 0.0))
            high = float(getattr(self, 'mask_prob_random_max', 1.0)) if hasattr(self, 'mask_prob_random_max') else float(self.config.get('mask_prob_random_max', 1.0))
            if not (0.0 <= low <= high <= 1.0):
                raise ValueError(
                    f"mask_prob_random_min/max must satisfy 0.0 <= min <= max <= 1.0, got min={low}, max={high}"
                )
            sampled_prob = float(np.random.uniform(low, high))
            self.mask_probs = [sampled_prob]  # 单视图
            self.sampled_mask_prob = sampled_prob
            print(f"[MODEL] [Epoch-Resample] RANDOM masking prob resampled to {sampled_prob:.4f} (range [{low}, {high}]); augment_factor=1")

    def set_masking_mode(self, strategy: str, **kw):
        """
        训练过程中的热切换：
        - strategy: 'guided' | 'sequential' | 'random'
        - kw: 对应策略需要的超参（见下）
        """
        self.masking_strategy = strategy

        if strategy == 'sequential':
            # steps
            seq_cfg = kw.get('sequential_steps', self.config.get('sequential_steps', 'auto'))
            self.seq_steps = self.n_digit if seq_cfg in (None, 'auto') else int(seq_cfg)
            self.sequential_paths = int(kw.get('sequential_paths', self.config.get('sequential_paths', 1)))
            self.augment_factor = self.seq_steps * self.sequential_paths
            self.mask_probs = None
            print(f"[SCHEDULE] → SEQUENTIAL: steps={self.seq_steps}, paths={self.sequential_paths}, augment_factor={self.augment_factor}")

        elif strategy == 'guided':
            guided_cfg = kw.get('guided_steps', self.config.get('guided_steps', 'auto'))
            self.guided_steps = self.n_digit if guided_cfg in (None, 'auto') else int(guided_cfg)
            self.guided_steps = min(self.guided_steps, self.n_digit, 4)
            self.guided_conf_metric = kw.get('guided_conf_metric', self.config.get('guided_conf_metric', 'msp'))
            self.guided_select = kw.get('guided_select', self.config.get('guided_select', 'least'))
            # 注意：forward 里读 self.config['guided_refresh_each_step']，所以要同步回 config
            self.config['guided_refresh_each_step'] = bool(kw.get(
                'guided_refresh_each_step',
                self.config.get('guided_refresh_each_step', False)
            ))
            self.augment_factor = self.guided_steps
            self.mask_probs = None
            print(f"[SCHEDULE] → GUIDED({self.guided_select}): steps={self.guided_steps}, metric={self.guided_conf_metric}, refresh={self.config['guided_refresh_each_step']}, augment_factor={self.augment_factor}")

        elif strategy == 'random':
            # 保留旧逻辑，按需覆盖
            self.mask_prob_random = bool(kw.get('mask_prob_random', self.config.get('mask_prob_random', False)))
            if self.mask_prob_random:
                self.mask_probs = [float(np.random.uniform(
                    float(kw.get('mask_prob_random_min', self.config.get('mask_prob_random_min', 0.0))),
                    float(kw.get('mask_prob_random_max', self.config.get('mask_prob_random_max', 1.0)))
                ))]
                self.augment_factor = 1
            else:
                if 'mask_probs' in kw and kw['mask_probs'] is not None:
                    self.mask_probs = [float(p) for p in (kw['mask_probs'] if isinstance(kw['mask_probs'], (list, tuple)) else str(kw['mask_probs']).split(','))]
                    self.augment_factor = len(self.mask_probs)
                else:
                    mp = float(kw.get('mask_prob', self.config.get('mask_prob', 0.5)))
                    af = int(kw.get('augment_factor', self.config.get('augment_factor', 4)))
                    self.mask_probs = [mp] * af
                    self.augment_factor = af
            print(f"[SCHEDULE] → RANDOM: mask_probs={self.mask_probs}, augment_factor={self.augment_factor}")
        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

    def _compute_digit_logits(self, hidden_last, digit):
        """
        使用共享embedding的dot-product计算logits
        
        Args:
            hidden_last: (B, d_model) - decoder输出的隐藏状态
            digit: 0..n_digit-1 - 要预测的digit位置
            
        Returns:
            logits: (B, codebook_size) - 预测logits
        """
        if digit is None:
            raise ValueError("digit参数不能为None，必须指定要计算的codebook位置")
        
        if digit >= self.n_digit:
            raise ValueError(f"digit={digit} 超出范围，应该在 [0, {self.n_digit-1}]")
        
        # 2.1 取出 embedding matrix 的相应切片
        # token ID 布局 = [PAD, BOS, EOS, digit0 256 个, digit1 256 个, ...]
        start = self.tokenizer.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size  # 不含 end
        # shape: (codebook_size, d_model)
        E_sub = self.embedding.weight[start:end]
        
        # 2.2 optional adapter
        h = self.output_adapter(hidden_last)  # (B, d_model)
        
        # 2.3 dot-product 得 logits
        # (B, d_model) @ (d_model, codebook_size).T → (B, codebook_size)
        logits = torch.matmul(h, E_sub.t())
        
        return logits

    def _codebook_ids_to_token_emb(self, codebook_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert per-digit codebook ids to decoder token embeddings.

        Args:
            codebook_ids: [B, n_digit]
        Returns:
            token_emb: [B, n_digit, n_embd]
        """
        B, n_digit = codebook_ids.shape
        token_emb = torch.zeros(B, n_digit, self.n_embd, device=codebook_ids.device)
        for d in range(n_digit):
            token_ids = codebook_ids[:, d] + self.tokenizer.sid_offset + d * self.codebook_size
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            token_emb[:, d, :] = self.embedding(token_ids)
        return token_emb

    def _compute_cadd_hint(self, token_emb: torch.Tensor, mask_positions: torch.Tensor, encoder_hidden: torch.Tensor):
        """
        Build a continuous hint for masked digits from the current partial SID and history encoding.
        """
        B, n_digit, _ = token_emb.shape
        visible_mask = (1.0 - mask_positions.float()).unsqueeze(-1)
        visible_count = visible_mask.sum(dim=1).clamp_min(1.0)
        partial_summary = (token_emb * visible_mask).sum(dim=1) / visible_count
        encoder_summary = encoder_hidden.mean(dim=1)

        digit_ids = torch.arange(n_digit, device=token_emb.device).unsqueeze(0).expand(B, -1)
        digit_emb = self.cadd_digit_emb(digit_ids)

        cadd_sem_prior = None
        cadd_opq_subvectors = None
        if self.cadd_aux_target in self.cadd_opq_aux_targets:
            subvec_input = torch.cat([
                partial_summary.unsqueeze(1).expand(-1, n_digit, -1),
                encoder_summary.unsqueeze(1).expand(-1, n_digit, -1),
                digit_emb
            ], dim=-1)
            cadd_opq_subvectors = self.cadd_opq_subvec_mlp(subvec_input)
            if self.cadd_hint_injection == 'soft_quantized_sid':
                soft_code_probs = self._compute_semantic_code_probs(cadd_opq_subvectors=cadd_opq_subvectors)
                cadd_hint = self._soft_code_probs_to_sid_embedding(soft_code_probs)
            else:
                cadd_hint = torch.stack([
                    self.cadd_opq_subvec_to_hidden[d](cadd_opq_subvectors[:, d, :])
                    for d in range(n_digit)
                ], dim=1)
        elif self.cadd_aux_target in self.cadd_preopq_aux_targets:
            sem_input = torch.cat([partial_summary, encoder_summary], dim=-1)
            cadd_sem_prior = self.cadd_sem_prior_mlp(sem_input)
            sem_hidden = self.cadd_sem_to_hidden(cadd_sem_prior)
            hint_input = torch.cat([
                partial_summary.unsqueeze(1).expand(-1, n_digit, -1),
                encoder_summary.unsqueeze(1).expand(-1, n_digit, -1),
                sem_hidden.unsqueeze(1).expand(-1, n_digit, -1),
                digit_emb
            ], dim=-1)
            cadd_hint = self.cadd_hint_from_sem_mlp(hint_input)
        else:
            hint_input = torch.cat([
                partial_summary.unsqueeze(1).expand(-1, n_digit, -1),
                encoder_summary.unsqueeze(1).expand(-1, n_digit, -1),
                digit_emb
            ], dim=-1)
            cadd_hint = self.cadd_hint_mlp(hint_input)

        return self.cadd_hint_dropout(cadd_hint), cadd_sem_prior, cadd_opq_subvectors

    def _soft_code_probs_to_sid_embedding(self, code_probs: torch.Tensor) -> torch.Tensor:
        if code_probs is None:
            raise ValueError("soft SID embedding hint requires code probabilities")
        if code_probs.dim() != 3:
            raise ValueError(f"code_probs must have shape [B, n_digit, codebook_size], got {tuple(code_probs.shape)}")
        B, D, K = code_probs.shape
        if D != self.n_digit or K != self.codebook_size:
            raise ValueError(
                f"code_probs shape {tuple(code_probs.shape)} is incompatible with "
                f"n_digit={self.n_digit}, codebook_size={self.codebook_size}"
            )

        sid_embs = []
        for d in range(self.n_digit):
            start = self.tokenizer.sid_offset + d * self.codebook_size
            end = start + self.codebook_size
            E_sub = self.embedding.weight[start:end].to(dtype=code_probs.dtype)
            sid_embs.append(torch.matmul(code_probs[:, d, :], E_sub))
        return torch.stack(sid_embs, dim=1)

    def _build_decoder_embeddings(self, decoder_input_ids: torch.Tensor, mask_positions: torch.Tensor, encoder_hidden: torch.Tensor):
        token_emb = self._codebook_ids_to_token_emb(decoder_input_ids)

        B, n_digit, _ = token_emb.shape
        digit_ids = torch.arange(n_digit, device=decoder_input_ids.device)
        mask_emb = self.mask_emb_table(digit_ids).unsqueeze(0).expand(B, -1, -1)

        cadd_hint = None
        cadd_sem_prior = None
        cadd_opq_subvectors = None
        if self.cadd_enabled or self.drift_moe_enabled:
            cadd_hint, cadd_sem_prior, cadd_opq_subvectors = self._compute_cadd_hint(
                token_emb,
                mask_positions,
                encoder_hidden,
            )
            if self.cadd_enabled:
                mask_emb = mask_emb + self.cadd_hint_scale * cadd_hint

        decoder_emb = torch.where(mask_positions.bool().unsqueeze(-1), mask_emb, token_emb)
        return decoder_emb, cadd_hint, cadd_sem_prior, cadd_opq_subvectors

    def _compute_semantic_code_probs(self, cadd_sem_prior: torch.Tensor = None, cadd_opq_subvectors: torch.Tensor = None):
        if self.cadd_code_centroids is None:
            return None
        if cadd_opq_subvectors is not None:
            centroids = self.cadd_code_centroids.to(
                device=cadd_opq_subvectors.device,
                dtype=cadd_opq_subvectors.dtype,
            )
            if cadd_opq_subvectors.size(-1) != centroids.size(-1):
                raise ValueError(
                    f"OPQ subvector dim {cadd_opq_subvectors.size(-1)} != centroid dim {centroids.size(-1)}"
                )
            prior_norm = (cadd_opq_subvectors * cadd_opq_subvectors).sum(dim=-1, keepdim=True)
            centroid_norm = (centroids * centroids).sum(dim=-1).unsqueeze(0)
            dot = torch.einsum('bdc,dkc->bdk', cadd_opq_subvectors, centroids)
            dist2 = (prior_norm + centroid_norm - 2.0 * dot).clamp_min(0.0)
            logits = -dist2 / max(float(self.cadd_code_prob_temperature), 1e-6)
            return F.softmax(logits, dim=-1)
        if cadd_sem_prior is None:
            return None
        centroids = self.cadd_code_centroids.to(
            device=cadd_sem_prior.device,
            dtype=cadd_sem_prior.dtype,
        )
        if cadd_sem_prior.size(-1) != centroids.size(-1):
            raise ValueError(
                f"Semantic prior dim {cadd_sem_prior.size(-1)} != centroid dim {centroids.size(-1)}"
            )
        prior_norm = (cadd_sem_prior * cadd_sem_prior).sum(dim=-1).view(-1, 1, 1)
        centroid_norm = (centroids * centroids).sum(dim=-1).unsqueeze(0)
        dot = torch.einsum('bc,dkc->bdk', cadd_sem_prior, centroids)
        dist2 = (prior_norm + centroid_norm - 2.0 * dot).clamp_min(0.0)
        logits = -dist2 / max(float(self.cadd_code_prob_temperature), 1e-6)
        return F.softmax(logits, dim=-1)

    def _compute_hint_sid_code_probs(
        self,
        decoder_input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        cadd_hint: torch.Tensor,
        cadd_sem_prior: torch.Tensor = None,
        cadd_opq_subvectors: torch.Tensor = None,
    ):
        """Return per-digit SID-code distributions from known digits plus CADD hint."""
        if cadd_hint is None:
            return None

        B, n_digit, _ = cadd_hint.shape
        mask_bool = mask_positions.bool()
        temperature = max(float(self.drift_moe_hint_code_temperature), 1e-6)
        semantic_code_probs = self._compute_semantic_code_probs(
            cadd_sem_prior=cadd_sem_prior,
            cadd_opq_subvectors=cadd_opq_subvectors,
        )
        code_probs = []

        for d in range(n_digit):
            if semantic_code_probs is not None:
                hint_probs = semantic_code_probs[:, d, :]
            else:
                start = self.tokenizer.sid_offset + d * self.codebook_size
                end = start + self.codebook_size
                E_sub = self.embedding.weight[start:end]
                hint_logits = torch.matmul(cadd_hint[:, d, :], E_sub.t()) / temperature
                hint_probs = F.softmax(hint_logits, dim=-1)

            known_ids = decoder_input_ids[:, d].clamp(0, self.codebook_size - 1)
            known_probs = F.one_hot(known_ids, num_classes=self.codebook_size).to(dtype=hint_probs.dtype)
            use_hint = mask_bool[:, d].unsqueeze(-1)
            code_probs.append(torch.where(use_hint, hint_probs, known_probs))

        return code_probs

    def _compute_exact_code_repeat_prob(
        self,
        code_prob: torch.Tensor,
        hist_codes: torch.Tensor,
    ) -> torch.Tensor:
        """Probability mass assigned to exact SID codes appearing in history."""
        B, S = hist_codes.shape
        return code_prob.gather(1, hist_codes.reshape(B, S)).view(B, S)

    def _compute_topk_novel_code_prob(
        self,
        code_prob: torch.Tensor,
        hist_codes: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Probability mass of hint top-k SID codes that never appear in the digit-wise history."""
        k = min(int(self.drift_moe_novel_topk), code_prob.size(1))
        topk_prob, topk_codes = torch.topk(code_prob, k=k, dim=1)
        in_history = (topk_codes.unsqueeze(-1) == hist_codes.unsqueeze(1)) & valid.unsqueeze(1)
        is_novel = ~in_history.any(dim=-1)
        return (topk_prob * is_novel.to(dtype=topk_prob.dtype)).sum(dim=1)

    def _compute_hard_drift_scores(
        self,
        history_sid: torch.Tensor,
        history_mask: torch.Tensor,
        decoder_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute recency-weighted exact-repeat drift against the true target SID."""
        B, S, D = history_sid.shape
        device = history_sid.device
        valid = history_mask.bool().unsqueeze(-1) & (history_sid >= 0)
        matches = (history_sid == decoder_labels.unsqueeze(1)) & valid

        positions = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        lengths = history_mask.float().sum(dim=1).clamp_min(1.0)
        last_pos = lengths.unsqueeze(1) - 1.0
        recency = torch.pow(self.drift_moe_recency_gamma, (last_pos - positions).clamp_min(0.0))
        recency = recency * history_mask.float()
        denom = recency.sum(dim=1, keepdim=True).clamp_min(1e-12)
        sim_d = (matches.float() * recency.unsqueeze(-1)).sum(dim=1) / denom

        digit_weights = self._parse_drift_digit_weights(D, device).to(dtype=sim_d.dtype)
        sid_sim = (sim_d * digit_weights.unsqueeze(0)).sum(dim=1).clamp(0.0, 1.0)
        return (1.0 - sid_sim).clamp(0.0, 1.0)

    def _drift_scalar_to_bucket_labels(self, drift: torch.Tensor) -> torch.Tensor:
        if self.drift_moe_bucket_strategy in ("train_quantile_full", "quantile_full"):
            edges = self.drift_moe_bucket_edges.to(device=drift.device, dtype=drift.dtype)
            bucket = torch.zeros_like(drift, dtype=torch.long)
            bucket = bucket + (drift >= edges[0]).long()
            bucket = bucket + (drift >= edges[1]).long()
            bucket = bucket.clamp(max=2)
            bucket[drift >= 1.0 - self.drift_moe_full_drift_eps] = 3
            return bucket.clamp_(0, self.drift_moe_n_experts - 1)

        bucket = torch.floor(drift.clamp(0.0, 1.0 - 1e-12) * self.drift_moe_n_experts).long()
        bucket[drift >= 1.0 - 1e-12] = self.drift_moe_n_experts - 1
        return bucket.clamp_(0, self.drift_moe_n_experts - 1)

    def _drift_scalar_to_bucket_probs(self, drift: torch.Tensor) -> torch.Tensor:
        centers = self.drift_moe_bucket_centers.to(device=drift.device, dtype=drift.dtype)
        logits = -((drift.unsqueeze(-1) - centers.unsqueeze(0)) ** 2) / self.drift_moe_bucket_temperature
        return F.softmax(logits, dim=-1)

    def fit_drift_buckets_from_dataset(self, train_dataset):
        """Fit drift buckets on train data: D<1 is split into 3 quantile buckets, D=1 is bucket 4."""
        if not self.drift_moe_enabled:
            return None
        if self.drift_moe_bucket_strategy not in ("train_quantile_full", "quantile_full"):
            return None
        if train_dataset is None or not hasattr(train_dataset, "__len__") or len(train_dataset) == 0:
            return None

        batch_size = int(self.drift_moe_bucket_fit_batch_size)
        drift_chunks = []
        for start in range(0, len(train_dataset), batch_size):
            stop = min(start + batch_size, len(train_dataset))
            batch = train_dataset[start:stop]
            if not all(k in batch for k in ("history_sid", "history_mask", "decoder_labels")):
                return None
            history_sid = torch.as_tensor(batch["history_sid"], dtype=torch.long)
            history_mask = torch.as_tensor(batch["history_mask"], dtype=torch.bool)
            decoder_labels = torch.as_tensor(batch["decoder_labels"], dtype=torch.long)
            drift = self._compute_hard_drift_scores(history_sid, history_mask, decoder_labels)
            drift_chunks.append(drift.cpu())

        if not drift_chunks:
            return None

        drifts = torch.cat(drift_chunks).float().clamp(0.0, 1.0)
        full_mask = drifts >= 1.0 - self.drift_moe_full_drift_eps
        partial = drifts[~full_mask]

        if partial.numel() >= 3:
            q = torch.quantile(partial, torch.tensor([1.0 / 3.0, 2.0 / 3.0]))
            q1, q2 = q[0].item(), q[1].item()
        elif partial.numel() > 0:
            q1 = float(partial.min().item())
            q2 = float(partial.max().item())
        else:
            q1, q2 = 1.0 / 3.0, 2.0 / 3.0

        q1 = float(max(0.0, min(q1, 1.0 - self.drift_moe_full_drift_eps)))
        q2 = float(max(q1, min(q2, 1.0 - self.drift_moe_full_drift_eps)))
        edges = torch.tensor([q1, q2, 1.0], dtype=self.drift_moe_bucket_edges.dtype)
        self.drift_moe_bucket_edges.copy_(edges.to(self.drift_moe_bucket_edges.device))

        labels = self._drift_scalar_to_bucket_labels(drifts)
        centers = []
        defaults = torch.tensor(
            [q1 / 2.0, (q1 + q2) / 2.0, (q2 + 1.0) / 2.0, 1.0],
            dtype=torch.float32,
        )
        for idx in range(self.drift_moe_n_experts):
            values = drifts[labels.cpu() == idx]
            centers.append(float(values.mean().item()) if values.numel() > 0 else float(defaults[idx].item()))
        centers[-1] = 1.0 if full_mask.any() else centers[-1]
        centers = torch.tensor(centers, dtype=self.drift_moe_bucket_centers.dtype).clamp(0.0, 1.0)
        self.drift_moe_bucket_centers.copy_(centers.to(self.drift_moe_bucket_centers.device))
        self.drift_moe_buckets_fitted = True

        counts = torch.bincount(labels.cpu(), minlength=self.drift_moe_n_experts)
        return {
            "num_samples": int(drifts.numel()),
            "num_full_drift": int(full_mask.sum().item()),
            "edges": [float(x) for x in self.drift_moe_bucket_edges.cpu().tolist()],
            "centers": [float(x) for x in self.drift_moe_bucket_centers.cpu().tolist()],
            "counts": [int(x) for x in counts.tolist()],
        }

    def _select_history_for_drift(self, batch: dict, decoder_batch_size: int, device: torch.device):
        history_sid = batch.get('history_sid')
        history_mask = batch.get('history_mask')

        if history_sid is None:
            history_sid = getattr(self, '_generation_history_sid', None)
            history_mask = getattr(self, '_generation_history_mask', None)
        if history_sid is None:
            return None, None

        history_sid = history_sid.to(device)
        if history_mask is None:
            history_mask = (history_sid >= 0).any(dim=-1)
        else:
            history_mask = history_mask.to(device)

        if history_sid.size(0) == decoder_batch_size:
            return history_sid, history_mask
        if decoder_batch_size % history_sid.size(0) != 0:
            raise ValueError(
                f"Cannot align history batch {history_sid.size(0)} to decoder batch {decoder_batch_size}"
            )

        repeat = decoder_batch_size // history_sid.size(0)
        history_sid = history_sid.unsqueeze(1).repeat(1, repeat, 1, 1).view(
            decoder_batch_size, history_sid.size(1), history_sid.size(2)
        )
        history_mask = history_mask.unsqueeze(1).repeat(1, repeat, 1).view(
            decoder_batch_size, history_mask.size(1)
        )
        return history_sid, history_mask

    def _compute_hint_drift_probs(
        self,
        history_sid: torch.Tensor,
        history_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        mask_positions: torch.Tensor,
        cadd_hint: torch.Tensor,
        cadd_sem_prior: torch.Tensor = None,
        cadd_opq_subvectors: torch.Tensor = None,
    ):
        if not self.drift_moe_enabled or cadd_hint is None or history_sid is None:
            return None, None

        code_probs = self._compute_hint_sid_code_probs(
            decoder_input_ids,
            mask_positions,
            cadd_hint,
            cadd_sem_prior=cadd_sem_prior,
            cadd_opq_subvectors=cadd_opq_subvectors,
        )
        B, S, D = history_sid.shape
        device = history_sid.device
        valid_seq = history_mask.bool()

        positions = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)
        lengths = valid_seq.float().sum(dim=1).clamp_min(1.0)
        last_pos = lengths.unsqueeze(1) - 1.0
        recency = torch.pow(self.drift_moe_recency_gamma, (last_pos - positions).clamp_min(0.0))
        recency = recency * valid_seq.float()

        drift_by_digit = []
        for d in range(D):
            hist_codes = history_sid[:, :, d]
            valid = valid_seq & (hist_codes >= 0)
            hist_codes = hist_codes.clamp(0, self.codebook_size - 1)
            prob_at_history = self._compute_exact_code_repeat_prob(code_probs[d], hist_codes)
            weight = recency * valid.float()
            denom = weight.sum(dim=1).clamp_min(1e-12)
            repeat_evidence = (prob_at_history * weight).sum(dim=1) / denom
            repeat_evidence = torch.where(valid.any(dim=1), repeat_evidence, torch.zeros_like(repeat_evidence))

            novel_evidence = self._compute_topk_novel_code_prob(
                code_prob=code_probs[d],
                hist_codes=hist_codes,
                valid=valid,
            )
            evidence = torch.stack([repeat_evidence, novel_evidence], dim=-1).clamp_min(1e-12)
            evidence_logits = torch.log(evidence) / self.drift_moe_evidence_temperature
            drift_by_digit.append(F.softmax(evidence_logits, dim=-1)[:, 1].clamp(0.0, 1.0))

        drift_by_digit = torch.stack(drift_by_digit, dim=1)
        digit_weights = self._parse_drift_digit_weights(D, device).to(dtype=drift_by_digit.dtype)
        drift = (drift_by_digit * digit_weights.unsqueeze(0)).sum(dim=1).clamp(0.0, 1.0)
        drift_probs = self._drift_scalar_to_bucket_probs(drift)
        return drift_probs, drift

    def _apply_drift_moe(
        self,
        decoder_hidden: torch.Tensor,
        drift_probs: torch.Tensor,
    ):
        if not self.drift_moe_enabled or drift_probs is None:
            return decoder_hidden, None

        expert_outputs = torch.stack([expert(decoder_hidden) for expert in self.drift_experts], dim=1)
        mixed_expert = (drift_probs.to(dtype=expert_outputs.dtype)[:, :, None, None] * expert_outputs).sum(dim=1)
        decoder_hidden = decoder_hidden + torch.tanh(self.drift_moe_alpha) * mixed_expert
        return decoder_hidden, drift_probs

    def _parse_drift_digit_weights(self, n_digit: int, device: torch.device) -> torch.Tensor:
        raw = self.drift_moe_digit_weights
        if raw is None:
            weights = torch.ones(n_digit, device=device)
        else:
            if isinstance(raw, str):
                raw = [x.strip() for x in raw.split(',') if x.strip()]
            if isinstance(raw, (list, tuple)):
                weights = torch.tensor([float(x) for x in raw], device=device)
            else:
                weights = torch.tensor([float(raw)], device=device)
            if weights.numel() != n_digit:
                raise ValueError(f"drift_moe.digit_weights length must equal n_digit={n_digit}, got {weights.numel()}")
        return weights / weights.sum().clamp_min(1e-12)

    def _compute_drift_bucket_labels(
        self,
        history_sid: torch.Tensor,
        history_mask: torch.Tensor,
        decoder_labels: torch.Tensor,
    ) -> torch.Tensor:
        drift = self._compute_hard_drift_scores(history_sid, history_mask, decoder_labels)
        return self._drift_scalar_to_bucket_labels(drift)

    @property
    def n_parameters(self) -> str:
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return f"{n_params:,}"

    def forward(self, batch: dict, return_loss=True) -> ModelOutput:
        """
        Diffusion训练：处理掩码数据，预测被掩码的位置
        
        Args:
            batch: 包含以下字段的字典：
                - history_sid: 历史SID序列 [B, seq_len, n_digit]
                - decoder_input_ids: decoder输入 [B, n_digit] 
                - decoder_labels: 真实标签 [B, n_digit]
        """
        device = next(self.parameters()).device
        
        # 添加调试信息
        if hasattr(self, '_debug_printed'):
            pass
        else:
            print(f"[DADRec] Using RPG_ED-style encoder: MLP compression + fixed 50-length sequence")
            print(f"[DADRec] vocab_size: {self.vocab_size}, codebook_size: {self.codebook_size}")
            print(f"[DADRec] masking_strategy: {self.masking_strategy}")
            print(
                f"[DADRec] CADD enabled: {self.cadd_enabled}, hint_scale: {self.cadd_hint_scale}, "
                f"aux_loss_weight: {self.cadd_aux_loss_weight}, aux_target: {self.cadd_aux_target}, "
                f"hint_injection: {self.cadd_hint_injection}"
            )
            if self.drift_moe_enabled:
                print(f"[DADRec] Drift MoE enabled: experts={self.drift_moe_n_experts}, bottleneck={self.drift_moe_bottleneck}, drift_loss_weight={self.drift_moe_gate_loss_weight}, bucket_temperature={self.drift_moe_bucket_temperature}")
            if self.masking_strategy == 'random' and self.mask_probs is not None:
                print(f"[DADRec] mask_probs: {self.mask_probs}")
            self._debug_printed = True
        
        # --- Encoder ---
        history_sid = batch['history_sid'].to(device)  # [B, seq_len, n_digit]
        B, seq_len, n_digit = history_sid.shape
        
        # 断言：history_sid 应该是 codebook id (0..K-1) 或 PAD (-1)
        valid_hist = ((history_sid == -1) | ((history_sid >= 0) & (history_sid < self.codebook_size))).all()
        assert bool(valid_hist), \
            f"history_sid 应为 codebook id(0..{self.codebook_size-1}) 或 -1(PAD)，但发现越界值"
        
        # 1. 将history SID转换为token IDs
        history_tokens = torch.zeros(B, seq_len, n_digit, dtype=torch.long, device=device)
        for d in range(n_digit):
            # 处理PAD：-1映射到token_id=0(PAD)，其他codebook_id正常加offset
            codebook_ids = history_sid[:, :, d]
            token_ids = torch.where(
                codebook_ids == -1,  # PAD位置
                torch.zeros_like(codebook_ids),  # 映射到token_id=0(PAD)
                codebook_ids + self.tokenizer.sid_offset + d * self.codebook_size  # 正常加offset
            )
            # 确保token ID在有效范围内
            token_ids = torch.clamp(token_ids, 0, self.vocab_size - 1)
            history_tokens[:, :, d] = token_ids
        
        # 2. 获取token嵌入
        tok_emb = self.embedding(history_tokens)  # [B, seq_len, n_digit, d]
        B, S, _, d = tok_emb.shape
        
        # 3. 重塑并通过MLP压缩：n_digit个SID token → 1个item token
        item_emb = tok_emb.reshape(B, S, self.n_digit * d)  # [B, S, n_digit*d]
        item_emb = self.item_mlp(item_emb)  # [B, S, d]
        
        # 4. 添加位置编码（与RPG_ED一致）
        pos_ids = torch.arange(S, device=item_emb.device)  # (S,)
        pos_emb = self.pos_emb_enc(pos_ids)  # (S, d)
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1)  # (B, S, d)
        
        # 5. 将位置编码加到item_emb上
        encoder_hidden = item_emb + pos_emb  # [B, S, d]
        encoder_hidden = self.drop(encoder_hidden)
        
        # 6. 处理PAD位置的注意力掩码
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, seq_len]
            # 创建注意力掩码：True=有效位置，False=PAD位置
            attention_mask = history_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, -1, seq_len, -1)  # [B, 1, seq_len, seq_len]
        else:
            attention_mask = None
        
        # Pass through encoder blocks
        encoder_hidden = encoder_hidden
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden, attention_mask=attention_mask)
        
        encoder_hidden = self.ln_f(encoder_hidden)  # [B, seq_len*n_digit, emb_dim]
        
        # >>> 新增：将 PAD 位置的 encoder_hidden 清零，避免 cross-attn 看到无效KV <<<
        if 'history_mask' in batch:
            history_mask = batch['history_mask'].to(device)  # [B, S]，True=有效
            encoder_hidden = encoder_hidden * history_mask.unsqueeze(-1).float()
        
        if not return_loss:
            # 推理模式，直接返回encoder输出
            output = ModelOutput()
            output.hidden_states = encoder_hidden
            return output
        
        # --- 多概率掩码扩展 ---
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        decoder_labels = batch['decoder_labels'].to(device)  # [B, n_digit]
        target_sem_emb = batch.get('target_sem_emb')
        if target_sem_emb is not None:
            target_sem_emb = target_sem_emb.to(device).float()
        target_opq_subvectors = batch.get('target_opq_subvectors')
        if target_opq_subvectors is not None:
            target_opq_subvectors = target_opq_subvectors.to(device).float()
        
        # 确保decoder输入在有效范围内
        decoder_input_ids = torch.clamp(decoder_input_ids, 0, self.codebook_size - 1)
        decoder_labels = torch.clamp(decoder_labels, 0, self.codebook_size - 1)

        if 'history_mask' in batch:
            base_history_mask = batch['history_mask'].to(device)
        else:
            base_history_mask = (history_sid >= 0).any(dim=-1)

        drift_bucket_labels = None
        if self.drift_moe_enabled and self.drift_moe_gate_loss_weight > 0.0:
            drift_bucket_labels = self._compute_drift_bucket_labels(history_sid, base_history_mask, decoder_labels)
        
        # ---------- 构造训练视图 ----------
        all_masked_input_ids = []
        all_labels = []
        all_mask_positions = []
        all_encoder_hidden = []
        
        if self.masking_strategy == 'sequential':
            # 连贯多视图：支持多路径并行
            for p in range(self.sequential_paths):  # 先生成多条路径
                # ① 本条路径各个样本的随机顺序
                orders = torch.argsort(torch.rand(B, self.n_digit, device=device), dim=1)

                # ② step-0: 全 MASK
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                inp0 = decoder_input_ids.new_zeros(B, self.n_digit)        # 全 0 → MASK
                all_masked_input_ids.append(inp0)
                all_labels.append(decoder_labels)
                all_mask_positions.append(full_mask.float())
                all_encoder_hidden.append(encoder_hidden)

                # ③ step-1 … step-(seq_steps-1) ：按随机顺序逐步揭开
                for reveal in range(1, self.seq_steps):        # 1 .. seq_steps-1
                    mask_pos = torch.ones_like(full_mask)      # 先全部 MASK

                    # orders[:, :reveal] 形状 (B, reveal)
                    reveal_idx = orders[:, :reveal]            # 每条样本本次需要揭开的列
                    mask_pos.scatter_(1, reveal_idx, 0)        # 置 0 表示「不掩码」

                    inp = decoder_input_ids.clone()
                    inp[mask_pos] = 0                          # 掩码位写 0

                    all_masked_input_ids.append(inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(mask_pos.float())
                    all_encoder_hidden.append(encoder_hidden)
        elif self.masking_strategy == 'guided':
            B = decoder_labels.size(0)
            device = decoder_labels.device

            def score_with_mask(cur_mask: torch.Tensor):
                # cur_mask: [B, n_digit], True=被掩盖（需要预测）
                cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                cur_inp[~cur_mask] = decoder_labels[~cur_mask]  # 未掩盖的位置放"真标签"

                _was_training = self.training
                self.eval()
                with torch.no_grad():
                    if B == 1:  # 只在单样本时打印，避免多worker刷屏
                        print(f"[GUIDED] scoring: self.training={self.training}")  # 这里应为 False
                    logits = self.forward_decoder_only(
                        {
                            'decoder_input_ids': cur_inp,
                            'encoder_hidden': encoder_hidden,
                            'mask_positions': cur_mask.float(),
                            'history_sid': history_sid,
                            'history_mask': base_history_mask,
                        },
                        return_loss=False, digit=None, use_cache=False
                    ).logits  # [B, n_digit, K]
                if _was_training:
                    self.train()

                # 计算置信度（与推理一致）
                probs = F.softmax(logits, dim=-1)
                if self.guided_conf_metric == 'entropy':
                    ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
                    conf = -ent
                else:  # 'msp'
                    conf = probs.max(dim=-1).values  # ★ 这里用 max(...).values

                return conf  # [B, n_digit]

            refresh = str(self.config.get('guided_refresh_each_step', False)).lower() in ('1','true','yes','y')
            all_masked_input_ids, all_labels, all_mask_positions, all_encoder_hidden = [], [], [], []

            if not refresh:
                # ------- 一次性排序，不刷新 -------
                full_mask = torch.ones(B, self.n_digit, dtype=torch.bool, device=device)
                conf = score_with_mask(full_mask)  # 用全掩盖打分得到 rank
                if self.guided_select == 'most':
                    order = torch.argsort(conf, 1, True)
                else:
                    order = torch.argsort(conf, 1, False)

                for t in range(1, self.guided_steps + 1):
                    cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                    cols = order[:, :t]
                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
            else:
                # ------- 每步刷新 -------
                cur_mask = torch.zeros(B, self.n_digit, dtype=torch.bool, device=device)
                for t in range(1, self.guided_steps + 1):
                    conf = score_with_mask(cur_mask)  # 本步置信度

                    # 已经掩盖过的列不再选择
                    if self.guided_select == 'most':
                        conf = conf.masked_fill(cur_mask, -1e9)
                        cols = torch.argmax(conf, dim=1, keepdim=True)  # 每个样本挑 1 列
                    else:
                        conf = conf.masked_fill(cur_mask,  1e9)
                        cols = torch.argmin(conf, dim=1, keepdim=True)

                    cur_mask.scatter_(1, cols, True)

                    cur_inp = decoder_input_ids.new_zeros(B, self.n_digit)
                    cur_inp[~cur_mask] = decoder_labels[~cur_mask]

                    all_masked_input_ids.append(cur_inp)
                    all_labels.append(decoder_labels)
                    all_mask_positions.append(cur_mask.float())
                    all_encoder_hidden.append(encoder_hidden)
        
        else:
            # 旧的随机掩码分支（保持原逻辑）
            # LLaDA风格：若启用mask_prob_random，则每个batch独立采样一次掩码率
            batch_mask_prob = None
            if getattr(self, 'mask_prob_random', False):
                low = float(self.config.get('mask_prob_random_min', 0.0))
                high = float(self.config.get('mask_prob_random_max', 1.0))
                # 使用torch采样以便与全局随机种子一致
                batch_mask_prob = float(torch.empty(1).uniform_(low, high).item())
            for view_idx, mask_prob in enumerate(self.mask_probs):
                if batch_mask_prob is not None:
                    mask_prob = batch_mask_prob
                # 为当前掩码概率生成掩码
                mask_positions = torch.rand(B, self.n_digit, device=device) < mask_prob  # [B, n_digit]
                
                # 确保每个样本至少有一个位置被掩码
                no_mask_samples = ~mask_positions.any(dim=1)  # [B]
                if no_mask_samples.any():
                    # 对于没有掩码的样本，强制掩码第一个位置
                    mask_positions[no_mask_samples, 0] = True
                
                # 应用掩码：被掩码的位置设为0
                masked_input_ids = decoder_input_ids.clone()  # [B, n_digit]
                masked_input_ids[mask_positions] = 0
                
                # 存储当前视图的数据
                all_masked_input_ids.append(masked_input_ids)
                all_labels.append(decoder_labels)  # 标签保持不变
                all_mask_positions.append(mask_positions.float())
                all_encoder_hidden.append(encoder_hidden)  # 每个视图使用相同的encoder输出
        
        # 合并所有视图：[B*n_views, ...]
        n_views = len(all_masked_input_ids)
        decoder_input_ids = torch.cat(all_masked_input_ids, dim=0)  # [B*n_views, n_digit]
        decoder_labels = torch.cat(all_labels, dim=0)  # [B*n_views, n_digit]
        mask_positions = torch.cat(all_mask_positions, dim=0)  # [B*n_views, n_digit]
        encoder_hidden = torch.cat(all_encoder_hidden, dim=0)  # [B*n_views, seq_len*n_digit, emb_dim]
        if self.drift_moe_enabled:
            expanded_history_sid = history_sid.repeat(n_views, 1, 1)
            expanded_history_mask = base_history_mask.repeat(n_views, 1)
        else:
            expanded_history_sid = None
            expanded_history_mask = None
        if drift_bucket_labels is not None:
            drift_bucket_labels = drift_bucket_labels.repeat(n_views)
        if target_sem_emb is not None:
            target_sem_emb = target_sem_emb.repeat(n_views, 1)
        if target_opq_subvectors is not None:
            target_opq_subvectors = target_opq_subvectors.repeat(n_views, 1, 1)
        
        # 更新batch大小并验证形状
        B_expanded = B * self.augment_factor
        
        # 形状验证
        assert decoder_input_ids.shape[0] == B_expanded, f"decoder_input_ids shape mismatch: {decoder_input_ids.shape[0]} vs {B_expanded}"
        assert decoder_labels.shape[0] == B_expanded, f"decoder_labels shape mismatch: {decoder_labels.shape[0]} vs {B_expanded}"
        assert mask_positions.shape[0] == B_expanded, f"mask_positions shape mismatch: {mask_positions.shape[0]} vs {B_expanded}"
        assert encoder_hidden.shape[0] == B_expanded, f"encoder_hidden shape mismatch: {encoder_hidden.shape[0]} vs {B_expanded}"
        
        # 一致性检查：guided策略应该逐步增加掩码数
        if self.masking_strategy == 'guided':
            m = mask_positions.view(B, self.augment_factor, self.n_digit).sum(-1)  # [B, 4]
            assert torch.all(m[:, 1:] >= m[:, :-1]), "guided views should increase masked count monotonically"
        
        # --- Decoder (训练模式) ---
        # 🚀 训练阶段也使用与推理一致的cross-attention投影
        encoder_kv_list = []
        for blk in self.decoder_blocks:
            # 执行 W_k/W_v 投影，与推理保持完全一致
            kv_proj = blk.cross_attn.qkv(encoder_hidden)  # [B_expanded, seq_len, 3*emb_dim]
            # 提取K和V部分（跳过Q部分）
            k = kv_proj[..., self.n_embd:2*self.n_embd]  # [B_expanded, seq_len, emb_dim]
            v = kv_proj[..., 2*self.n_embd:]              # [B_expanded, seq_len, emb_dim]
            # 拼接K和V
            layer_kv = torch.cat([k, v], dim=-1)  # [B_expanded, seq_len, 2*emb_dim]
            encoder_kv_list.append(layer_kv)
        
        decoder_emb, cadd_hint, cadd_sem_prior, cadd_opq_subvectors = self._build_decoder_embeddings(
            decoder_input_ids=decoder_input_ids,
            mask_positions=mask_positions,
            encoder_hidden=encoder_hidden
        )

        # 移除位置编码：decoder只使用掩码，不需要位置编码
        decoder_emb = self.drop(decoder_emb)
        
        # Pass through decoder blocks with consistent cross-attention
        decoder_hidden = decoder_emb
        for i, block in enumerate(self.decoder_blocks):
            block_output = block(
                decoder_hidden, 
                encoder_hidden=encoder_hidden,     # 仍传递H，方便fallback
                past_key_value=None,               # 训练时不使用KV cache
                use_cache=False,                   # 训练时不使用KV cache
                cross_key_value=encoder_kv_list[i] # 🚀 使用预计算的KV，与推理一致
            )
            decoder_hidden = block_output['hidden_states']
        
        decoder_hidden = self.ln_f(decoder_hidden)  # [B_expanded, n_digit, emb_dim]
        drift_probs = None
        drift_scores = None
        if self.drift_moe_enabled:
            drift_probs, drift_scores = self._compute_hint_drift_probs(
                history_sid=expanded_history_sid,
                history_mask=expanded_history_mask,
                decoder_input_ids=decoder_input_ids,
                mask_positions=mask_positions,
                cadd_hint=cadd_hint,
                cadd_sem_prior=cadd_sem_prior,
                cadd_opq_subvectors=cadd_opq_subvectors,
            )
        decoder_hidden, drift_probs = self._apply_drift_moe(
            decoder_hidden=decoder_hidden,
            drift_probs=drift_probs,
        )
        
        # 计算损失
        if self.masking_strategy == 'random' and getattr(self, 'mask_prob_random', False):
            # LLaDA 风格：对每个样本先按掩码位汇总，再乘以 1/t，有效抑制不同掩码率带来的尺度差异
            # 这里的 t 使用“实际掩码率”而非采样参数，避免极小 t 被强制掩一个位时产生过大权重
            per_sample_loss = torch.zeros(B_expanded, device=device)
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                per_sample_loss += loss_d * mask_d  # 只计掩码位
            # 实际掩码率 t_i：每个样本被掩的比例
            t_actual = mask_positions.float().mean(dim=1)  # [B_expanded]
            t_actual = torch.clamp(t_actual, min=1e-6)
            total_loss = (per_sample_loss / t_actual).mean()  # 按batch求平均
        else:
            # 原逻辑：只在掩码位计算损失，并按被掩码token数做平均
            total_loss = 0.0
            total_weight = 0.0
            for d in range(self.n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                labels_d = decoder_labels[:, d]
                mask_d = mask_positions[:, d].float()
                loss_d = F.cross_entropy(
                    logits_d, labels_d, reduction='none',
                    label_smoothing=self.config.get('label_smoothing', 0.1)
                )
                total_loss += (loss_d * mask_d).sum()
                total_weight += mask_d.sum()
            if total_weight > 0:
                total_loss = total_loss / total_weight
            else:
                total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        aux_loss = None
        if self.cadd_enabled and cadd_hint is not None and self.cadd_aux_loss_weight > 0.0:
            if self.cadd_aux_target in self.cadd_opq_aux_targets:
                if target_opq_subvectors is None:
                    raise ValueError("cadd.aux_target=opq_subvector requires batch['target_opq_subvectors']")
                if cadd_opq_subvectors is None:
                    raise ValueError("cadd.aux_target=opq_subvector requires predicted OPQ subvectors")
                if cadd_opq_subvectors.shape != target_opq_subvectors.shape:
                    raise ValueError(
                        f"Predicted OPQ subvector shape {tuple(cadd_opq_subvectors.shape)} "
                        f"!= target shape {tuple(target_opq_subvectors.shape)}"
                    )
                subvec_loss = F.mse_loss(cadd_opq_subvectors, target_opq_subvectors, reduction='none').mean(dim=-1)
                mask_weight = mask_positions.float()
                aux_weight = mask_weight.sum().clamp_min(1.0)
                aux_loss = (subvec_loss * mask_weight).sum() / aux_weight
            elif self.cadd_aux_target in self.cadd_preopq_aux_targets:
                if target_sem_emb is None:
                    raise ValueError("cadd.aux_target=pre_opq requires batch['target_sem_emb']")
                if cadd_sem_prior is None:
                    raise ValueError("cadd.aux_target=pre_opq requires a semantic prior")
                if cadd_sem_prior.size(-1) != target_sem_emb.size(-1):
                    raise ValueError(
                        f"Predicted semantic dim {cadd_sem_prior.size(-1)} != target dim {target_sem_emb.size(-1)}"
                    )
                if self.cadd_semantic_loss == 'cosine':
                    pred_sem = F.normalize(cadd_sem_prior, dim=-1)
                    target_sem = F.normalize(target_sem_emb, dim=-1)
                    aux_loss = 1.0 - (pred_sem * target_sem).sum(dim=-1).mean()
                else:
                    aux_loss = F.mse_loss(cadd_sem_prior, target_sem_emb)
            else:
                target_emb = self._codebook_ids_to_token_emb(decoder_labels)
                hint_loss = F.mse_loss(cadd_hint, target_emb, reduction='none').mean(dim=-1)
                mask_weight = mask_positions.float()
                aux_weight = mask_weight.sum().clamp_min(1.0)
                aux_loss = (hint_loss * mask_weight).sum() / aux_weight
            total_loss = total_loss + self.cadd_aux_loss_weight * aux_loss

        drift_gate_loss = None
        if (
            self.drift_moe_enabled
            and drift_probs is not None
            and drift_bucket_labels is not None
            and self.drift_moe_gate_loss_weight > 0.0
        ):
            drift_gate_loss = F.nll_loss(torch.log(drift_probs.clamp_min(1e-12)), drift_bucket_labels)
            total_loss = total_loss + self.drift_moe_gate_loss_weight * drift_gate_loss

        output = ModelOutput()
        output.loss = total_loss
        output.hidden_states = decoder_hidden
        output.logits = None  # 不返回所有logits，节省内存
        output.drift_gate_probs = drift_probs
        output.drift_probs = drift_probs
        output.drift_scores = drift_scores
        output.drift_gate_loss = drift_gate_loss
        output.cadd_aux_loss = aux_loss
        output.cadd_semantic_prior = cadd_sem_prior
        output.cadd_opq_subvectors = cadd_opq_subvectors
        
        return output

    def forward_decoder_only(self, batch: dict, return_loss=False, digit=None, 
                            past_key_values=None, use_cache=False) -> ModelOutput:
        """
        仅运行decoder部分，用于推理时的迭代预测
        
        Args:
            batch: 包含以下字段的字典：
                - decoder_input_ids: decoder输入 [B, n_digit]
                - encoder_hidden: encoder输出 [B, seq_len, emb_dim]
                - mask_positions: 掩码位置 [B, n_digit] (可选)
            digit: 要预测的digit位置
            past_key_values: 缓存的key-value对，用于加速推理
            use_cache: 是否使用KV缓存
        """
        device = next(self.parameters()).device
        
        decoder_input_ids = batch['decoder_input_ids'].to(device)  # [B, n_digit]
        encoder_hidden = batch['encoder_hidden'].to(device)  # [B, seq_len, emb_dim]
        B, n_digit = decoder_input_ids.shape
        
        # 获取掩码位置，如果没有提供则假设所有位置都不被掩码
        if 'mask_positions' in batch:
            mask_positions = batch['mask_positions'].to(device)  # [B, n_digit]
        else:
            mask_positions = torch.zeros(B, n_digit, device=device)
        
        # 🚀 Cross-KV Cache优化：第一步计算，后续步从past_key_values复用
        encoder_kv_list = None
        
        if past_key_values is None and use_cache:
            # 第一步：为每层预计算cross-attention的KV
            encoder_kv_list = []
            for blk in self.decoder_blocks:
                with torch.no_grad():
                    kv_proj = blk.cross_attn.qkv(encoder_hidden)  # [B, seq_len, 3*emb_dim]
                    k = kv_proj[..., self.n_embd:2*self.n_embd]  # [B, seq_len, emb_dim]
                    v = kv_proj[..., 2*self.n_embd:]              # [B, seq_len, emb_dim]
                    layer_kv = torch.cat([k, v], dim=-1)  # [B, seq_len, 2*emb_dim]
                encoder_kv_list.append(layer_kv)
        elif past_key_values is not None:
            # 后续步：从past_key_values中提取cross-KV，实现真正的cache复用
            encoder_kv_list = []
            for layer_cache in past_key_values:
                if layer_cache is not None and len(layer_cache) >= 2:
                    _, cross_kv = layer_cache
                    if cross_kv is not None:
                        cross_key, cross_value = cross_kv
                        layer_kv = torch.cat([cross_key, cross_value], dim=-1)
                        encoder_kv_list.append(layer_kv)
                    else:
                        encoder_kv_list.append(None)
                else:
                    encoder_kv_list.append(None)
        
        decoder_emb, cadd_hint, cadd_sem_prior, cadd_opq_subvectors = self._build_decoder_embeddings(
            decoder_input_ids=decoder_input_ids,
            mask_positions=mask_positions,
            encoder_hidden=encoder_hidden
        )

        # 移除位置编码：decoder只使用掩码，不需要位置编码
        decoder_emb = self.drop(decoder_emb)
        
        # Pass through decoder blocks with KV cache support
        decoder_hidden = decoder_emb
        present_key_values = []
        
        for i, block in enumerate(self.decoder_blocks):
            # 获取当前层的past_key_value
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # 🚀 传入预计算的cross-KV，实现cache复用
            current_cross_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
            
            block_output = block(
                decoder_hidden, 
                encoder_hidden=encoder_hidden,     # 仍传递H，方便fallback
                past_key_value=layer_past,
                use_cache=use_cache,
                cross_key_value=current_cross_kv   # 只在第一次调用时传入
            )
            decoder_hidden = block_output['hidden_states']
            
            # 收集新的key-value cache
            if use_cache:
                layer_present = block_output.get('present_key_value')
                if layer_present is not None and len(layer_present) >= 2:
                    self_present, cross_present = layer_present
                    # 确保cross_present保存分离的K和V用于下次缓存
                    if cross_present is not None:
                        # cross_present应该是(K, V)格式
                        layer_kv = encoder_kv_list[i] if encoder_kv_list is not None else None
                        if layer_kv is not None:
                            k, v = layer_kv.chunk(2, dim=-1)  # 分离K和V
                            cross_present = (k, v)  # 保存分离格式
                    present_key_values.append((self_present, cross_present))
                else:
                    present_key_values.append(layer_present)
        
        # 如果不使用cache，设为None
        if not use_cache:
            present_key_values = None
        
        decoder_hidden = self.ln_f(decoder_hidden)  # [B, n_digit, emb_dim]
        drift_probs = None
        drift_scores = None
        if self.drift_moe_enabled:
            history_sid, history_mask = self._select_history_for_drift(batch, B, device)
            drift_probs, drift_scores = self._compute_hint_drift_probs(
                history_sid=history_sid,
                history_mask=history_mask,
                decoder_input_ids=decoder_input_ids,
                mask_positions=mask_positions,
                cadd_hint=cadd_hint,
                cadd_sem_prior=cadd_sem_prior,
                cadd_opq_subvectors=cadd_opq_subvectors,
            )
        decoder_hidden, drift_probs = self._apply_drift_moe(
            decoder_hidden=decoder_hidden,
            drift_probs=drift_probs,
        )
        
        # 计算指定digit的logits
        if digit is not None:
            logits = self._compute_digit_logits(decoder_hidden[:, digit, :], digit=digit)
        else:
            # 如果没有指定digit，计算所有位置的logits
            logits = []
            for d in range(n_digit):
                logits_d = self._compute_digit_logits(decoder_hidden[:, d, :], digit=d)
                logits.append(logits_d)
            logits = torch.stack(logits, dim=1)  # [B, n_digit, codebook_size]
        
        output = ModelOutput()
        output.hidden_states = decoder_hidden
        output.logits = logits
        output.past_key_values = present_key_values
        output.drift_gate_probs = drift_probs
        output.drift_probs = drift_probs
        output.drift_scores = drift_scores
        
        return output

    def generate(self, batch, n_return_sequences=1, mode="confidence"):
        """
        使用向量化迭代式掩码填充进行推理生成
        
        Args:
            batch: 包含encoder输入的批次数据
            n_return_sequences: 返回序列数量
            mode: "confidence" 或 "random"
        
        Returns:
            generated_sequences: [B, top_k_final, n_digit]
        """
        from .beam import fast_beam_search_for_eval
        
        # 🚀 确保推理时使用eval模式，关闭dropout
        was_training = self.training
        self.eval()
        if self.drift_moe_enabled:
            self._generation_history_sid = None
            self._generation_history_mask = None
        
        try:
            # 获取encoder输出
            with torch.no_grad():
                encoder_outputs = self.forward(batch, return_loss=False)
                encoder_hidden = encoder_outputs.hidden_states
                if self.drift_moe_enabled and 'history_sid' in batch:
                    self._generation_history_sid = batch['history_sid'].to(encoder_hidden.device)
                    if 'history_mask' in batch:
                        self._generation_history_mask = batch['history_mask'].to(encoder_hidden.device)
                    else:
                        self._generation_history_mask = (self._generation_history_sid >= 0).any(dim=-1)

                # 路由：原生4步 / 随机
                if mode in ("confidence", "random"):
                    generated_sequences = fast_beam_search_for_eval(
                        model=self,
                        encoder_hidden=encoder_hidden,
                        beam_size=n_return_sequences,
                        max_len=self.n_digit,
                        tokenizer=self.tokenizer,
                        mode=mode,
                        rand_cfg=self.config.get("random_beam", {})
                    )
                    return generated_sequences

                # 路由：消融式 1/2/3 步（仅置信度）
                if mode.startswith("confidence_s") and bool(self.config.get('ablate_decode', {}).get('enabled', False)):
                    try:
                        steps = int(mode.split("confidence_s")[-1])
                    except Exception:
                        steps = int(self.config.get('ablate_decode', {}).get('steps_default', 3))
                    if steps >= 4:
                        # 回退到原4步
                        generated_sequences = fast_beam_search_for_eval(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            beam_size=n_return_sequences,
                            max_len=self.n_digit,
                            tokenizer=self.tokenizer,
                            mode="confidence",
                            rand_cfg=self.config.get("random_beam", {})
                        )
                    else:
                        generated_sequences = decode_ablate_confidence(
                            model=self,
                            encoder_hidden=encoder_hidden,
                            tokenizer=self.tokenizer,
                            steps=steps,
                            n_return_sequences=n_return_sequences,
                        )
                    return generated_sequences

                # 兜底：未知模式，走原4步
                generated_sequences = fast_beam_search_for_eval(
                    model=self,
                    encoder_hidden=encoder_hidden,
                    beam_size=n_return_sequences,
                    max_len=self.n_digit,
                    tokenizer=self.tokenizer,
                    mode="confidence",
                    rand_cfg=self.config.get("random_beam", {})
                )
                return generated_sequences
            
        finally:
            if self.drift_moe_enabled:
                self._generation_history_sid = None
                self._generation_history_mask = None
            # 恢复原始训练状态，避免影响后续训练
            if was_training:
                self.train()

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # LN: 有 bias；RMSNorm: 只有 weight（无 bias）
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if hasattr(module, "weight") and module.weight is not None:
                torch.nn.init.ones_(module.weight)
        # 注意：output_adapter如果是Identity()，不需要初始化 
