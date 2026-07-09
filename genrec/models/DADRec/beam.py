# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def _beam_step_select(mode,
                      logp_matrix,      # [B, act, n_digit*VOC]
                      cur_beam_logp,    # [B, act]
                      beam_ids,         # [B, act, n_digit]  (父节点)
                      n_digit, VOC, beam_act,
                      rand_cfg):
    """
    统一的一步分支选择逻辑
    
    Args:
        mode: "confidence" 或 "random"
        logp_matrix: 当前步骤的log概率矩阵 [B, act, n_digit*VOC]
        cur_beam_logp: 当前beam的log概率 [B, act]
        beam_ids: 当前beam的token序列 [B, act, n_digit]
        n_digit: 数字位数
        VOC: 词汇表大小
        beam_act: 活跃beam数量
        rand_cfg: 随机采样配置字典
    
    Returns:
        next_lp: 下一步的log概率 [B, act]
        next_ids: 下一步的token序列 [B, act, n_digit]
    """
    B = logp_matrix.size(0)

    if mode == "confidence":
        # 置信度模式：选择最高概率的路径
        cand_lp  = cur_beam_logp.unsqueeze(-1) + logp_matrix      # logP
        flat_lp  = cand_lp.view(B, -1)
        best_lp, flat_idx = torch.topk(flat_lp, k=beam_act)       # [B, act]
    else:   # "random"
        # 随机模式：使用temperature和top-p/top-k采样
        temperature = rand_cfg.get("temperature", 1.0)
        logits = (cur_beam_logp.unsqueeze(-1) + logp_matrix) / temperature      # [B, act, *]

        # top-k截断
        top_k = rand_cfg.get("top_k")
        if top_k is not None:
            kth_vals, _ = logits.topk(top_k, dim=-1)
            min_valid   = kth_vals[..., -1:].detach()
            logits      = torch.where(logits < min_valid, logits.new_full((), -1e9), logits)

        # top-p (nucleus) 采样
        top_p = rand_cfg.get("top_p")
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

            # 去掉超过阈值的以及其后的 token（保留至少一个）
            sorted_indices_to_remove = cumsum_probs > top_p
            # 把第一个位置强制保留（避免全部被去掉）
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # 还原回原顺序的布尔掩码
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        probs = torch.softmax(logits, dim=-1)                   # 真概率
        flat_prob = probs.view(B, -1)

        # 🚀 修复：保存当前随机种子状态，避免污染训练
        original_state = torch.get_rng_state()
        try:
            # 固定 seed（可选）
            seed = rand_cfg.get("seed")
            if seed is not None:
                torch.manual_seed(seed)

            flat_idx = torch.multinomial(flat_prob, beam_act, replacement=False)  # [B, act]
            idx_rows = torch.arange(B, device=flat_idx.device).unsqueeze(1)
            best_lp  = logits.view(B, -1)[idx_rows, flat_idx]        # 对应 logP
        finally:
            # 🚀 恢复原始随机种子状态
            torch.set_rng_state(original_state)
    # -------------------------------------------------------------------------

    parent   = flat_idx // (n_digit * VOC)
    remain   = flat_idx %  (n_digit * VOC)
    d_pos    = remain // VOC
    tok      = remain %  VOC

    batch_idx = torch.arange(B, device=beam_ids.device).unsqueeze(1)
    next_ids  = beam_ids[batch_idx, parent].clone()
    next_ids.scatter_(2, d_pos.unsqueeze(-1), tok.unsqueeze(-1))
    return best_lp, next_ids


def expand_cross_kv_for_beams(initial_kv_cache, beam_size):
    """
    把第一步得到的cross-KV复制到每个beam，self-KV仍设为None；
    这样DecoderBlock的自注意KV会继续累加，而cross-KV不会重复计算。
    
    Args:
        initial_kv_cache: 初始KV cache
        beam_size: beam大小
    
    Returns:
        扩展后的KV cache
    """
    if initial_kv_cache is None:
        return None

    expanded = []
    for layer_cache in initial_kv_cache:
        if layer_cache is None:
            expanded.append(None)
            continue

        self_kv, cross_kv = layer_cache        # self_kv仅第一步有，后续靠cache累加
        if cross_kv is not None:
            k, v = cross_kv                    # [B, S, d]
            k = k.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *k.shape[1:])
            v = v.unsqueeze(1).repeat(1, beam_size, 1, 1).view(-1, *v.shape[1:])
            cross_kv = (k, v)
        # ⚠ self_kv设None，避免把第一步decoder的token重复broadcast
        expanded.append((None, cross_kv))
    return expanded



def iterative_mask_decode(model, encoder_hidden, n_return_sequences=1, tokenizer=None, mode="confidence", rand_cfg=None):
    """
    向量化迭代式掩码填充解码，完全消除Python循环瓶颈
    
    Args:
        model: DADRec模型
        encoder_hidden: encoder输出 [B, seq_len, emb_dim]
        n_return_sequences: 返回序列数量(会被top_k_final覆盖)
        tokenizer: tokenizer对象
        mode: "confidence" 或 "random"
        rand_cfg: 随机采样配置字典
    
    Returns:
        generated_sequences: [B, top_k_final, n_digit] 生成的序列
    """
    device = encoder_hidden.device
    batch_size = encoder_hidden.size(0)
    n_digit = model.n_digit
    codebook_size = model.codebook_size
    
    # 🚀 从配置中获取向量化beam search参数（支持split-specific配置）
    if hasattr(model, 'config') and 'vectorized_beam_search' in model.config:
        beam_config = model.config['vectorized_beam_search']
        
        # 获取当前split（默认为val）
        split = model.config.get("current_split", "val")   # "val" / "test"
        
        # 检查是否为split-specific配置（支持三种写法）
        if split in beam_config:                           # ← 先查 split-specific
            BEAM_ACT = int(beam_config[split]["beam_act"])
            BEAM_MAX = int(beam_config[split]["beam_max"])
        elif isinstance(beam_config.get("beam_act"), dict): # 兼容另一种写法：beam_act 本身就是 dict
            BEAM_ACT = int(beam_config["beam_act"].get(split,
                                                       beam_config["beam_act"]["val"]))
            BEAM_MAX = int(beam_config["beam_max"].get(split,
                                                       beam_config["beam_max"]["val"]))
        else:                                              # 最后才落回全局
            BEAM_ACT = int(beam_config["beam_act"])
            BEAM_MAX = int(beam_config["beam_max"])
        
        TOP_K_FINAL = min(int(beam_config['top_k_final']), n_return_sequences)
        # 🚀 修复：确保NEG_INF值是float类型，避免YAML字符串问题
        NEG_INF_FP32 = float(beam_config['neg_inf_fp32'])
        NEG_INF_FP16 = float(beam_config['neg_inf_fp16'])
        # 确保beam_act不超过beam_max
        assert BEAM_ACT <= BEAM_MAX, "beam_act should not exceed beam_max"
    else:
        # 🚀 修复：配置统一，不再有fallback
        raise ValueError("Missing 'vectorized_beam_search' configuration in model.config")
    
    # ---------- ① 解析 beam_size（random模式特殊处理） ----------
    if mode == "random":
        # 如果 random_beam 指定了 beam_act/beam_max 就覆盖
        rb_cfg = model.config.get("random_beam", {})
        BEAM_ACT = int(rb_cfg.get("beam_act", BEAM_ACT))
        BEAM_MAX = int(rb_cfg.get("beam_max", BEAM_MAX))
        # 确保beam_act不超过beam_max
        assert BEAM_ACT <= BEAM_MAX, "random_beam.beam_act should not exceed random_beam.beam_max"
    
    # ---------- ② 随机一次列顺序（仅random模式） ----------
    decode_order = None
    if mode == "random":
        # 🚀 修复：保存当前随机种子状态，避免污染训练
        original_state = torch.get_rng_state()
        try:
            seed = model.config.get("random_beam", {}).get("seed")
            if seed is not None:
                torch.manual_seed(seed)
            decode_order = torch.randperm(n_digit).tolist()      # e.g. [1,5,3,7,0,2,6,4]
            if batch_size == 1:  # 只在单样本时打印，避免多worker刷屏
                print(f"[RANDOM_BEAM] 🎲 Decode order: {decode_order}")
        finally:
            # 🚀 恢复原始随机种子状态
            torch.set_rng_state(original_state)
    
    # 常量
    MASK_ID = tokenizer.mask_token if tokenizer is not None else -1
    VOC = codebook_size
    
    # 减少日志噪音
    if batch_size == 1:  # 只在单样本时打印，避免多worker刷屏
        print(f"[VECTORIZED_BEAM] 🚀 Using optimized beam search:")
        print(f"[VECTORIZED_BEAM] BEAM_ACT: {BEAM_ACT}, BEAM_MAX: {BEAM_MAX}, TOP_K_FINAL: {TOP_K_FINAL}")
    
    # Step 0: 全掩码预测，获取所有位置的概率
    with torch.no_grad():
        # 构建mask_positions：全1表示全部被掩码
        mask_positions = torch.ones(batch_size, n_digit, device=device)
        
        # 构建batch
        batch_dict = {
            'decoder_input_ids': torch.zeros(batch_size, n_digit, device=device, dtype=torch.long),
            'encoder_hidden': encoder_hidden,
            'mask_positions': mask_positions
        }
        
        # 前向传播 - 启用KV cache以加速后续推理
        outputs = model.forward_decoder_only(batch_dict, digit=None, use_cache=True)
        all_logits = outputs.logits  # [B, n_digit, codebook_size]
        initial_kv_cache = outputs.past_key_values  # 保存初始KV cache
        
        # 计算log probabilities
        all_log_probs = F.log_softmax(all_logits, dim=-1)  # [B, n_digit, codebook_size]
        
        if mode == "random":
            # === random模式：只看第一列 ===
            first_col = decode_order[0]
            probs_col = all_log_probs[:, first_col, :]          # [B, VOC]
            top_k_probs, top_k_idx = torch.topk(probs_col, k=BEAM_ACT, dim=-1)  # [B, BEAM_ACT]
            
            # 解析位置和token
            first_col_tensor = torch.full((batch_size, BEAM_ACT), first_col, device=device, dtype=torch.long)
            first_token = top_k_idx
        else:
            # === confidence模式：全局top-k ===
            # 拼接所有位置的概率: [B, n_digit * codebook_size]  
            flattened_log_probs = all_log_probs.view(batch_size, -1)
            
            # 取top BEAM_ACT个候选
            top_k_probs, top_k_indices = torch.topk(flattened_log_probs, k=BEAM_ACT)
            
            # 解析位置和token
            first_col_tensor = top_k_indices // VOC      # 第几个digit [B, BEAM_ACT]
            first_token = top_k_indices % VOC     # codebook内的ID [B, BEAM_ACT]
        
        # 🚀 固定大小beam tensor，一次分配（关键优化）
        beam_ids = torch.full((batch_size, BEAM_MAX, n_digit), MASK_ID, 
                             dtype=torch.long, device=device)
        
        # 确定NEG_INF值
        NEG_INF = NEG_INF_FP16 if top_k_probs.dtype == torch.float16 else NEG_INF_FP32
        beam_logp = torch.full((batch_size, BEAM_MAX), NEG_INF, 
                              dtype=top_k_probs.dtype, device=device)
        
        # 填充第一步的结果
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
        beam_indices = torch.arange(BEAM_ACT, device=device).unsqueeze(0)     # [1, BEAM_ACT]
        
        beam_ids[batch_indices, beam_indices, first_col_tensor] = first_token
        beam_logp[:, :BEAM_ACT] = top_k_probs
        
        # 🚀 修复：扩展到BEAM_MAX确保充足容量
        encoder_hidden_expanded = encoder_hidden.unsqueeze(1).repeat(1, BEAM_MAX, 1, 1)
        encoder_hidden_expanded = encoder_hidden_expanded.view(-1, encoder_hidden.size(1), encoder_hidden.size(2))
        
        # Step-0结束后，生成一次broadcast后的cache供后续复用
        kv_cache_for_act = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)
        kv_cache_final = expand_cross_kv_for_beams(initial_kv_cache, BEAM_ACT)  # 用于最后一步
    
    # Steps 1-2: 向量化beam扩展（完全消除Python循环）
    if mode == "random":
        # === random模式：按decode_order循环 ===
        for step, cur_col in enumerate(decode_order[1:], 1):
            with torch.no_grad():
                # 只使用前BEAM_ACT个有效beam
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]
                
                # 构建当前状态的mask_positions
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]
                
                # 重塑为decoder输入格式
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                
                # 🚀 使用预生成的KV cache，实现真正的cache复用
                expanded_kv_cache = kv_cache_for_act
                
                # 构建batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # 只使用前BEAM_ACT部分
                    'mask_positions': mask_pos_flat
                }
                
                # 前向传播
                outputs = model.forward_decoder_only(batch_dict, digit=None, 
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]
                
                # 重塑为beam维度
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
                
                # 🚀 向量化掩码处理（核心优化）
                all_log_probs = F.log_softmax(all_logits, dim=-1)
                
                # 只考虑被掩码的位置
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF
                
                # === random模式：只看当前列 ===
                logits = masked_log_probs[:, :, cur_col, :]                     # [B, BEAM_ACT, VOC]
                
                joint_lp = logits + active_beam_logp.unsqueeze(-1)              # [B, BEAM_ACT, VOC]
                flat_lp  = joint_lp.view(batch_size, -1)                        # [B, BEAM_ACT*VOC]
                best_lp, flat_idx = torch.topk(flat_lp, k=BEAM_ACT)            # ← top-k，不采样
                
                # 解析索引
                parent_beam_ids = flat_idx // VOC                               # [B, BEAM_ACT]
                token_ids = flat_idx % VOC                                      # [B, BEAM_ACT]
                
                # 更新beam
                batch_range = torch.arange(batch_size, device=device).unsqueeze(1)  # [B, 1]
                new_beam_ids = active_beam_ids[batch_range, parent_beam_ids]        # [B, BEAM_ACT, n_digit]
                new_beam_ids.scatter_(2, torch.full((batch_size, BEAM_ACT), cur_col, device=device, dtype=torch.long).unsqueeze(-1), token_ids.unsqueeze(-1))
                
                # 更新beam状态
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_lp
                
                # 清空无效beam（保持BEAM_MAX大小）
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    else:
        # === confidence模式：原有逻辑 ===
        for step in range(1, n_digit - 1):
            with torch.no_grad():
                # 只使用前BEAM_ACT个有效beam
                active_beam_ids = beam_ids[:, :BEAM_ACT, :]      # [B, BEAM_ACT, n_digit]
                active_beam_logp = beam_logp[:, :BEAM_ACT]       # [B, BEAM_ACT]
                
                # 构建当前状态的mask_positions
                mask_positions = (active_beam_ids == MASK_ID).float()  # [B, BEAM_ACT, n_digit]
                
                # 重塑为decoder输入格式
                decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                mask_pos_flat = mask_positions.view(-1, n_digit)  # [B*BEAM_ACT, n_digit]
                
                # 🚀 使用预生成的KV cache，实现真正的cache复用
                expanded_kv_cache = kv_cache_for_act
                
                # 构建batch
                batch_dict = {
                    'decoder_input_ids': decoder_input,
                    'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # 只使用前BEAM_ACT部分
                    'mask_positions': mask_pos_flat
                }
                
                # 前向传播
                outputs = model.forward_decoder_only(batch_dict, digit=None, 
                                                   past_key_values=expanded_kv_cache, use_cache=True)
                all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]
                
                # 重塑为beam维度
                all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
                
                # 🚀 向量化掩码处理（核心优化）
                all_log_probs = F.log_softmax(all_logits, dim=-1)
                
                # 只考虑被掩码的位置
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, BEAM_ACT, n_digit, 1]
                masked_log_probs = all_log_probs + (1 - mask_expanded) * NEG_INF
                
                # 拼接所有可能的候选：[B, BEAM_ACT, n_digit * codebook_size]
                flattened_log_probs = masked_log_probs.view(batch_size, BEAM_ACT, -1)
                
                # 🚀 使用统一的分支选择逻辑
                best_logprobs, new_beam_ids = _beam_step_select(
                    mode=mode,
                    logp_matrix=flattened_log_probs,          # [B, act, n_digit*VOC]
                    cur_beam_logp=active_beam_logp,           # [B, act]
                    beam_ids=active_beam_ids,                 # [B, act, n_digit]
                    n_digit=n_digit, VOC=VOC, beam_act=BEAM_ACT,
                    rand_cfg=rand_cfg or {}
                )
                
                # 更新beam状态
                beam_ids[:, :BEAM_ACT, :] = new_beam_ids
                beam_logp[:, :BEAM_ACT] = best_logprobs
                
                # 清空无效beam（保持BEAM_MAX大小）
                if BEAM_ACT < BEAM_MAX:
                    beam_ids[:, BEAM_ACT:, :] = MASK_ID
                    beam_logp[:, BEAM_ACT:] = NEG_INF
    
    # 最终步骤: 填充最后一个位置并选择top-K
    with torch.no_grad():
        if mode == "random":
            # === random模式：已经通过循环填完了所有位置，直接使用当前结果 ===
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            final_beam_logp = beam_logp[:, :BEAM_ACT]
        else:
            # === confidence模式：需要填充最后一个位置 ===
            # 只处理前BEAM_ACT个beam
            active_beam_ids = beam_ids[:, :BEAM_ACT, :]
            active_beam_logp = beam_logp[:, :BEAM_ACT]
            
            # 找到每个beam的最后一个MASK位置
            mask_positions = (active_beam_ids == MASK_ID).float()
            
            # 构建decoder输入
            decoder_input = torch.clamp(active_beam_ids, min=0).view(-1, n_digit)
            mask_pos_flat = mask_positions.view(-1, n_digit)
            
            # 使用预生成的KV cache用于最终步骤
            final_expanded_kv_cache = kv_cache_final
            
            batch_dict = {
                'decoder_input_ids': decoder_input,
                'encoder_hidden': encoder_hidden_expanded[:batch_size * BEAM_ACT],  # 只使用前BEAM_ACT部分
                'mask_positions': mask_pos_flat
            }
            
            # 获取所有位置的logits
            outputs = model.forward_decoder_only(batch_dict, digit=None, 
                                               past_key_values=final_expanded_kv_cache, use_cache=True)
            all_logits = outputs.logits  # [B*BEAM_ACT, n_digit, codebook_size]
            
            # 重塑并计算log probs
            all_logits = all_logits.view(batch_size, BEAM_ACT, n_digit, codebook_size)
            all_log_probs = F.log_softmax(all_logits, dim=-1)
            
            # 找到每个beam需要填充的最后一个位置
            last_mask_pos = torch.argmax(mask_positions.float(), dim=-1)  # [B, BEAM_ACT]
            
            # 为每个beam选择对应位置的最佳token
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, BEAM_ACT)
            beam_idx = torch.arange(BEAM_ACT, device=device).unsqueeze(0).expand(batch_size, -1)
            
            final_logits = all_log_probs[batch_idx, beam_idx, last_mask_pos]  # [B, BEAM_ACT, codebook_size]
            best_token_logprobs, best_tokens = torch.max(final_logits, dim=-1)  # [B, BEAM_ACT]
            
            # 更新最后的token
            active_beam_ids.scatter_(2, last_mask_pos.unsqueeze(-1), best_tokens.unsqueeze(-1))
            final_beam_logp = active_beam_logp + best_token_logprobs
        
        # 🚀 灵活的去重策略
        dedup_strategy = "simple"  # 默认使用 simple 去重
        if hasattr(model, 'config') and 'dedup_strategy' in model.config:
            dedup_strategy = model.config['dedup_strategy']
        
        if dedup_strategy == "none":
            # 策略1: 不去重，直接选择top-K
            top_logprobs, top_indices = torch.topk(final_beam_logp, k=min(TOP_K_FINAL, BEAM_ACT), dim=-1)
            batch_range = torch.arange(batch_size, device=device).unsqueeze(1)
            final_sequences = active_beam_ids[batch_range, top_indices]  # [B, TOP_K_FINAL, n_digit]
            
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} sequences (no deduplication)")
                
        elif dedup_strategy == "simple":
            # 策略2: 简单去重 + 合法性检查（改进的方法）
            # ① 需要 tokenizer 传进来
            assert tokenizer is not None, "tokenizer is required for legality check"
            
            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]
                
                # 按概率排序，然后简单去重 + 合法性检查
                sorted_indices = torch.argsort(batch_logprobs, descending=True)
                unique_sequences = []
                
                for idx in sorted_indices:
                    seq = batch_sequences[idx]
                    # --------- 新增：合法性检查 ----------
                    is_legal = tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                    if not is_legal:
                        continue  # 直接跳过非法序列
                    # ------------------------------------
                    is_duplicate = any(torch.equal(seq, existing) for existing in unique_sequences)
                    if not is_duplicate:
                        unique_sequences.append(seq)
                        if len(unique_sequences) >= TOP_K_FINAL:
                            break
                            
                # 填充不足的部分（确保填充的序列也是合法的）
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # 如果有合法序列，重复最后一个
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # 如果没有合法序列，找一个合法的填充
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # 如果还是找不到合法序列，用第一个（虽然不合法，但总比崩溃好）
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])
                
                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)
            
            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (simple deduplication + legality check)")
                
        else:  # weighted
            # 策略3: 概率加权去重 + 合法性检查（改进的方法）
            # ① 需要 tokenizer 传进来
            assert tokenizer is not None, "tokenizer is required for legality check"
            
            final_sequences = []
            for b in range(batch_size):
                batch_sequences = active_beam_ids[b]  # [BEAM_ACT, n_digit]
                batch_logprobs = final_beam_logp[b]   # [BEAM_ACT]
                
                # 构建序列到概率的映射，累加重复序列的概率（只考虑合法序列）
                seq_to_logprob = {}
                for i in range(BEAM_ACT):
                    seq_tuple = tuple(batch_sequences[i].cpu().tolist())
                    # --------- 新增：合法性检查 ----------
                    is_legal = tokenizer.codebooks_to_item_id(list(seq_tuple)) is not None
                    if not is_legal:
                        continue  # 直接跳过非法序列
                    # ------------------------------------
                    if seq_tuple in seq_to_logprob:
                        # 重复序列：使用log-sum-exp累加概率（更稳定）
                        seq_to_logprob[seq_tuple] = torch.logaddexp(
                            seq_to_logprob[seq_tuple], 
                            batch_logprobs[i]
                        )
                    else:
                        seq_to_logprob[seq_tuple] = batch_logprobs[i]
                
                # 按累加后的概率排序
                sorted_items = sorted(seq_to_logprob.items(), 
                                    key=lambda x: x[1].item(), reverse=True)
                
                # 选择前TOP_K_FINAL个不重复序列（已经按加权概率排序）
                unique_sequences = []
                for seq_tuple, _ in sorted_items[:TOP_K_FINAL]:
                    seq_tensor = torch.tensor(seq_tuple, device=device, dtype=torch.long)
                    unique_sequences.append(seq_tensor)
                
                # 填充不足的部分（确保填充的序列也是合法的）
                while len(unique_sequences) < TOP_K_FINAL:
                    if unique_sequences:
                        # 如果有合法序列，重复最后一个
                        unique_sequences.append(unique_sequences[-1])
                    else:
                        # 如果没有合法序列，找一个合法的填充
                        for idx in range(BEAM_ACT):
                            seq = batch_sequences[idx]
                            if tokenizer.codebooks_to_item_id(seq.tolist()) is not None:
                                unique_sequences.append(seq)
                                break
                        # 如果还是找不到合法序列，用第一个（虽然不合法，但总比崩溃好）
                        if not unique_sequences:
                            unique_sequences.append(batch_sequences[0])
                
                batch_final = torch.stack(unique_sequences[:TOP_K_FINAL])
                final_sequences.append(batch_final)
            
            final_sequences = torch.stack(final_sequences)
            if batch_size == 1:
                print(f"[VECTORIZED_BEAM] ✅ Generated {final_sequences.shape[1]} unique sequences (probability-weighted deduplication + legality check)")
    
    # ------- 计算当前batch的统计信息 -------
    if tokenizer is not None:  # 不再限制 batch_size==1
        # 修复合法率计算：使用序列数作为分母，而不是token数
        total_seqs = final_sequences.numel() // n_digit
        legal_final = sum(tokenizer.codebooks_to_item_id(seq.tolist()) is not None
                          for seq in final_sequences.view(-1, n_digit))
        final_legal_ratio = legal_final / total_seqs

        # 修复重复率计算：使用正确的公式
        unique_seqs = len({tuple(seq.tolist()) for seq in final_sequences.view(-1, n_digit)})
        duplicate_ratio = 1 - unique_seqs / total_seqs

        # 返回统计信息供evaluator使用，而不是直接打印
        return final_sequences, final_legal_ratio, duplicate_ratio
    # --------------------------------
    
    return final_sequences


def fast_beam_search_for_eval(model, encoder_hidden, beam_size=10, max_len=4, tokenizer=None, mode="confidence", rand_cfg=None):
    """
    专门用于验证的快速向量化beam search
    采用与TensorFlow一致的策略：前3步固定512beam，最后取top-K
    
    Args:
        model: DADRec模型
        encoder_hidden: Encoder输出 [batch_size, seq_len, hidden_dim]
        beam_size: 最终beam大小（会被TOP_K_FINAL覆盖）
        max_len: 最大生成长度
        tokenizer: Tokenizer
        mode: "confidence" 或 "random"
        rand_cfg: 随机采样配置字典
    
    Returns:
        torch.Tensor: 生成的token序列 [batch_size, TOP_K_FINAL, max_len]
    """
    # 直接调用向量化的iterative_mask_decode
    result = iterative_mask_decode(
        model=model,
        encoder_hidden=encoder_hidden,
        n_return_sequences=beam_size,
        tokenizer=tokenizer,
        mode=mode,
        rand_cfg=rand_cfg or {}
    )
    
    # 处理返回值：可能是元组（序列+统计信息）或只是序列
    if isinstance(result, tuple):
        return result[0]  # 只返回序列部分
    else:
        return result


 