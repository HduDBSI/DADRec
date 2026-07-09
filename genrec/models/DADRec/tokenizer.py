# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import pickle
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class DADRecTokenizer(AbstractTokenizer):
    """
    DADRec Tokenizer for Diffusion-based Generative Recommendation Model
    
    Special tokens:
    - PAD=0, BOS=1, EOS=2, SID_OFFSET=3
    
    SID Configuration:
    - n_digit: configurable (e.g., 4, 8, 12), codebook_size=256
    - vocab_size = 3 + n_digit * codebook_size
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        # 兜底，避免 KeyError
        config.setdefault('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        config.setdefault('num_proc', 1)

        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        
        # 选择量化器：opq_pq(默认) | rq_kmeans | none(随机)
        self.sid_quantizer = config.get('sid_quantizer', 'opq_pq')
        assert self.sid_quantizer in ('opq_pq', 'rq_kmeans', 'none'), \
            f"sid_quantizer must be one of ['opq_pq','rq_kmeans','none'], got {self.sid_quantizer}"
        
        # 🚀 兼容旧配置：仅在 opq_pq 模式下使用 disable_opq/index_factory
        if self.sid_quantizer == 'opq_pq':
            use_opq = not config.get('disable_opq', False)
            if use_opq:
                self.index_factory = f'OPQ{config["n_digit"]},IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
            else:
                self.index_factory = f'IVF1,PQ{config["n_digit"]}x{self.n_codebook_bits}'
        elif self.sid_quantizer == 'rq_kmeans':
            self.index_factory = f'RQKMEANS{config["n_digit"]}x{self.n_codebook_bits}'
        else:  # 'none'
            self.index_factory = f'RAND{config["n_digit"]}x{self.n_codebook_bits}'

        # 先初始化父类，保证 self.config / self.logger 等字段可用
        super(DADRecTokenizer, self).__init__(config, dataset)
        
        # 现在再写日志
        self.log(f'[TOKENIZER] Index factory: {self.index_factory}')
        self.dataset = dataset  # 添加dataset引用
        self.item2id = dataset.item2id
        self.id2item = dataset.id_mapping['id2item']
        self.sid_code_centroids = None
        self.item_id2opq_subvectors = None
        
        # Special tokens - 简化token ID分配
        self.pad_token = 0
        self.bos_token = 1
        self.eos_token = 2
        self.mask_token = -1  # MASK token用于推理，不在vocab中
        self.sid_offset = 3  # SID token从3开始
        
        self.item2tokens = self._init_tokenizer(dataset)
        
        # Create reverse mapping for inference (如果还没有创建的话)
        if not hasattr(self, 'tokens2item'):
            self.tokens2item = self._create_reverse_mapping()
        
        # Set collate functions
        from genrec.models.DADRec.collate import collate_fn_train, collate_fn_val, collate_fn_test
        self.collate_fn = {
            'train': collate_fn_train,
            'val': collate_fn_val,
            'test': collate_fn_test
        }

    def _use_pre_opq_hint_target(self) -> bool:
        cadd_cfg = self.config.get('cadd', {}) or {}
        aux_target = str(cadd_cfg.get('aux_target', 'sid_embedding')).lower()
        return bool(cadd_cfg.get('enabled', False)) and aux_target in (
            'pre_opq', 'pre_opq_semantic', 'sent_emb', 'semantic'
        )

    def _use_opq_subvector_hint_target(self) -> bool:
        cadd_cfg = self.config.get('cadd', {}) or {}
        aux_target = str(cadd_cfg.get('aux_target', 'sid_embedding')).lower()
        return bool(cadd_cfg.get('enabled', False)) and aux_target in (
            'opq_subvector', 'opq_subvectors', 'pre_opq_subvector', 'pre_opq_subvectors'
        )

    @property
    def n_digit(self):
        return self.config['n_digit']

    @property
    def codebook_size(self):
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        return 1 + self.n_digit  # [BOS] + n_digit SID tokens

    @property
    def vocab_size(self) -> int:
        return 3 + self.n_digit * self.codebook_size  # PAD(0) + BOS(1) + EOS(2) + SID tokens

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """编码句子嵌入：支持任意 Hugging Face SentenceTransformer 模型 id，并做向量归一化"""
        assert self.config['metadata'] == 'sentence', \
            'DADRecTokenizer only supports sentence metadata.'

        meta_sentences = []
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        # 接受任意HF模型id（如 Alibaba-NLP/gte-large-en-v1.5 或 BAAI/bge-large-en-v1.5）
        model_id = self.config['sent_emb_model']
        sent_emb_model = SentenceTransformer(model_id, trust_remote_code=True).to(self.config['device'])

        # 直接encode（GTE/BGE无需前缀），并进行L2归一化
        sent_embs = sent_emb_model.encode(
            meta_sentences,
            convert_to_numpy=True,
            batch_size=self.config['sent_emb_batch_size'],
            show_progress_bar=True,
            device=self.config['device'],
            normalize_embeddings=True,
        )

        # 按模型basename分别落盘，避免不同模型冲突
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """获取训练用的商品"""
        items_for_training = set()
        
        # 首先触发数据集分割（如果还没有分割）
        split_data = dataset.split()
        
        # 从训练集中收集所有items
        if 'train' in split_data:
            train_dataset = split_data['train']
            # train_dataset是Hugging Face Dataset对象
            if hasattr(train_dataset, 'column_names') and 'item_seq' in train_dataset.column_names:
                # 遍历所有item_seq
                for item_seq in train_dataset['item_seq']:
                    if isinstance(item_seq, (list, tuple)):
                        items_for_training.update(item_seq)
                    else:
                        items_for_training.add(item_seq)
        
        # 修复：确保mask大小与sent_embs匹配
        # sent_embs只包含item_id从1到n_items-1的商品
        n_sent_embs = dataset.n_items - 1  # 与_encode_sent_emb中的range(1, dataset.n_items)匹配
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {n_sent_embs}')
        self.log(f'[TOKENIZER] Training items sample: {list(items_for_training)[:10]}')
        
        mask = np.zeros(n_sent_embs, dtype=bool)
        for item in items_for_training:
            item_id = dataset.item2id[item]
            if 1 <= item_id < dataset.n_items:  # 确保item_id在有效范围内
                mask[item_id - 1] = True  # 转换为0-based索引
        
        self.log(f'[TOKENIZER] Mask shape: {mask.shape}, True count: {np.sum(mask)}')
        return mask

    def _generate_semantic_id_opq(
        self,
        sent_embs,
        sem_ids_path,
        train_mask,
        opq_subvectors_path=None,
        pq_centroids_path=None,
    ):
        """使用OPQ/PQ生成语义ID（兼容 disable_opq），并用 invlists 的 ids 对齐。"""
        import faiss

        # 调试信息
        self.log(f'[TOKENIZER] sent_embs shape: {sent_embs.shape}')
        self.log(f'[TOKENIZER] train_mask shape: {train_mask.shape}')
        self.log(f'[TOKENIZER] train_mask True count: {np.sum(train_mask)}')

        # 构建索引
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        self.log(f'[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        self._save_opq_subvector_targets(
            index=index,
            sent_embs=sent_embs,
            opq_subvectors_path=opq_subvectors_path,
            pq_centroids_path=pq_centroids_path,
        )

        # 兼容 IndexPreTransform 与非 PreTransform
        if isinstance(index, faiss.IndexPreTransform):
            ivf_index = faiss.downcast_index(index.index)
        else:
            ivf_index = faiss.downcast_index(index)

        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        # 取 codes 与 ids，并保持同序对齐
        codes_ptr = invlists.get_codes(0)
        ids_ptr = invlists.get_ids(0)
        pq_codes_u8 = faiss.rev_swig_ptr(codes_ptr, ls * invlists.code_size)
        ids = faiss.rev_swig_ptr(ids_ptr, ls).copy()
        pq_codes_u8 = pq_codes_u8.reshape(-1, invlists.code_size)

        # 解析 PQ Code
        faiss_sem_ids = []
        n_bytes = invlists.code_size
        for u8code in pq_codes_u8:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for _ in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)

        # 用 ids 对齐 item 顺序
        item2sem_ids = {}
        for pos, iid0 in enumerate(ids):
            item = self.id2item[int(iid0) + 1]
            item2sem_ids[item] = tuple(int(v) for v in faiss_sem_ids[pos])

        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _save_opq_subvector_targets(self, index, sent_embs, opq_subvectors_path=None, pq_centroids_path=None):
        """Save OPQ-rotated pre-quantization subvectors and fixed PQ centroids."""
        if opq_subvectors_path is None and pq_centroids_path is None:
            return
        import faiss

        sent_embs = sent_embs.astype(np.float32, copy=False)
        rotated = sent_embs
        inner_index = index

        if isinstance(index, faiss.IndexPreTransform):
            if index.chain.size() > 0:
                vt = faiss.downcast_VectorTransform(index.chain.at(0))
                if hasattr(vt, 'apply_py'):
                    rotated = vt.apply_py(sent_embs).astype(np.float32, copy=False)
            inner_index = faiss.downcast_index(index.index)
        else:
            inner_index = faiss.downcast_index(index)

        if rotated.shape[1] % self.n_digit != 0:
            raise ValueError(
                f"OPQ rotated dim {rotated.shape[1]} must be divisible by n_digit={self.n_digit}"
            )
        sub_dim = rotated.shape[1] // self.n_digit
        item_id2opq_subvectors = np.zeros(
            (self.dataset.n_items, self.n_digit, sub_dim),
            dtype=np.float32
        )
        item_id2opq_subvectors[1:] = rotated.reshape(-1, self.n_digit, sub_dim)

        ivf_index = faiss.downcast_index(inner_index)
        if not hasattr(ivf_index, 'pq'):
            raise ValueError("Expected an IndexIVFPQ-like FAISS index with a pq field")
        pq = ivf_index.pq
        centroids = faiss.vector_to_array(pq.centroids).astype(np.float32, copy=False)
        centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
        if centroids.shape[0] != self.n_digit or centroids.shape[1] != self.codebook_size:
            raise ValueError(
                f"PQ centroids shape {centroids.shape} does not match "
                f"(n_digit={self.n_digit}, codebook_size={self.codebook_size}, sub_dim={sub_dim})"
            )

        if opq_subvectors_path is not None:
            os.makedirs(os.path.dirname(opq_subvectors_path), exist_ok=True)
            np.save(opq_subvectors_path, item_id2opq_subvectors)
            self.log(f'[TOKENIZER] Saved OPQ subvectors to {opq_subvectors_path}')
        if pq_centroids_path is not None:
            os.makedirs(os.path.dirname(pq_centroids_path), exist_ok=True)
            np.save(pq_centroids_path, centroids)
            self.log(f'[TOKENIZER] Saved OPQ/PQ centroids to {pq_centroids_path}')

    def _generate_semantic_id_random(self, sem_ids_path, n_items, seed=12345):
        """为每个商品随机生成 n_digit 个 codebook ID（均匀[0, K-1]）。"""
        rng = np.random.default_rng(seed)
        item2sem_ids = {}
        for i in range(1, n_items):
            item = self.id2item[i]
            codes = rng.integers(low=0, high=self.codebook_size, size=self.n_digit, endpoint=False, dtype=np.int64)
            item2sem_ids[item] = tuple(int(c) for c in codes.tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _generate_semantic_id_rq_kmeans(self, sent_embs, sem_ids_path, train_mask):
        """使用 Residual Quantization（KMeans）生成语义ID。"""
        import faiss
        d = sent_embs.shape[1]
        K = self.codebook_size
        niter = int(self.config.get('rq_kmeans_niters', 20))
        seed = int(self.config.get('rq_kmeans_seed', 1234))
        
        # 初始化残差为原始向量
        residuals = sent_embs.copy().astype(np.float32, copy=False)
        codes_all = np.zeros((sent_embs.shape[0], self.n_digit), dtype=np.int64)
        
        for stage in range(self.n_digit):
            kmeans = faiss.Kmeans(d=d, k=K, niter=niter, verbose=False, seed=seed + stage)
            kmeans.train(residuals[train_mask])
            # In current Faiss Python, Kmeans.centroids is already a numpy array
            centroids = np.asarray(kmeans.centroids, dtype=np.float32)
            if centroids.ndim == 1:
                centroids = centroids.reshape(K, d)
            elif centroids.shape == (d, K):
                centroids = centroids.T
            assert centroids.shape == (K, d), f"centroids shape {centroids.shape} != {(K, d)}"
            
            # 为全部样本分配最近质心
            index = faiss.IndexFlatL2(d)
            index.add(centroids)
            D, I = index.search(residuals, 1)  # I: [N, 1]
            codes_all[:, stage] = I[:, 0].astype(np.int64)
            
            # 更新残差
            residuals = residuals - centroids[I[:, 0]]
        
        # 转成 dict
        item2sem_ids = {}
        for i in range(codes_all.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(int(v) for v in codes_all[i].tolist())
        os.makedirs(os.path.dirname(sem_ids_path), exist_ok=True)
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """将语义ID转换为token"""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            # 修复：重新引入offset，避免与PAD/BOS冲突
            # 每个digit的codebook ID加上对应的offset
            tokens = [t + self.sid_offset + d * self.codebook_size 
                     for d, t in enumerate(tokens)]
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """初始化tokenizer"""
        # 构建路径 - 修复：使用类名和category
        dataset_name = dataset.__class__.__name__  # 使用类名，如"AmazonReviews2014"
        
        # 如果有category属性，加入路径中
        if hasattr(dataset, 'category') and dataset.category:
            cache_dir = os.path.join(
                dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )
        
        # 确保缓存目录存在
        os.makedirs(cache_dir, exist_ok=True)

        # 加载语义ID（在文件名中加入 PCA 维度与量化器标签，避免配置冲突）
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        sem_ids_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}.sem_ids'
        )
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        opq_subvectors_path = os.path.join(cache_dir, f'item_id2opq_subvectors_{map_tag}.npy')
        pq_centroids_path = os.path.join(cache_dir, f'opq_pq_centroids_{map_tag}.npy')

        # 🚀 新增：检查是否需要强制重新生成量化结果
        force_regenerate = self.config.get('force_regenerate_opq', False)
        
        # 两份嵌入文件：raw 和 pca 版本，避免命名歧义与冲突
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        raw_path = os.path.join(
            cache_dir,
            f'{model_basename}_raw_d{self.config["sent_emb_dim"]}.sent_emb'
        )
        pca_path = os.path.join(
            cache_dir,
            f'{model_basename}_pca{self.config["sent_emb_pca"]}.sent_emb'
        )

        # 如果量化器需要句向量，则准备句向量；none 模式无需
        sent_embs = None
        self.pre_opq_sem_embs = None
        self.pre_opq_sem_emb_dim = 0
        if self.sid_quantizer == 'opq_pq':
            # opq_pq：允许使用 PCA（保持原逻辑）
            if self.config['sent_emb_pca'] > 0 and os.path.exists(pca_path):
                self.log(f'[TOKENIZER] Loading PCA-ed sentence embeddings from {pca_path}...')
                sent_embs = np.fromfile(pca_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_pca']
                )
            elif os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                raw_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                raw_embs = self._encode_sent_emb(dataset, raw_path)
                if self.config['sent_emb_pca'] > 0:
                    self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                    training_item_mask = self._get_items_for_training(dataset)
                    pca.fit(raw_embs[training_item_mask])
                    sent_embs = pca.transform(raw_embs)
                    sent_embs = sent_embs.astype(np.float32, copy=False)
                    if self.config.get('normalize_after_pca', True):
                        norms = np.linalg.norm(sent_embs, axis=1, keepdims=True) + 1e-12
                        sent_embs = sent_embs / norms
                    sent_embs.tofile(pca_path)
                else:
                    sent_embs = raw_embs
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')
        elif self.sid_quantizer == 'rq_kmeans':
            # rq_kmeans：不进行 PCA，直接使用 RAW（满足你的需求）
            if os.path.exists(raw_path):
                self.log(f'[TOKENIZER] Loading RAW sentence embeddings from {raw_path}...')
                sent_embs = np.fromfile(raw_path, dtype=np.float32).reshape(
                    -1, self.config['sent_emb_dim']
                )
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings (RAW, no PCA for RQ-KMeans)...')
                sent_embs = self._encode_sent_emb(dataset, raw_path)
            self.log(f'[TOKENIZER] Sentence embeddings shape (RAW): {sent_embs.shape}')

        if sent_embs is not None:
            self.pre_opq_sem_embs = sent_embs.astype(np.float32, copy=False)
            self.pre_opq_sem_emb_dim = int(self.pre_opq_sem_embs.shape[1])
            self.log(f'[TOKENIZER] Pre-OPQ semantic targets shape: {self.pre_opq_sem_embs.shape}')
        if (self._use_pre_opq_hint_target() or self._use_opq_subvector_hint_target()) and self.pre_opq_sem_embs is None:
            raise ValueError(
                "cadd.aux_target=pre_opq requires sentence embeddings, "
                f"but sid_quantizer={self.sid_quantizer} did not load them"
            )

        # 🚀 生成或加载量化结果
        needs_opq_subvector_cache = (
            self._use_opq_subvector_hint_target()
            and (
                not os.path.exists(opq_subvectors_path)
                or not os.path.exists(pq_centroids_path)
            )
        )
        if force_regenerate or not os.path.exists(sem_ids_path) or needs_opq_subvector_cache:
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerating quantization results ({self.sid_quantizer})...')
            elif needs_opq_subvector_cache:
                self.log(f'[TOKENIZER] OPQ subvector targets not found, regenerating ({self.sid_quantizer})...')
            else:
                self.log(f'[TOKENIZER] Quantization results not found, generating ({self.sid_quantizer})...')
            training_item_mask = self._get_items_for_training(dataset)
            if self.sid_quantizer == 'opq_pq':
                self._generate_semantic_id_opq(
                    sent_embs,
                    sem_ids_path,
                    training_item_mask,
                    opq_subvectors_path=opq_subvectors_path,
                    pq_centroids_path=pq_centroids_path,
                )
            elif self.sid_quantizer == 'rq_kmeans':
                self._generate_semantic_id_rq_kmeans(sent_embs, sem_ids_path, training_item_mask)
            else:  # 'none'
                self._generate_semantic_id_random(
                    sem_ids_path, n_items=self.dataset.n_items,
                    seed=int(self.config.get('sid_random_seed', 12345))
                )
        else:
            self.log(f'[TOKENIZER] Using existing quantization results from {sem_ids_path}')

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        # 🚀 映射文件名：复用前面构造的 quant_tag
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        fwd_path = os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy')
        inv_path = os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl')
        
        # 🚀 修复①：处理映射文件的一致性
        if force_regenerate:
            # 强制重新生成时，直接忽略旧文件，让下面逻辑走"重新保存"
            fwd_exists = inv_exists = False
            self.log(f'[TOKENIZER] Force regenerate enabled, ignoring existing mapping files')
        else:
            fwd_exists = os.path.exists(fwd_path)
            inv_exists = os.path.exists(inv_path)
        
        if fwd_exists and inv_exists:
            # ---------- ① 文件已存在 ----------
            self.log(f'[TOKENIZER] Loading existing mappings for tag: {map_tag} from {fwd_path}')
            
            # 重新构建item2tokens映射
            item_id2tokens = np.load(fwd_path)
            item2tokens = {}
            for iid, toks in enumerate(item_id2tokens):
                if iid == 0:  # PAD行全0，跳过
                    continue
                item2tokens[self.id2item[iid]] = tuple(toks.tolist())
            
            # 加载倒排索引
            with open(inv_path, 'rb') as f:
                self.tokens2item = pickle.load(f)
                
            self.log(f'[TOKENIZER] Successfully loaded {len(item2tokens)} item mappings')
        else:
            # ---------- ② 文件不存在或强制重新生成，需要重新生成 ----------
            if force_regenerate:
                self.log(f'[TOKENIZER] Force regenerate enabled, generating new mappings')
            else:
                self.log(f'[TOKENIZER] No existing mappings found for {self.n_digit}-digit, will generate new ones')
            
            # 无论是文件不存在还是 forceRegenerate，都按新的 item2tokens 保存
            self.item2tokens = item2tokens
            self.tokens2item = self._create_reverse_mapping()
            self._save_mappings()  # 只在"新建"时真正落盘

        # ---- ③ 统一：把映射挂到实例属性再返回 ----
        # 注意：在"文件已存在"分支中，需要设置self.item2tokens
        if not hasattr(self, 'item2tokens'):
            self.item2tokens = item2tokens
        if os.path.exists(opq_subvectors_path):
            self.item_id2opq_subvectors = np.load(opq_subvectors_path).astype(np.float32, copy=False)
            self.log(f'[TOKENIZER] Loaded OPQ subvectors from {opq_subvectors_path}: {self.item_id2opq_subvectors.shape}')
        if os.path.exists(pq_centroids_path):
            self.sid_code_centroids = np.load(pq_centroids_path).astype(np.float32, copy=False)
            self.log(f'[TOKENIZER] Loaded OPQ/PQ centroids from {pq_centroids_path}: {self.sid_code_centroids.shape}')
        elif self._use_opq_subvector_hint_target():
            raise ValueError(
                f"cadd.aux_target=opq_subvector requires OPQ/PQ centroids at {pq_centroids_path}"
            )
        else:
            self.sid_code_centroids = self._build_sid_code_centroids(item2tokens)
        return item2tokens

    def _create_reverse_mapping(self):
        """创建反向映射用于推理"""
        tokens2item = {}
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            tokens2item[tuple(tokens)] = item_id
        return tokens2item

    def _save_mappings(self):
        """保存映射文件"""
        # 构建路径 - 修复：使用类名和category
        dataset_name = self.dataset.__class__.__name__  # 使用类名，如"AmazonReviews2014"
        
        # 如果有category属性，加入路径中
        if hasattr(self.dataset, 'category') and self.dataset.category:
            cache_dir = os.path.join(
                self.dataset.cache_dir, 'processed'
            )
        else:
            cache_dir = os.path.join(
                'data', dataset_name, 'processed'
            )
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # 🚀 文件名包含：模型+PCA+量化器标签(+种子/iters)+n_digit，完全避免不同配置冲突
        model_basename = os.path.basename(self.config["sent_emb_model"]) 
        quant_tag = self.index_factory
        if self.sid_quantizer == 'rq_kmeans':
            quant_tag += f'_seed{self.config.get("rq_kmeans_seed",1234)}_it{self.config.get("rq_kmeans_niters",20)}'
        elif self.sid_quantizer == 'none':
            quant_tag += f'_seed{self.config.get("sid_random_seed",12345)}'
        map_tag = f'{model_basename}_pca{self.config["sent_emb_pca"]}_{quant_tag}_{self.n_digit}d'
        
        # 保存正排索引：item_id → SID-tokens
        item_id2tokens = np.zeros((self.dataset.n_items, self.n_digit), dtype=np.int64)
        for item, tokens in self.item2tokens.items():
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = np.array(tokens)
        
        np.save(os.path.join(cache_dir, f'item_id2tokens_{map_tag}.npy'), item_id2tokens)
        
        # 保存倒排索引：SID-tokens → item_id
        with open(os.path.join(cache_dir, f'tokens2item_{map_tag}.pkl'), 'wb') as f:
            pickle.dump(self.tokens2item, f)
        
        self.log(f'[TOKENIZER] Saved mappings with tag: {map_tag} to {cache_dir}')
        self.log(f'[TOKENIZER] Files: item_id2tokens_{map_tag}.npy, tokens2item_{map_tag}.pkl')

    def encode_history(self, item_seq, max_len=None):
        """编码用户历史序列"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        
        history_sid = []
        for item in item_seq:
            if item in self.item2tokens:
                # 将带offset的token ID转换为codebook ID (0..K-1)
                tokens = list(self.item2tokens[item])  # 带offset的token ID
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
            else:
                # 未知商品用PAD填充（使用-1作为PAD的哨兵值，避免与codebook_id=0混淆）
                history_sid.append([-1] * self.n_digit)
        
        # 填充到固定长度
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
        
        return history_sid  # 返回list，让datasets.map自动张量化
    
    def encode_history_with_mask(self, item_seq, max_len=None):
        """编码用户历史序列，同时返回padding mask"""
        if max_len is None:
            max_len = self.config.get('max_history_len', 50)
        if len(item_seq) > max_len:
            item_seq = item_seq[-max_len:]
        
        history_sid = []
        history_mask = []  # True=有效位置，False=PAD位置
        
        for item in item_seq:
            if item in self.item2tokens:
                # 将带offset的token ID转换为codebook ID (0..K-1)
                tokens = list(self.item2tokens[item])  # 带offset的token ID
                codebook_ids = []
                for digit, token_id in enumerate(tokens):
                    codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                    codebook_ids.append(codebook_id)
                history_sid.append(codebook_ids)
                history_mask.append(True)  # 有效位置
            else:
                # 未知商品用PAD填充（使用-1作为PAD的哨兵值，避免与codebook_id=0混淆）
                history_sid.append([-1] * self.n_digit)
                history_mask.append(False)  # PAD位置
        
        # 填充到固定长度
        while len(history_sid) < max_len:
            history_sid.append([-1] * self.n_digit)
            history_mask.append(False)  # PAD位置
        
        return history_sid, history_mask  # 返回list，让datasets.map自动张量化

    def encode_decoder_input(self, target_item):
        """编码decoder输入 - 与RPG_ED保持一致"""
        if target_item in self.item2tokens:
            tokens = list(self.item2tokens[target_item])  # 4个token ID（带offset）
            
            # 将token ID转换为codebook ID
            codebook_tokens = []
            for digit, token_id in enumerate(tokens):
                codebook_id = token_id - (self.sid_offset + digit * self.codebook_size)
                codebook_tokens.append(codebook_id)
            
            # decoder输入和标签都是codebook IDs
            decoder_input = codebook_tokens  # [cb0, cb1, cb2, cb3]
            decoder_labels = codebook_tokens  # [cb0, cb1, cb2, cb3]
        else:
            # 未知商品
            decoder_input = [self.pad_token] * self.n_digit  # 长度n_digit
            decoder_labels = [self.pad_token] * self.n_digit  # 长度n_digit
        
        return decoder_input, decoder_labels

    def encode_target_sem_emb(self, target_item):
        """Return the fixed pre-quantization semantic vector for the target item."""
        if self.pre_opq_sem_embs is None:
            return None
        item_id = self.item2id.get(target_item, 0)
        if item_id <= 0 or item_id > self.pre_opq_sem_embs.shape[0]:
            return [0.0] * self.pre_opq_sem_emb_dim
        return self.pre_opq_sem_embs[item_id - 1].astype(np.float32, copy=False).tolist()

    def encode_target_opq_subvectors(self, target_item):
        """Return OPQ-rotated, pre-quantization target subvectors."""
        if self.item_id2opq_subvectors is None:
            return None
        item_id = self.item2id.get(target_item, 0)
        if item_id <= 0 or item_id >= self.item_id2opq_subvectors.shape[0]:
            return np.zeros(
                (self.n_digit, self.item_id2opq_subvectors.shape[-1]),
                dtype=np.float32
            ).tolist()
        return self.item_id2opq_subvectors[item_id].astype(np.float32, copy=False).tolist()

    def _build_sid_code_centroids(self, item2tokens):
        """Build fixed pre-OPQ semantic centroids for each SID level/code."""
        if self.pre_opq_sem_embs is None:
            return None

        dim = int(self.pre_opq_sem_embs.shape[1])
        centroids = np.zeros((self.n_digit, self.codebook_size, dim), dtype=np.float32)
        counts = np.zeros((self.n_digit, self.codebook_size), dtype=np.int64)

        for item, tokens in item2tokens.items():
            item_id = self.item2id.get(item, 0)
            if item_id <= 0 or item_id > self.pre_opq_sem_embs.shape[0]:
                continue
            sem = self.pre_opq_sem_embs[item_id - 1]
            for digit, token_id in enumerate(tokens):
                code_id = int(token_id) - (self.sid_offset + digit * self.codebook_size)
                if 0 <= code_id < self.codebook_size:
                    centroids[digit, code_id] += sem
                    counts[digit, code_id] += 1

        global_mean = self.pre_opq_sem_embs.mean(axis=0).astype(np.float32, copy=False)
        missing = 0
        for digit in range(self.n_digit):
            for code_id in range(self.codebook_size):
                count = counts[digit, code_id]
                if count > 0:
                    centroids[digit, code_id] /= float(count)
                else:
                    centroids[digit, code_id] = global_mean
                    missing += 1

        self.log(
            f'[TOKENIZER] Built SID code semantic centroids: shape={centroids.shape}, '
            f'missing_codes={missing}'
        )
        return centroids

    def decode_tokens_to_item(self, tokens):
        """将token序列解码为商品ID"""
        if len(tokens) != self.n_digit:
            return None
        
        token_tuple = tuple(tokens)
        return self.tokens2item.get(token_tuple)

    def codebooks_to_item_id(self, cb_ids):
        """
        将codebook ID序列转换为item_id，检查合法性
        
        Args:
            cb_ids: List[int] 长度 n_digit, 原始 codebook ID (0-255)
            
        Returns:
            item_id(int) 或 None（如果非法）
        """
        if len(cb_ids) != self.n_digit:
            return None
        
        # 将codebook ID转换为token ID
        token_ids = [
            cb_ids[d] + self.sid_offset + d * self.codebook_size
            for d in range(self.n_digit)
        ]
        
        # 查找对应的item_id
        return self.tokens2item.get(tuple(token_ids))

    def tokenize_function(self, example: dict, split: str) -> dict:
        """tokenize函数 - 修复数据泄露问题"""
        item_seq = example['item_seq']  # Python list
        target_item = item_seq[-1]  # 原始字符串
        
        # 修复：所有split都应该用item_seq[:-1]作为历史，避免数据泄露
        history_sid, history_mask = self.encode_history_with_mask(item_seq[:-1])
        
        if split == 'train':
            # 训练时编码decoder输入
            decoder_input, decoder_labels = self.encode_decoder_input(target_item)
            out = {
                'history_sid': history_sid,  # 直接list
                'history_mask': history_mask,  # 直接list
                'decoder_input_ids': decoder_input,  # 直接list
                'decoder_labels': decoder_labels  # 直接list
            }
            target_sem_emb = self.encode_target_sem_emb(target_item)
            if target_sem_emb is not None:
                out['target_sem_emb'] = target_sem_emb
            target_opq_subvectors = self.encode_target_opq_subvectors(target_item)
            if target_opq_subvectors is not None:
                out['target_opq_subvectors'] = target_opq_subvectors
            return out
        else:
            # 验证/测试时生成真标签
            _, decoder_labels = self.encode_decoder_input(target_item)
            return {
                'history_sid': history_sid,  # 直接list
                'history_mask': history_mask,  # 直接list
                'labels': decoder_labels  # 新增：真标签序列
            }

    def tokenize(self, datasets: dict) -> dict:
        """tokenize数据集"""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=False,  # 关闭批处理，避免数据结构混乱
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets 

    # ====== 新增：SID→items 映射与工具 ======
    def _sid_tokens_to_cb_tuple(self, tokens):
        """
        把带 offset 的 SID-token（长度 n_digit）转换为 codebook 索引 tuple（每位 0..K-1）。
        例如 [sid_offset + 0*K + a, sid_offset + 1*K + b, ...] → (a,b,...)
        """
        assert len(tokens) == self.n_digit
        cb = []
        for d, tok in enumerate(tokens):
            cb.append(int(tok) - (self.sid_offset + d * self.codebook_size))
        return tuple(cb)

    def _build_cb2items_map(self):
        """
        基于 self.item2tokens 构建 SID 组合 → items 的倒排表。
        注意：允许一对多（发生“冲突”时都放进去）。
        """
        from collections import defaultdict
        cb2items = defaultdict(list)
        for item, toks in self.item2tokens.items():
            cb = self._sid_tokens_to_cb_tuple(toks)
            cb2items[cb].append(item)
        return cb2items

    @property
    def cb2items(self):
        """
        惰性缓存：首次访问时构建 SID→items 映射并缓存到 _cb2items
        """
        if not hasattr(self, "_cb2items") or self._cb2items is None:
            self._cb2items = self._build_cb2items_map()
        return self._cb2items

    def cb_tuple_to_item_ids(self, cb):
        """
        给定一个 codebook tuple，返回对应的 item_id 列表（按构建顺序稳定）。
        """
        items = self.cb2items.get(cb, [])
        out = []
        for it in items:
            iid = self.item2id.get(it, 0)
            if iid > 0:
                out.append(iid)
        return out
