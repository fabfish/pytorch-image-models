class _battention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4)

        masker = Masker(prune_ratio = 0.8)
        q_mask = masker(q)
        k_mask = masker(k)
        v_mask = masker(v)
        shape_q, mask_q, sparse_q = sparsify(q, mask_q, with_batch_size=False)
        shape_k, mask_k, sparse_k = sparsify(k, mask_k, with_batch_size=False)
        shape_v, mask_v, sparse_v = sparsify(v, mask_v, with_batch_size=False)

        ctx.save_for_backward(shape_q, mask_q, sparse_q, shape_k, mask_k, sparse_k, shape_v, mask_v, sparse_v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        # q, k, v, o, L = ctx.saved_tensors
        shape_q, mask_q, sparse_q, shape_k, mask_k, sparse_k, shape_v, mask_v, sparse_v, o, L = ctx.saved_tensors
        sparse_q = sparse_q.float()
        sparse_k = sparse_k.float()
        sparse_v = sparse_v.float()
        # sparse_qkv = sparse_qkv.float()
        # qkv = unsparsify(shape_qkv, mask_qkv, sparse_qkv, with_batch_size=False)    
        q = unsparsify(shape_q, mask_q, sparse_q, with_batch_size=False)
        k = unsparsify(shape_k, mask_k, sparse_k, with_batch_size=False)
        v = unsparsify(shape_v, mask_v, sparse_v, with_batch_size=False)
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q, k, v, ctx.sm_scale,
            o, do,
            dq, dk, dv,
            L, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            CAUSAL=ctx.causal,
            num_stages=1,
        )
        return dq, dk, dv, None, None


battention = _battention.apply