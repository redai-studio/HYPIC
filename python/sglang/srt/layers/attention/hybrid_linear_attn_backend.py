import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb


def derotate_kv(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Reverse rotary: rotate K from current position back to position 0.

    R(-pos) = R(pos)^{-1}, achieved by negating sin.
    """
    return apply_rotary_emb(k, cos, -sin, is_neox_style)


def rerotate_kv(
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Apply rotary to position-0-stored K to reach a new position."""
    return apply_rotary_emb(k, cos, sin, is_neox_style)

from sglang.srt.layers.attention.mamba.causal_conv1d_triton import PAD_SLOT_ID
from sglang.srt.layers.attention.mamba.mamba import MambaMixer2
from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
    fused_mamba_state_scatter_with_mask,
    track_mamba_states_if_needed,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.pic.policy import PICCompose
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cpu


def _derotate_with_partial(x, cos, sin, rotary_dim, head_dim, is_neox, derotate_fn):
    """Apply derotate_kv to x, handling partial rotary (rotary_dim < head_dim)."""
    if rotary_dim < head_dim:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = derotate_fn(x_rot, cos.to(x_rot.dtype), sin.to(x_rot.dtype), is_neox)
        return torch.cat([x_rot, x_pass], dim=-1)
    return derotate_fn(x, cos.to(x.dtype), sin.to(x.dtype), is_neox)


def _rerotate_with_partial(x, cos, sin, rotary_dim, head_dim, is_neox, rerotate_fn):
    """Apply rerotate_kv to x, handling partial rotary (rotary_dim < head_dim)."""
    if rotary_dim < head_dim:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_rot = rerotate_fn(x_rot, cos.to(x_rot.dtype), sin.to(x_rot.dtype), is_neox)
        return torch.cat([x_rot, x_pass], dim=-1)
    return rerotate_fn(x, cos.to(x.dtype), sin.to(x.dtype), is_neox)

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

logger = logging.getLogger(__name__)


def _moti2_rank() -> int:
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _dump_moti2_context_kv(
    layer,
    q,
    kv_buf,
    kv_indices,
    positions,
    qo_indptr=None,
    kv_indptr=None,
    kv_positions=None,
):
    dump_dir = os.environ.get("PIC_MOTI2_DUMP_DIR")
    if not dump_dir:
        return
    if q.shape[0] < int(os.environ.get("PIC_MOTI2_MIN_TOKENS", "0")):
        return
    os.makedirs(dump_dir, exist_ok=True)
    rank = _moti2_rank()
    idx = kv_indices.detach().to(torch.long)
    k_buf, v_buf = kv_buf
    torch.save(
        {
            "layer": int(layer.layer_id),
            "rank": int(rank),
            "q": q.detach().view(q.shape[0], layer.tp_q_head_num, layer.head_dim).to(torch.float32).cpu(),
            "k": k_buf.index_select(0, idx).detach().to(torch.float32).cpu(),
            "v": v_buf.index_select(0, idx).detach().to(torch.float32).cpu(),
            "positions": positions.detach().cpu() if positions is not None else None,
            "kv_indices": idx.cpu(),
            "qo_indptr": qo_indptr.detach().cpu() if qo_indptr is not None else None,
            "kv_indptr": kv_indptr.detach().cpu() if kv_indptr is not None else None,
            "kv_positions": kv_positions.detach().cpu()
            if kv_positions is not None
            else None,
        },
        os.path.join(dump_dir, f"qwen_attn_plan_L{int(layer.layer_id):03d}_r{rank}.pt"),
    )


class MambaAttnBackendBase(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = PAD_SLOT_ID
        self.device = model_runner.device
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.is_draft_worker = model_runner.is_draft_worker
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []
        self.retrieve_next_token_list = []
        self.retrieve_next_sibling_list = []
        self.retrieve_parent_token_list = []
        self.cached_cuda_graph_decode_query_start_loc: torch.Tensor = None
        self.cached_cuda_graph_verify_query_start_loc: torch.Tensor = None
        self.conv_states_shape: tuple[int, int] = None

    def _execute_deferred_mamba_cow_and_clear(self, forward_batch: ForwardBatch):
        """Run deferred clear/COW ops on the forward stream to avoid races."""
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
            or self.is_draft_worker
        ):
            return
        if (
            forward_batch.mamba_clear_indices is not None
            and len(forward_batch.mamba_clear_indices) > 0
        ):
            self.req_to_token_pool.mamba_pool.clear_slots(
                forward_batch.mamba_clear_indices
            )
        if (
            forward_batch.mamba_cow_src_indices is not None
            and len(forward_batch.mamba_cow_src_indices) > 0
        ):
            ckpt_pool = getattr(self.req_to_token_pool, "mamba_ckpt_pool", None)
            if ckpt_pool is not None:
                # int8 checkpoints: dequantize the cached state (src = int8 ckpt slot)
                # into the request's active bf16 slot (dst).
                ckpt_pool.load_to_active(
                    self.req_to_token_pool.mamba_pool,
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
            else:
                self.req_to_token_pool.mamba_pool.copy_from(
                    forward_batch.mamba_cow_src_indices,
                    forward_batch.mamba_cow_dst_indices,
                )
        forward_batch.mamba_clear_indices = None
        forward_batch.mamba_cow_src_indices = None
        forward_batch.mamba_cow_dst_indices = None

    def _forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size

        retrieve_next_token = None
        retrieve_next_sibling = None
        retrieve_parent_token = None
        track_conv_indices = None
        track_ssm_h_src = None
        track_ssm_h_dst = None
        track_ssm_final_src = None
        track_ssm_final_dst = None

        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        _real_bs = getattr(forward_batch, "_original_batch_size", None)
        if _real_bs is not None and _real_bs < mamba_cache_indices.shape[0]:
            mamba_cache_indices = mamba_cache_indices.clone()
            mamba_cache_indices[_real_bs:] = -1

        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
            if forward_batch.forward_mode.is_draft_extend_v2():
                # HybridLinearAttnBackend.init_forward_metadata calls all sub-backends
                # unconditionally, but DRAFT_EXTEND_V2 only runs full-attn layers in
                # the draft model, so mamba metadata can be skipped.
                query_start_loc = None
            elif forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )

                if self.topk > 1:
                    retrieve_next_token = forward_batch.spec_info.retrieve_next_token
                    retrieve_next_sibling = (
                        forward_batch.spec_info.retrieve_next_sibling
                    )
                    # retrieve_next_token is None during dummy run so skip tensor creation
                    if retrieve_next_token is not None:
                        retrieve_parent_token = torch.empty_like(retrieve_next_token)
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
                if (
                    forward_batch.mamba_track_mask is not None
                    and forward_batch.mamba_track_mask.any()
                ):
                    track_conv_indices = self._init_track_conv_indices(
                        query_start_loc, forward_batch
                    )

                    (
                        track_ssm_h_src,
                        track_ssm_h_dst,
                        track_ssm_final_src,
                        track_ssm_final_dst,
                    ) = self._init_track_ssm_indices(mamba_cache_indices, forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")

        has_mamba_track_mask = bool(
            forward_batch.mamba_track_mask is not None
            and forward_batch.mamba_track_mask.any()
        )

        return ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
            track_conv_indices=track_conv_indices,
            track_ssm_h_src=track_ssm_h_src,
            track_ssm_h_dst=track_ssm_h_dst,
            track_ssm_final_src=track_ssm_final_src,
            track_ssm_final_dst=track_ssm_final_dst,
            has_mamba_track_mask=has_mamba_track_mask,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # seq_lens_cpu is unused by _replay_metadata for the non-target-verify
        # case but kept in the contract for compatibility.
        self.forward_metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        self.forward_metadata = self._forward_metadata(forward_batch)

    def _init_track_conv_indices(
        self, query_start_loc: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute indices for extracting conv states from the input sequence during extend.

        In Mamba models, the conv layer maintains a sliding window of recent inputs.
        After processing a prefill chunk, we need to save the last `conv_state_len` tokens
        of the processed region for prefix caching.

        The key insight is that FLA (Flash Linear Attention) and Mamba2 processes sequences in chunks
        of the chunk size (FLA_CHUNK_SIZE=64 for FLA, mamba_chunk_size for Mamba2).
        We only track the conv state up to the last complete chunk boundary (aligned_len).

        start_indices is the starting token index of the conv state to track in this extend batch.
        indices include all pos to track in this extend batch, conv_state_len for each req that
        needs to be tracked (i.e. mamba_track_mask is True)

        Returns:
            indices: Tensor of shape [num_tracked_requests, conv_state_len] containing
                     flattened positions into the packed input tensor.
        """
        conv_state_len = self.conv_states_shape[-1]

        # Calculate the end position of the last aligned chunk
        lens_to_track = (
            forward_batch.mamba_track_seqlens - forward_batch.extend_prefix_lens
        )
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        aligned_len = (lens_to_track // mamba_cache_chunk_size) * mamba_cache_chunk_size
        start_indices = query_start_loc[:-1] + aligned_len - conv_state_len
        start_indices = start_indices[forward_batch.mamba_track_mask]

        # Create indices: [batch_size, conv_state_len]
        indices = start_indices.unsqueeze(-1) + torch.arange(
            conv_state_len,
            device=self.device,
            dtype=start_indices.dtype,
        )

        return indices.clamp(0, query_start_loc[-1] - 1)

    def _init_track_ssm_indices(
        self, mamba_cache_indices: torch.Tensor, forward_batch: ForwardBatch
    ):
        """
        Compute source and destination indices for tracking SSM states for prefix caching.

        After processing a prefill, we need to save the SSM recurrent state for prefix caching.
        The kernel outputs intermediate hidden states `h` at each chunk boundary,
        plus a `last_recurrent_state` at the end of the chunked prefill size.

        The chunk size varies by model type:
        - FLA models: FLA_CHUNK_SIZE (64)
        - Mamba2 models: mamba_chunk_size (256)

        The challenge is that sequences may or may not end on a chunk boundary:
          - Aligned case (len % chunk_size == 0): The to-cache state is stored in
            the last_recurrent_state.
          - Unaligned case (len % chunk_size != 0): The last_recurrent_state includes the
            unaligned position, but we only want state up to the last chunk boundary.
            We must extract from the intermediate `h` tensor at the appropriate chunk index.

        We compute the src and dst indices for all requests that need to be cached
        (i.e. mamba_track_mask is True) based on the rule above.

        For example (assuming chunk_size=64):
        1. If chunked prefill length is < chunk_size, then only final state has value.
           In this case we cache `final` state.
        2. If chunked prefill length == chunk_size, then only final state has value.
           In this case we cache pos chunk_size, from `final` state.
        3. If chunked prefill length > chunk_size and < 2 * chunk_size, then both h and
           final state have value. We cache pos chunk_size from `h` state.
        4. If chunked prefill length == 2 * chunk_size, then both h and final state have
           value. We cache pos 2 * chunk_size from `final` state. Note `h` doesn't include
           the final position.

        Returns:
            track_ssm_h_src: Source indices into the packed `h` tensor (for unaligned seqs)
            track_ssm_h_dst: Destination cache slot indices (for unaligned seqs)
            track_ssm_final_src: Source indices into last_recurrent_state buffer (for aligned seqs)
            track_ssm_final_dst: Destination cache slot indices (for aligned seqs)
        """
        mamba_cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        # Move to CPU to avoid kernel launches for masking operations
        mamba_track_mask = forward_batch.mamba_track_mask.cpu()
        extend_seq_lens = forward_batch.extend_seq_lens.cpu()
        mamba_track_indices = forward_batch.mamba_track_indices.cpu()
        mamba_cache_indices = mamba_cache_indices.cpu()
        mamba_track_seqlens = forward_batch.mamba_track_seqlens.cpu()
        prefix_lens = forward_batch.extend_prefix_lens.cpu()

        # Calculate the number of hidden states per request
        if isinstance(self, Mamba2AttnBackend):
            num_h_states = extend_seq_lens // mamba_cache_chunk_size
        else:
            num_h_states = (extend_seq_lens - 1) // mamba_cache_chunk_size + 1

        # Calculate the starting offset for each sequence in the packed batch
        track_ssm_src_offset = torch.zeros_like(num_h_states)
        track_ssm_src_offset[1:] = torch.cumsum(num_h_states[:-1], dim=0)

        # Filter variables by track mask
        lens_to_track = mamba_track_seqlens - prefix_lens
        lens_masked = lens_to_track[mamba_track_mask]
        offset_masked = track_ssm_src_offset[mamba_track_mask]
        dst_masked = mamba_track_indices[mamba_track_mask]

        # Determine if the sequence ends at a chunk boundary
        is_aligned = (lens_masked % mamba_cache_chunk_size) == 0

        # Case 1: Aligned. Use last_recurrent_state from ssm_states.
        track_ssm_final_src = mamba_cache_indices[mamba_track_mask][is_aligned]
        track_ssm_final_dst = dst_masked[is_aligned]

        # Case 2: Unaligned. Use intermediate state from h.
        # TODO: if support mamba_cache_chunk_size % page size != 0, then need to modify this
        not_aligned = ~is_aligned
        track_ssm_h_src = offset_masked[not_aligned] + (
            lens_masked[not_aligned] // mamba_cache_chunk_size
        )
        track_ssm_h_dst = dst_masked[not_aligned]

        # Move back to GPU
        return (
            track_ssm_h_src.to(self.device, non_blocking=True),
            track_ssm_h_dst.to(self.device, non_blocking=True),
            track_ssm_final_src.to(self.device, non_blocking=True),
            track_ssm_final_dst.to(self.device, non_blocking=True),
        )

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        self.forward_metadata = self._capture_metadata(
            bs, req_pool_indices, forward_mode, spec_info
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        draft_token_num = max_num_tokens // max_bs
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.zeros((i + 2,), dtype=torch.int32, device=self.device)
            )
            self.retrieve_next_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_next_sibling_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
            self.retrieve_parent_token_list.append(
                torch.zeros(
                    (i + 1, draft_token_num), dtype=torch.int32, device=self.device
                )
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.cached_cuda_graph_verify_query_start_loc = torch.arange(
            0,
            max_bs * draft_token_num + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=self.device,
        )

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        assert (
            max_num_tokens % max_bs == 0
        ), f"max_num_tokens={max_num_tokens} must be divisible by max_bs={max_bs}"
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full(
                    (i + 1,), self.pad_slot_id, dtype=torch.int32, device=self.device
                )
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device=self.device)
            )
        self.cached_cuda_graph_decode_query_start_loc = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )

    def _capture_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
            )
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            # They are None during cuda graph capture so skip the copy_...
            # self.retrieve_next_token_list[bs - 1].copy_(spec_info.retrieve_next_token)
            # self.retrieve_next_sibling_list[bs - 1].copy_(spec_info.retrieve_next_sibling)
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def _replay_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        num_padding: Optional[int] = None,
    ):
        if num_padding is None:
            if seq_lens_cpu is None:
                num_padding = 0
            else:
                num_padding = torch.count_nonzero(
                    seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
                )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_decode_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    bs - num_padding
                )
        elif forward_mode.is_target_verify():
            if num_padding == 0:
                self.query_start_loc_list[bs - 1].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs + 1]
                )
            else:
                self.query_start_loc_list[bs - 1][: bs - num_padding].copy_(
                    self.cached_cuda_graph_verify_query_start_loc[: bs - num_padding]
                )
                self.query_start_loc_list[bs - 1][bs - num_padding :].fill_(
                    (bs - num_padding) * spec_info.draft_token_num
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        # If topk > 1, we need to use retrieve_next_token and retrieve_next_sibling to handle the eagle tree custom attention mask
        if forward_mode.is_target_verify() and self.topk > 1:
            if (
                spec_info is not None
                and getattr(spec_info, "retrieve_next_token", None) is not None
            ):
                bs_without_pad = spec_info.retrieve_next_token.shape[0]
                self.retrieve_next_token_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_token
                )
                self.retrieve_next_sibling_list[bs - 1][:bs_without_pad].copy_(
                    spec_info.retrieve_next_sibling
                )
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
                retrieve_next_token=self.retrieve_next_token_list[bs - 1],
                retrieve_next_sibling=self.retrieve_next_sibling_list[bs - 1],
                retrieve_parent_token=self.retrieve_parent_token_list[bs - 1],
            )
        else:
            return ForwardMetadata(
                query_start_loc=self.query_start_loc_list[bs - 1],
                mamba_cache_indices=self.state_indices_list[bs - 1],
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def get_cpu_graph_seq_len_fill_value(self):
        return 1

    def _track_mamba_state_decode(
        self,
        forward_batch: ForwardBatch,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ):
        """
        Track and copy Mamba conv/SSM states during decode for prefix caching.

        During decode, each token update modifies conv_states and ssm_states in-place
        at positions indexed by cache_indices (the working slots). For prefix caching,
        we need to copy these updated states to persistent cache slots (mamba_track_indices)
        so they can be prefix cached.

        This delegates to `track_mamba_states_if_needed`, which performs:
            conv_states[mamba_track_indices[i]] = conv_states[cache_indices[i]]
            ssm_states[mamba_track_indices[i]] = ssm_states[cache_indices[i]]
        for all requests where mamba_track_mask[i] is True.
        """
        if forward_batch.mamba_track_mask is not None:
            track_mamba_states_if_needed(
                conv_states,
                ssm_states,
                cache_indices,
                forward_batch.mamba_track_mask,
                forward_batch.mamba_track_indices,
                forward_batch.batch_size,
            )

    def _track_mamba_state_extend(
        self,
        forward_batch: ForwardBatch,
        h: torch.Tensor,
        ssm_states: torch.Tensor,
        forward_metadata: ForwardMetadata,
    ):
        """
        Track and copy SSM states during extend for prefix caching.

        After the chunked prefill kernel runs, we need to save the SSM recurrent
        state at the last chunk boundary so it can be reused for prefix caching.
        The source of the state depends on whether the sequence length is aligned
        to the chunk size. See `_init_track_ssm_indices` for more details on how
        the source and destination indices are computed.

        Note: Conv state tracking for extend is handled separately via gather operations
        using indices computed by `_init_track_conv_indices`.
        """
        if forward_metadata.has_mamba_track_mask:
            h = h.squeeze(0)

            if forward_metadata.track_ssm_h_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_h_dst] = h[
                    forward_metadata.track_ssm_h_src
                ].to(ssm_states.dtype, copy=False)
            if forward_metadata.track_ssm_final_src.numel() > 0:
                ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                    forward_metadata.track_ssm_final_src
                ]


class Mamba2AttnBackend(MambaAttnBackendBase):
    """Attention backend wrapper for Mamba2Mixer kernels."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        config = model_runner.mamba2_config
        assert config is not None
        self.mamba_chunk_size = config.mamba_chunk_size
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )

        if model_runner.server_args.enable_mamba_extra_buffer():
            assert (
                self.conv_states_shape[-1] < self.mamba_chunk_size
            ), f"{self.conv_states_shape[-1]=} should be less than {self.mamba_chunk_size}"
            assert (
                model_runner.server_args.mamba_track_interval >= self.mamba_chunk_size
            ), f"mamba_track_interval ({model_runner.server_args.mamba_track_interval}) must be >= mamba_chunk_size ({self.mamba_chunk_size})"

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        metadata = self._replay_metadata(
            forward_batch.batch_size,
            forward_batch.req_pool_indices,
            forward_batch.forward_mode,
            forward_batch.spec_info,
            forward_batch.seq_lens_cpu if not in_capture else None,
            num_padding=(
                0 if in_capture else getattr(forward_batch, "num_padding", None)
            ),
        )
        spec_info = forward_batch.spec_info
        draft_token_num = spec_info.draft_token_num if spec_info is not None else 1
        self.forward_metadata = Mamba2Metadata.prepare_decode(
            metadata,
            forward_batch.seq_lens,
            is_target_verify=forward_batch.forward_mode.is_target_verify(),
            draft_token_num=draft_token_num,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self._execute_deferred_mamba_cow_and_clear(forward_batch)
        metadata = self._forward_metadata(forward_batch)
        self.forward_metadata = Mamba2Metadata.prepare_mixed(
            metadata,
            self.mamba_chunk_size,
            forward_batch,
        )

    def forward(
        self,
        mixer: MambaMixer2,
        hidden_states: torch.Tensor,
        output: Optional[torch.Tensor],
        layer_id: int,
        forward_batch: ForwardBatch,
        mup_vector: Optional[torch.Tensor] = None,
        use_triton_causal_conv: bool = False,
    ):
        assert isinstance(self.forward_metadata, Mamba2Metadata)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
        mixer_out, intermediate_states = mixer.forward(
            hidden_states=hidden_states,
            output=output,
            layer_cache=layer_cache,
            metadata=self.forward_metadata,
            forward_batch=forward_batch,
            mup_vector=mup_vector,
            use_triton_causal_conv=use_triton_causal_conv,
        )

        if forward_batch.mamba_track_mask is not None:
            if (
                intermediate_states is not None
                and forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                self._track_mamba_state_extend(
                    forward_batch,
                    intermediate_states,
                    layer_cache.temporal,
                    self.forward_metadata,
                )

            if self.forward_metadata.num_decodes > 0:
                num_decodes = self.forward_metadata.num_decodes
                track_mamba_states_if_needed(
                    layer_cache.conv[0],
                    layer_cache.temporal,
                    self.forward_metadata.mamba_cache_indices[-num_decodes:],
                    forward_batch.mamba_track_mask[-num_decodes:],
                    forward_batch.mamba_track_indices[-num_decodes:],
                    num_decodes,
                )

        return mixer_out

    def forward_decode(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )

    def forward_extend(self, *args, **kwargs):
        raise NotImplementedError(
            "Mamba2AttnBackend's forward is called directly instead of through HybridLinearAttnBackend, as it supports mixed prefill and decode"
        )


class HybridLinearAttnBackend(AttentionBackend):
    """Manages a full and linear attention backend"""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: MambaAttnBackendBase,
        full_attn_layers: list[int],
        model_runner: Optional[ModelRunner] = None,
    ):
        self.full_attn_layers = full_attn_layers
        self.full_attn_backend = full_attn_backend
        self.linear_attn_backend = linear_attn_backend
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]
        # Dispatcher aliases the full-attn backend's pool refs.
        self.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        self.req_to_token_pool = full_attn_backend.req_to_token_pool
        self.max_context_len = getattr(full_attn_backend, "max_context_len", None)
        self.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or linear_attn_backend.needs_cpu_seq_lens
        )
        self._pic_prefill_wrapper = None
        self._pic_workspace = None
        # T1 fix: keep wrapper instance alive across batches; this flag gates
        # whether the current batch has any PIC miss segments needing the plan.
        self._pic_has_plan = False
        # transition_rope: dedicated wrappers for Phase A (local isolated) and
        # Phase C (global cross-seg). See _init_pic_rope_plans.
        self._pic_rope_local_wrapper = None
        self._pic_rope_global_wrapper = None
        self._pic_rope_has_local_plan = False
        self._pic_rope_has_global_plan = False
        self._pic_rope_local_plan_ready = False
        self._pic_rope_global_plan_ready = False
        self._pic_rope_phase_c_cache = None
        self._pic_rope_phase_b_cache = None
        self._pic_rope_cross_real_cache = None
        # For transition_rope: rotary embedding cache (lazily initialized)
        self._pic_rope_cos_sin_cache = None
        self._pic_rope_is_neox = True
        self._pic_rope_rotary_dim = 0
        self._model_runner = model_runner

    def _pic_is_bailing_linear_model(self) -> bool:
        if self._model_runner is None:
            return False
        model_config = getattr(self._model_runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        if hf_config is None:
            return False
        archs = set(getattr(hf_config, "architectures", []) or [])
        return bool(
            archs
            & {
                "BailingMoeLinearV2ForCausalLM",
                "BailingMoELinearForCausalLM",
            }
        ) or getattr(hf_config, "model_type", None) == "bailing_moe_linear"

    def _is_full_attn(
        self, layer: Optional[RadixAttention], layer_id: Optional[int] = None
    ) -> bool:
        if layer is not None:
            layer_id = layer.layer_id
        assert layer_id is not None, "either layer or layer_id must be provided"
        return layer_id in self.full_attn_layers

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_out_graph(
                forward_batch, in_capture=in_capture
            )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            # DRAFT_EXTEND_V2 only runs full-attn layers in the draft model,
            # so skip linear/mamba backend metadata which requires query_start_loc.
            self.full_attn_backend.init_forward_metadata(forward_batch)
            return
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)
        if getattr(forward_batch, "pic_mode", None) is not None:
            # transition_rope uses dedicated dual-wrapper plans (Phase A local
            # isolated, Phase C global cross-seg). All other modes share the
            # original per-segment plan.
            if forward_batch.pic_policy.rope:
                if self._pic_rope_cos_sin_cache is None:
                    self._init_pic_rope_cache()
                self._init_pic_rope_plans(forward_batch)
                if self._pic_rope_use_cross_segment_full_attn(forward_batch):
                    self._init_pic_prefill_plan(forward_batch)
                else:
                    # The shared single wrapper is not used in rope mode.
                    self._pic_has_plan = False
            else:
                self._init_pic_prefill_plan(forward_batch)
            # init_pic_metadata deferred to first GDN layer call
            self._pic_gdn_metadata_ready = False
            linear_needs_rope = bool(
                getattr(self.linear_attn_backend, "pic_state_needs_rope_rerotate", False)
            )
            if linear_needs_rope and self._pic_rope_cos_sin_cache is None:
                self._init_pic_rope_cache()

    def init_mha_chunk_metadata(
        self, forward_batch: ForwardBatch, disable_flashinfer_ragged: bool = False
    ):
        # Hybrid MLA models (Ring/Ling, Kimi-Linear) resolve this via
        # get_attn_backend(), which returns this wrapper; delegate to the
        # full-attn backend so its chunked/one-shot prefill metadata is planned.
        init = getattr(self.full_attn_backend, "init_mha_chunk_metadata", None)
        if init is not None:
            init(forward_batch, disable_flashinfer_ragged)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cpu_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cpu_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cuda_graph_seq_len_fill_value()

    def get_cpu_graph_seq_len_fill_value(self):
        return self.full_attn_backend.get_cpu_graph_seq_len_fill_value()

    def forward_decode(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For GDN linear attention
        b: Optional[torch.Tensor] = None,  # For GDN linear attention
        **kwargs,
    ):
        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_decode(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def forward_extend(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # PIC dispatch: route through mode-specific paths; leave
        # the non-PIC path untouched so existing requests keep working.
        if getattr(forward_batch, "pic_mode", None) is not None:
            if forward_batch.pic_policy.compose is PICCompose.ADDITION:
                return self._forward_extend_pic_addition(
                    layer,
                    forward_batch,
                    save_kv_cache,
                    q=q,
                    k=k,
                    v=v,
                    mixed_qkv=mixed_qkv,
                    a=a,
                    b=b,
                    **kwargs,
                )
            # TRANSITION family: transition / transition_rope / transition_rope_recompute.
            return self._forward_extend_pic_transition_family(
                layer,
                forward_batch,
                save_kv_cache,
                q=q,
                k=k,
                v=v,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                **kwargs,
            )

        if self._is_full_attn(layer, kwargs.get("layer_id")):
            return self.full_attn_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear attention backend
        return self.linear_attn_backend.forward_extend(
            q=q,
            k=k,
            v=v,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            **kwargs,
        )

    def _pic_linear_rope_kwargs(self, forward_batch: ForwardBatch) -> dict:
        """Bundle PIC mode + RoPE cache for linear backends that need it."""
        if not getattr(self.linear_attn_backend, "pic_state_needs_rope_rerotate", False):
            return {"pic_mode": getattr(forward_batch, "pic_mode", None)}
        if self._pic_rope_cos_sin_cache is None:
            self._init_pic_rope_cache()
        return {
            "pic_mode": getattr(forward_batch, "pic_mode", None),
            "pic_rope_cos_sin_cache": self._pic_rope_cos_sin_cache,
            "pic_rope_is_neox": self._pic_rope_is_neox,
            "pic_rope_rotary_dim": self._pic_rope_rotary_dim,
        }

    def _pic_rope_use_cross_segment_full_attn(self, forward_batch: ForwardBatch) -> bool:
        """Use Ring/Bailing's less lossy full-attn transition_rope path.

        The clean Qwen path computes local miss segments with isolated
        per-segment attention. Ring's hybrid stack is much more sensitive to
        those local full-attn hidden differences, so for Bailing/Ring we keep
        the RoPE K correction but run the normal cross-segment PIC full-attn
        plan over private real-position slots.
        """
        if getattr(forward_batch, "pic_mode", None) not in (
            "transition_rope", "transition_rope_recompute",
        ):
            return False
        return self._pic_is_bailing_linear_model()

    def _forward_extend_pic_addition(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """PIC lapic_addition dispatch for hybrid (full-attn + linear-attn) layers.

        Full-attn layers:
          Dispatches to ``_forward_extend_pic_full_attn`` which uses a dedicated
          FlashInfer BatchPrefillWithPagedKVCacheWrapper with per-segment
          qo_indptr/kv_indptr. Each miss segment is an independent "request"
          in the varlen batch; Q attends only to K/V slots at [0, seg_end)
          ensuring correct causal masking. Plan is built once per batch in
          ``_init_pic_prefill_plan`` and reused across all full-attn layers.
          Note: no RoPE correction — cached KV retains original positions.

        Linear-attn (GDN) layers:
          Dispatches to ``linear_attn_backend.forward_extend_pic_addition``
          which treats each miss segment as an independent sequence in a
          varlen batch (via seg_cu_seqlens). Conv1d + GDN kernel run once
          over all segments with initial_state=zeros (per-segment S=0).
          After the kernel, a fused Triton gather+sum kernel accumulates
          hit-segment states (from MambaPool) + miss-segment states into
          S_total per request, and persists non-last miss segment states
          to their pre-allocated cache slots.
        """
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self._forward_extend_pic_full_attn(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear-attn (GDN) layers: per-segment kernel calls + state summation.
        if not self._pic_gdn_metadata_ready:
            self.linear_attn_backend.init_pic_metadata(forward_batch)
            self._pic_gdn_metadata_ready = True
        merged_kwargs = {**kwargs, **self._pic_linear_rope_kwargs(forward_batch)}
        return self.linear_attn_backend.forward_extend_pic_addition(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            q=q,
            k=k,
            v=v,
            **merged_kwargs,
        )

    def _forward_extend_pic_transition_family(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        q: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        mixed_qkv: Optional[torch.Tensor] = None,
        a: Optional[torch.Tensor] = None,
        b: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """PIC transition-family dispatch (transition / transition_rope / recompute).

        Full-attn layers:
          rope=False (transition): no RoPE correction; per-segment FlashInfer plan
            (same as addition mode).
          rope=True (transition_rope[_recompute]): re-rotate hit-seg K to real
            positions before attention (cross-segment variant when applicable).

        Linear-attn (GDN / Lightning) layers:
          Position-independent, so identical across the family; dispatches to the
          shared linear_attn_backend.forward_extend_pic_transition, which itself
          self-routes to the recompute path when policy.recompute.
        """
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            if not forward_batch.pic_policy.rope:
                return self._forward_extend_pic_full_attn(
                    q, k, v, layer, forward_batch, save_kv_cache, **kwargs
                )
            if self._pic_rope_use_cross_segment_full_attn(forward_batch):
                return self._forward_extend_pic_full_attn_rope_cross_segment(
                    q, k, v, layer, forward_batch, save_kv_cache, **kwargs
                )
            return self._forward_extend_pic_full_attn_rope(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        # Linear-attn (GDN / Lightning) layers: shared transition path.
        if not self._pic_gdn_metadata_ready:
            self.linear_attn_backend.init_pic_metadata(forward_batch)
            self._pic_gdn_metadata_ready = True
        merged_kwargs = {**kwargs, **self._pic_linear_rope_kwargs(forward_batch)}
        return self.linear_attn_backend.forward_extend_pic_transition(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            q=q,
            k=k,
            v=v,
            **merged_kwargs,
        )

    def _forward_extend_pic_full_attn_rope_cross_segment(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Ring/Bailing transition_rope full-attn path.

        Prepare private slots at real positions for all visible segments, while
        still writing local-miss public slots in segment-anchored form so future
        requests can reuse them. Then run the normal cross-segment PIC prefill
        attention plan.
        """

        rope_meta = getattr(forward_batch, "pic_rope_meta", None)
        if rope_meta is None:
            return q.new_zeros(0, layer.tp_q_head_num * layer.head_dim)

        device = q.device
        head_dim = layer.head_dim
        rotary_dim = self._pic_rope_rotary_dim
        is_neox = self._pic_rope_is_neox
        cos_sin_cache = self._pic_rope_cos_sin_cache
        token_to_kv_pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)
        q3 = q.view(-1, layer.tp_q_head_num, head_dim)
        k3 = k.view(-1, layer.tp_k_head_num, head_dim)
        v3 = v.view(-1, layer.tp_k_head_num, head_dim)

        cache = self._pic_rope_cross_real_cache
        if cache is None:
            cache = self._build_pic_rope_cross_real_cache(rope_meta, device)
            self._pic_rope_cross_real_cache = cache
        assert cache["expected_q_rows"] == q3.shape[0], (
            f"PIC rope cross-segment mapped {cache['expected_q_rows']} q rows, "
            f"expected {q3.shape[0]}"
        )

        miss_q = cache["miss_q"]
        if miss_q.numel() > 0:
            k_miss = k3.index_select(0, miss_q)
            v_miss = v3.index_select(0, miss_q)
            k_buffer[cache["miss_pub"]] = k_miss
            v_buffer[cache["miss_pub"]] = v_miss
            k_buffer[cache["miss_priv"]] = k_miss
            v_buffer[cache["miss_priv"]] = v_miss

        global_q = cache["global_q"]
        if global_q.numel() > 0:
            k_global = k3.index_select(0, global_q)
            v_global = v3.index_select(0, global_q)
            k_buffer[cache["global_slots"]] = k_global
            v_buffer[cache["global_slots"]] = v_global

        hit_priv = cache["hit_priv"]
        if hit_priv.numel() > 0:
            hit_entry = cache["hit_entry"]
            k_buffer[hit_priv] = k_buffer.index_select(0, hit_entry)
            v_buffer[hit_priv] = v_buffer.index_select(0, hit_entry)

        wrapper = self._pic_prefill_wrapper
        if not self._pic_has_plan or wrapper is None:
            return q.new_zeros(0, layer.tp_q_head_num * layer.head_dim)

        if not self._pic_plan_ready:
            wrapper.begin_forward(
                qo_indptr=self._pic_qo_indptr,
                paged_kv_indptr=self._pic_kv_indptr,
                paged_kv_indices=self._pic_kv_indices,
                paged_kv_last_page_len=self._pic_kv_last_page_len,
                num_qo_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim_qk=layer.head_dim,
                page_size=1,
                causal=True,
                q_data_type=q.dtype,
            )
            self._pic_plan_ready = True

        kv_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        _dump_moti2_context_kv(
            layer,
            q,
            kv_buf,
            self._pic_kv_indices,
            forward_batch.positions,
            self._pic_qo_indptr,
            self._pic_kv_indptr,
            self._pic_kv_positions,
        )
        o = wrapper.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_buf,
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )
        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _build_pic_rope_cross_real_cache(
        self,
        rope_meta: List[Dict[str, object]],
        device: torch.device,
    ) -> Dict[str, object]:
        miss_q: List[int] = []
        global_q: List[int] = []
        miss_pub_chunks: List[torch.Tensor] = []
        miss_priv_chunks: List[torch.Tensor] = []
        global_slot_chunks: List[torch.Tensor] = []
        hit_priv_chunks: List[torch.Tensor] = []
        hit_entry_chunks: List[torch.Tensor] = []
        cursor = 0

        for meta in rope_meta:
            entries = []
            for start, end, _priv, _pub in meta["local_miss"]:
                entries.extend(range(int(start), int(end)))
            global_info = meta["global"]
            if global_info is not None:
                gs, ge, _gp = global_info
                entries.extend(range(int(gs), int(ge)))
            # Include seam tokens (hit-segment sink positions) in the
            # q-row mapping so the cross-segment attention plan covers them.
            seam_info = meta.get("seam") if isinstance(meta, dict) else None
            if seam_info is not None:
                hit_seam = seam_info.get("hit_seam", {})
                for (_s, _e), sink_pos in hit_seam.items():
                    entries.extend(int(p) for p in sink_pos)
            entries.sort()
            abs_to_q = {pos: cursor + i for i, pos in enumerate(entries)}
            cursor += len(entries)

            for start, end, priv, pub in meta["local_miss"]:
                miss_q.extend(abs_to_q[int(start) + ofs] for ofs in range(end - start))
                miss_pub_chunks.append(pub.to(device).to(torch.int64))
                miss_priv_chunks.append(priv.to(device).to(torch.int64))

            if global_info is not None:
                gs, ge, gp = global_info
                global_q.extend(abs_to_q[int(gs) + ofs] for ofs in range(ge - gs))
                global_slot_chunks.append(gp.to(device).to(torch.int64))

            for _start, _end, priv, entry_pub in meta["local_hit"]:
                hit_priv_chunks.append(priv.to(device).to(torch.int64))
                hit_entry_chunks.append(entry_pub.to(device).to(torch.int64))

        empty = torch.empty(0, dtype=torch.int64, device=device)
        return {
            "expected_q_rows": cursor,
            "miss_q": torch.tensor(miss_q, dtype=torch.int64, device=device)
            if miss_q
            else empty,
            "miss_pub": torch.cat(miss_pub_chunks, dim=0)
            if miss_pub_chunks
            else empty,
            "miss_priv": torch.cat(miss_priv_chunks, dim=0)
            if miss_priv_chunks
            else empty,
            "global_q": torch.tensor(global_q, dtype=torch.int64, device=device)
            if global_q
            else empty,
            "global_slots": torch.cat(global_slot_chunks, dim=0)
            if global_slot_chunks
            else empty,
            "hit_priv": torch.cat(hit_priv_chunks, dim=0)
            if hit_priv_chunks
            else empty,
            "hit_entry": torch.cat(hit_entry_chunks, dim=0)
            if hit_entry_chunks
            else empty,
        }

    def _forward_extend_pic_full_attn_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """PIC transition_rope full-attn: per-segment isolated semantics.

        Per-layer, three phases (see qianyou/2026-06-07-pic-transition-rope-
        isolated-design.md):

          Phase A (LOCAL ISOLATED, pos=0):
            For every local miss + local hit token, compute attention with
            Q and K placed at pos=0 and K visibility restricted to the same
            segment (causal within segment, no cross-segment). The pos=0 K
            for local miss is derived by derotating the model-provided
            real-pos K. The pos=0 K for local hit comes from entry.full_kv_slots
            (already pos=0 by construction).
            Local miss Q is derotated to pos=0; the pos=0 K (derotated) is
            written to BOTH public and private (private is overwritten in
            Phase B). The phase output for local miss tokens is the final
            attention output for those tokens.
            Local hit tokens have no Q (they are cache-only), but their K
            is loaded into the FlashInfer plan as visible context for the
            (no) local hit Q — this phase produces no output for hit tokens.

          Phase B (REROTATE → PRIVATE @ real pos):
            Local miss: read public K (pos=0), rerotate to real pos, write
              to private K. Copy V from public to private.
            Local hit:  read entry public K (pos=0), rerotate to real pos,
              write to private K. Copy V from entry public to private.

          Phase C (GLOBAL CROSS-SEG, real pos):
            Global tokens use real-pos Q and real-pos K (model already
            rope-applied them; save them to global's private slot).
            Build a single-segment plan whose K visibility is concat of all
            private slots in real-pos order: [seg_1_priv, seg_2_priv, ...,
            global_priv]. Causal mask handles within-global ordering.
            Output is written to the global tokens.

        Final output (concatenated over all miss tokens, matching input_ids
        order): local-miss-token outputs from Phase A, global-token outputs
        from Phase C.
        """
        rope_meta = getattr(forward_batch, "pic_rope_meta", None)
        if rope_meta is None:
            return q.new_zeros(0, layer.tp_q_head_num * layer.head_dim)

        device = q.device
        head_dim = layer.head_dim
        num_q_heads = layer.tp_q_head_num
        num_kv_heads = layer.tp_k_head_num
        token_to_kv_pool = self.token_to_kv_pool
        layer_id = layer.layer_id

        q3 = q.view(-1, num_q_heads, head_dim)
        k3 = k.view(-1, num_kv_heads, head_dim) if k is not None else None
        v3 = v.view(-1, num_kv_heads, head_dim) if v is not None else None
        T = q3.shape[0]
        o_buf = q3.new_zeros(T, num_q_heads, head_dim)

        # input_ids order is real-pos sorted (per req). For each token at q_index k,
        # its absolute position is the k-th element of the per-req sorted abs_pos list.
        # Here we build:
        #   - miss_token_role[k] = (req_idx, kind)  kind: 0 local miss, 1 global, 2 hit seam
        #   - local_q_indices: q-indices of local-miss tokens in rope_meta seg-walk order
        #   - global_q_indices: q-indices of global tokens in rope_meta order
        #   - hit_seam_q_info: list of (q_index, req_idx, (s,e), local_ofs)
        is_recompute = forward_batch.pic_policy.recompute
        miss_token_role: List[Tuple[int, int]] = []
        miss_token_seg_idx_in_req_local: List[int] = []
        # Per-req: abs_pos -> q_index
        per_req_abs_to_q: List[Dict[int, int]] = []
        # Reverse map: q_index -> (req_idx, abs_pos). Reset per forward (this
        # function is the canonical entry point for transition_rope full-attn
        # — Phase A/B/C all read from a cache built off rope_meta, never from
        # cross-batch state). Used by Phase C cache builder to bucket and sort
        # Q rows by abs_pos.
        self._pic_q_to_req_abs_dict: Dict[int, Tuple[int, int]] = {}
        cursor = 0
        for req_idx, meta in enumerate(rope_meta):
            local_miss = meta["local_miss"]
            global_info = meta["global"]
            seam_info = meta.get("seam") if isinstance(meta, dict) else None
            # Collect this req's abs_pos list with kind labels.
            entries: List[Tuple[int, int, object]] = []  # (abs_pos, kind, payload)
            for li, (start, end, _priv, _pub) in enumerate(local_miss):
                for ofs in range(end - start):
                    entries.append((start + ofs, 0, li))
            if global_info is not None:
                (gs, ge, _gp) = global_info
                for ofs in range(ge - gs):
                    entries.append((gs + ofs, 1, -1))
            if is_recompute and seam_info is not None:
                hit_seam = seam_info.get("hit_seam", {})
                for (s, e), sink_pos in hit_seam.items():
                    for j, ap in enumerate(sink_pos):
                        entries.append((ap, 2, ((s, e), j)))
            entries.sort(key=lambda x: x[0])
            abs_to_q: Dict[int, int] = {}
            for (ap, kind, _payload) in entries:
                abs_to_q[ap] = cursor
                self._pic_q_to_req_abs_dict[cursor] = (req_idx, ap)
                miss_token_role.append((req_idx, kind))
                miss_token_seg_idx_in_req_local.append(-1)  # backfill below
                cursor += 1
            per_req_abs_to_q.append(abs_to_q)

        assert len(miss_token_role) == T, (
            f"miss-token count mismatch: rope_meta says {len(miss_token_role)}, q has {T}"
        )

        # Build seg-walk-ordered q-index lists by looking up abs_pos -> q.
        local_q_indices: List[int] = []
        global_q_indices: List[int] = []
        hit_seam_q_info: List[Tuple[int, int, Tuple[int, int], int]] = []
        for req_idx, meta in enumerate(rope_meta):
            abs_to_q = per_req_abs_to_q[req_idx]
            local_miss = meta["local_miss"]
            global_info = meta["global"]
            for li, (start, end, _priv, _pub) in enumerate(local_miss):
                for ofs in range(end - start):
                    qi = abs_to_q[start + ofs]
                    local_q_indices.append(qi)
                    miss_token_seg_idx_in_req_local[qi] = li
            if global_info is not None:
                (gs, ge, _gp) = global_info
                for ofs in range(ge - gs):
                    global_q_indices.append(abs_to_q[gs + ofs])
            seam_info = meta.get("seam") if isinstance(meta, dict) else None
            if is_recompute and seam_info is not None:
                hit_seam = seam_info.get("hit_seam", {})
                for (s, e), sink_pos in hit_seam.items():
                    for j, ap in enumerate(sink_pos):
                        hit_seam_q_info.append((abs_to_q[ap], req_idx, (s, e), j))
        hit_seam_q_indices = [info[0] for info in hit_seam_q_info]

        # Phase A — kv-only writer
        # (miss-seg query attention is deferred to Phase C with cross-seg KV).
        if local_q_indices or any(meta["local_hit"] for meta in rope_meta):
            self._pic_rope_phase_a_kv_only(
                q3, k3, v3, layer, forward_batch, rope_meta,
                local_q_indices,
            )

        # Phase B — REROTATE private to real pos
        self._pic_rope_phase_b(layer, forward_batch, rope_meta)

        # Phase C — cross-seg @ real pos. Miss-seg query (local_q_indices)
        # joins global Q to attend over ALL prior segs' private slots.
        phase_c_local_q = local_q_indices
        if global_q_indices or hit_seam_q_indices or phase_c_local_q:
            self._pic_rope_phase_c(
                q3, k3, v3, layer, forward_batch, rope_meta,
                global_q_indices, o_buf,
                hit_seam_q_info=hit_seam_q_info if is_recompute else None,
                local_miss_q_indices=phase_c_local_q,
            )

        return o_buf.view(-1, num_q_heads * head_dim)

    def _pic_rope_phase_a_kv_only(
        self,
        q3: torch.Tensor, k3: torch.Tensor, v3: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        rope_meta: List[Dict[str, object]],
        local_q_indices: List[int],
    ):
        """Phase A degraded: stage miss-seg K/V into PUBLIC (pos=0 K, derotated)
        and PRIVATE (real-pos K, as model produced). NO attention here.
        miss-seg query attention is computed in Phase C alongside global Q.

        Invariant after this function:
          k_buffer[pub_slot]  = derotated K (pos=0, for caching as future hit-seg)
          k_buffer[priv_slot] = real-pos K (for Phase C cross-seg attn)
          v_buffer[pub_slot]  = v_buffer[priv_slot] = v_seg
        """

        device = q3.device
        head_dim = layer.head_dim
        rotary_dim = self._pic_rope_rotary_dim
        is_neox = self._pic_rope_is_neox
        cos_sin_cache = self._pic_rope_cos_sin_cache
        token_to_kv_pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

        lq_cursor = 0
        for req_idx, meta in enumerate(rope_meta):
            local_miss = meta["local_miss"]
            for (start, end, priv, pub) in local_miss:
                seg_len = end - start
                seg_q_idx = torch.tensor(
                    local_q_indices[lq_cursor:lq_cursor + seg_len],
                    dtype=torch.int64, device=device,
                )
                lq_cursor += seg_len

                seg_positions = torch.full(
                    (seg_len,), int(start), dtype=torch.int64, device=device,
                )
                cos_sin = cos_sin_cache.index_select(0, seg_positions)
                cos, sin = cos_sin.chunk(2, dim=-1)

                k_seg = k3.index_select(0, seg_q_idx)   # model already rope-applied → real pos
                v_seg = v3.index_select(0, seg_q_idx)
                k_zero = _derotate_with_partial(
                    k_seg, cos, sin, rotary_dim, head_dim, is_neox, derotate_kv,
                )

                pub_dev = pub.to(device)
                priv_dev = priv.to(device)
                # PUBLIC: pos=0 K (cacheable, future hit-seg fills from here)
                k_buffer[pub_dev] = k_zero
                v_buffer[pub_dev] = v_seg
                # PRIVATE: real-pos K (model output as-is, for Phase C cross-seg attn)
                k_buffer[priv_dev] = k_seg
                v_buffer[priv_dev] = v_seg

    # ------------------------------------------------------------------
    # Phase B: rerotate K @ pos=0 → real pos and write to private
    # ------------------------------------------------------------------
    def _pic_rope_phase_b(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        rope_meta: List[Dict[str, object]],
    ):

        device = forward_batch.input_ids.device
        head_dim = layer.head_dim
        rotary_dim = self._pic_rope_rotary_dim
        is_neox = self._pic_rope_is_neox
        token_to_kv_pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

        # Build batch-level cache once: concatenated src/dst indices and the
        # per-row cos/sin tensors. Phase B rerotates K from "segment-anchored"
        # form (k_raw[j]·R(j)) to real pos by multiplying by R(start) — the
        # same R(start) for every token in the segment, so we expand `start`
        # over seg_len rows. cos/sin are layer-invariant.
        cache = getattr(self, "_pic_rope_phase_b_cache", None)
        if cache is None:
            cache = self._build_pic_rope_phase_b_cache(rope_meta, device)
            self._pic_rope_phase_b_cache = cache
        if cache is None:
            return

        src = cache["src"]
        dst = cache["dst"]
        cos = cache["cos"]
        sin = cache["sin"]

        k_zero = k_buffer.index_select(0, src)
        v_zero = v_buffer.index_select(0, src)
        k_real = _rerotate_with_partial(
            k_zero, cos, sin, rotary_dim, head_dim, is_neox, rerotate_kv,
        )
        k_buffer[dst] = k_real
        v_buffer[dst] = v_zero

    def _build_pic_rope_phase_b_cache(
        self,
        rope_meta: List[Dict[str, object]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build batch-level Phase B cache: concat (src, dst, cos, sin) over
        all hit segments. cos/sin use the segment's `start` repeated
        seg_len times. Returns None if no segments need rerotation.

        Miss segs are skipped: Phase A (`_pic_rope_phase_a_kv_only`) already
        writes real-pos (model-output) K directly to priv. Re-running
        rerotate(derotate(K_real)) here would round-trip through bf16 and
        silently drift the priv slots that Phase C reads.
        Hit segs are always covered: their entry pub is the canonical pos=0
        cache K which has to be rerotated to real pos for cross-seg attn.
        """
        include_miss = False
        cos_sin_cache = self._pic_rope_cos_sin_cache
        src_chunks: List[torch.Tensor] = []
        dst_chunks: List[torch.Tensor] = []
        pos_list: List[int] = []  # one position per row, len = total_rows

        for meta in rope_meta:
            if include_miss:
                for (start, end, priv, pub) in meta["local_miss"]:
                    if priv.numel() == 0:
                        continue
                    seg_len = int(end - start)
                    src_chunks.append(pub.to(device).to(torch.int64))
                    dst_chunks.append(priv.to(device).to(torch.int64))
                    pos_list.extend([int(start)] * seg_len)
            for (start, end, priv, entry_pub) in meta["local_hit"]:
                if priv.numel() == 0:
                    continue
                seg_len = int(end - start)
                src_chunks.append(entry_pub.to(device).to(torch.int64))
                dst_chunks.append(priv.to(device).to(torch.int64))
                pos_list.extend([int(start)] * seg_len)

        if not src_chunks:
            return None

        positions = torch.tensor(pos_list, dtype=torch.int64, device=device)
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        return {
            "src": torch.cat(src_chunks, dim=0),
            "dst": torch.cat(dst_chunks, dim=0),
            "cos": cos,
            "sin": sin,
        }

    # ------------------------------------------------------------------
    # Phase C: global cross-seg attention at real pos
    # ------------------------------------------------------------------
    def _pic_rope_phase_c(
        self,
        q3: torch.Tensor,
        k3: torch.Tensor,
        v3: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        rope_meta: List[Dict[str, object]],
        global_q_indices: List[int],
        o_buf: torch.Tensor,
        hit_seam_q_info: Optional[List[Tuple[int, int, Tuple[int, int], int]]] = None,
        local_miss_q_indices: Optional[List[int]] = None,
    ):
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        device = q3.device
        head_dim = layer.head_dim
        token_to_kv_pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
        v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

        # Layer-invariant cache: built once per batch, reused by all 40 layers.
        # Captures Q packing order, K/V destination slots, KV visibility indptrs,
        # scatter index. Layer-variant work = K/V buffer scatter + wrapper.forward.
        cache = getattr(self, "_pic_rope_phase_c_cache", None)
        if cache is None:
            cache = self._build_pic_rope_phase_c_cache(
                rope_meta, global_q_indices, hit_seam_q_info, device,
                local_miss_q_indices=local_miss_q_indices,
            )
            self._pic_rope_phase_c_cache = cache
        if cache is None:
            return  # nothing to do this batch

        # Layer-variant work: scatter K/V into pool buffers at private slots.
        if cache["kv_dst_slots"].numel() > 0:
            k_buffer[cache["kv_dst_slots"]] = k3.index_select(0, cache["kv_src_qis"])
            v_buffer[cache["kv_dst_slots"]] = v3.index_select(0, cache["kv_src_qis"])

        q_packed = q3.index_select(0, cache["q_packed_src"])

        if self._pic_workspace is None:
            self._pic_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )
        if self._pic_rope_global_wrapper is None:
            self._pic_rope_global_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=self._pic_workspace,
            )

        wrapper = self._pic_rope_global_wrapper
        if not self._pic_rope_global_plan_ready:
            wrapper.begin_forward(
                qo_indptr=cache["qo_indptr"],
                paged_kv_indptr=cache["kv_indptr"],
                paged_kv_indices=cache["kv_indices"],
                paged_kv_last_page_len=cache["kv_last_page_len"],
                num_qo_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim_qk=head_dim,
                page_size=1,
                causal=True,
                q_data_type=q_packed.dtype,
            )
            self._pic_rope_global_plan_ready = True

        kv_buf = token_to_kv_pool.get_kv_buffer(layer_id)
        q_pos = forward_batch.positions
        if q_pos is not None and q_pos.shape[0] == q3.shape[0]:
            q_pos = q_pos.index_select(0, cache["q_packed_src"])
        _dump_moti2_context_kv(
            layer,
            q_packed,
            kv_buf,
            cache["kv_indices"],
            q_pos,
            cache["qo_indptr"],
            cache["kv_indptr"],
            cache.get("kv_positions"),
        )
        o_packed = wrapper.forward(
            q_packed,
            kv_buf,
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        o_buf.index_copy_(0, cache["scatter_idx"], o_packed)

    def _build_pic_rope_phase_c_cache(
        self,
        rope_meta: List[Dict[str, object]],
        global_q_indices: List[int],
        hit_seam_q_info: Optional[List[Tuple[int, int, Tuple[int, int], int]]],
        device: torch.device,
        local_miss_q_indices: Optional[List[int]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build layer-invariant Phase C metadata once per batch.

        Returns dict with q_packed_src, kv_dst_slots, kv_src_qis, kv_indices,
        qo_indptr, kv_indptr, kv_last_page_len, scatter_idx. None if no work.

        local_miss_q_indices: miss-seg query attention is deferred from
        Phase A to here. Each miss-seg Q row joins as its own
        FlashInfer prefill segment (qo_len=1) with a causal KV slice = concat
        of priv slots of all earlier-end segs of its req + the in-seg prefix
        of its own seg (up to abs_pos). Priv slots are populated by Phase A
        (miss → real-pos K from model) and Phase B (hit → rerotated real-pos
        K from entry pub).
        """
        local_miss_q_indices = local_miss_q_indices or []
        # Group hit-seam Q entries by req with abs_pos derived from rope_meta.
        hit_seam_by_req: Dict[int, List[Tuple[int, int, Tuple[int, int], int]]] = {}
        if hit_seam_q_info:
            seam_idx = 0
            for ri, meta in enumerate(rope_meta):
                seam_info = meta.get("seam") if isinstance(meta, dict) else None
                if seam_info is None:
                    continue
                hit_seam = seam_info.get("hit_seam", {})
                for (s, e), sink_pos in hit_seam.items():
                    for j, ap in enumerate(sink_pos):
                        qi = hit_seam_q_info[seam_idx][0]
                        hit_seam_by_req.setdefault(ri, []).append(
                            (qi, ap, (s, e), j)
                        )
                        seam_idx += 1

        q_packed_src: List[int] = []
        seg_q_offsets: List[int] = [0]
        kv_indices_chunks: List[torch.Tensor] = []
        kv_pos_chunks: List[torch.Tensor] = []
        seg_kv_offsets: List[int] = [0]
        scatter_q_idx_all: List[int] = []
        # K/V destination slots and source q indices (paired, scatter together).
        kv_dst_slots_chunks: List[torch.Tensor] = []
        kv_src_qis_list: List[int] = []
        gq_cursor = 0

        if not self._pic_is_bailing_linear_model():
            # Bucket miss-seg Q by req for fast lookup.
            miss_q_by_req: Dict[int, List[Tuple[int, int]]] = {}
            for qi in local_miss_q_indices:
                if qi not in self._pic_q_to_req_abs_dict:
                    raise RuntimeError(
                        f"PIC Phase C: miss Q {qi} not in q_to_req_abs map"
                    )
                ri, ap = self._pic_q_to_req_abs_dict[qi]
                miss_q_by_req.setdefault(ri, []).append((ap, qi))

            for req_idx, meta in enumerate(rope_meta):
                global_info = meta["global"]
                seam_for_req = hit_seam_by_req.get(req_idx, [])
                miss_q_for_req = miss_q_by_req.get(req_idx, [])
                if (
                    global_info is None
                    and not seam_for_req
                    and not miss_q_for_req
                ):
                    continue

                # Order hit+miss segs by abs start. Append global as the last
                # seg in priv_tensors so per-Q seg lookup is unified.
                all_segs_in_order: List[Tuple[int, int, torch.Tensor]] = []
                for (s, e, priv, _pub) in meta["local_miss"]:
                    all_segs_in_order.append((s, e, priv))
                for (s, e, priv, _entry) in meta["local_hit"]:
                    all_segs_in_order.append((s, e, priv))
                all_segs_in_order.sort(key=lambda t: t[0])

                # ---- K/V scatter scheduling (unchanged semantics) ----------
                gq_slice: List[int] = []
                gs = ge = None
                gp = None
                if global_info is not None:
                    (gs, ge, gp) = global_info
                    gseg_len = ge - gs
                    gq_slice = global_q_indices[gq_cursor:gq_cursor + gseg_len]
                    gq_cursor += gseg_len
                    kv_dst_slots_chunks.append(gp.to(device).to(torch.int64))
                    kv_src_qis_list.extend(gq_slice)

                hit_priv = {
                    (hs, he): p for (hs, he, p, _entry) in meta["local_hit"]
                }
                for (qi, ap, (s, e), _lofs) in seam_for_req:
                    priv = hit_priv.get((s, e))
                    if priv is None:
                        continue
                    slot_ofs = int(ap) - int(s)
                    kv_dst_slots_chunks.append(
                        priv.to(device).to(torch.int64)[slot_ofs:slot_ofs + 1]
                    )
                    kv_src_qis_list.append(qi)

                # ---- Batched per-run Q planning (1 plan / 1 kernel) -------
                # Group Qs into maximal abs_pos-consecutive runs within the
                # same containing seg. Each run → one FlashInfer seg with
                # qo_len = run_len, kv_len = all_priors_priv + own_priv[:last-s+1].
                # FlashInfer causal=True suffix mask in this layout gives
                # Q[i] visibility = priors + own_priv[:first-s+i+1] — exactly
                # the causal prefix for the i-th run token. Correct equivalent
                # of per-row planning, but seg count drops from O(tokens) to
                # O(runs) ≈ O(segs/req).

                # Per-req cached priv int32 tensors + bounds + cum lens.
                priv_tensors: List[torch.Tensor] = []
                priv_bounds: List[Tuple[int, int]] = []
                for (s, e, priv) in all_segs_in_order:
                    priv_tensors.append(priv.to(device).to(torch.int32))
                    priv_bounds.append((s, e))
                if global_info is not None:
                    priv_tensors.append(gp.to(device).to(torch.int32))
                    priv_bounds.append((gs, ge))
                cum_lens: List[int] = [0]
                for t in priv_tensors:
                    cum_lens.append(cum_lens[-1] + int(t.numel()))

                # Collect all Qs of this req, sorted by abs_pos.
                all_qs: List[Tuple[int, int]] = []
                if global_info is not None:
                    for j, qi in enumerate(gq_slice):
                        all_qs.append((gs + j, qi))
                for (qi, ap, _rng, _lofs) in seam_for_req:
                    all_qs.append((ap, qi))
                for (ap, qi) in miss_q_for_req:
                    all_qs.append((ap, qi))
                all_qs.sort(key=lambda x: x[0])

                # Walk Qs, emit one seg per consecutive run within same seg.
                i = 0
                while i < len(all_qs):
                    ap0, qi0 = all_qs[i]
                    k = None
                    for kk, (s, e) in enumerate(priv_bounds):
                        if s <= ap0 < e:
                            k = kk
                            break
                    if k is None:
                        raise RuntimeError(
                            f"PIC Phase C: Q {qi0} abs_pos={ap0} not in any seg"
                        )
                    s_k, e_k = priv_bounds[k]
                    run_qis = [qi0]
                    expected_ap = ap0 + 1
                    j = i + 1
                    while j < len(all_qs):
                        apj, qij = all_qs[j]
                        if apj != expected_ap or apj >= e_k:
                            break
                        run_qis.append(qij)
                        expected_ap += 1
                        j += 1
                    ap_last = expected_ap - 1
                    own_prefix_len = ap_last - s_k + 1
                    if cum_lens[k] > 0:
                        kv_indices_chunks.append(
                            torch.cat(priv_tensors[:k])
                            if k > 1
                            else priv_tensors[0]
                        )
                        kv_pos_chunks.append(
                            torch.cat(
                                [
                                    torch.arange(
                                        s, e, dtype=torch.int64, device=device
                                    )
                                    for s, e in priv_bounds[:k]
                                ]
                            )
                        )
                    kv_indices_chunks.append(priv_tensors[k][:own_prefix_len])
                    kv_pos_chunks.append(
                        torch.arange(
                            s_k,
                            s_k + own_prefix_len,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
                    total_kv = cum_lens[k] + own_prefix_len
                    q_packed_src.extend(run_qis)
                    scatter_q_idx_all.extend(run_qis)
                    seg_q_offsets.append(seg_q_offsets[-1] + len(run_qis))
                    seg_kv_offsets.append(seg_kv_offsets[-1] + total_kv)
                    i = j

            if not q_packed_src:
                return None

            num_segs = len(seg_q_offsets) - 1
            return {
                "q_packed_src": torch.tensor(
                    q_packed_src, dtype=torch.int64, device=device
                ),
                "kv_dst_slots": torch.cat(kv_dst_slots_chunks, dim=0)
                if kv_dst_slots_chunks
                else torch.empty(0, dtype=torch.int64, device=device),
                "kv_src_qis": torch.tensor(
                    kv_src_qis_list, dtype=torch.int64, device=device
                )
                if kv_src_qis_list
                else torch.empty(0, dtype=torch.int64, device=device),
                "kv_indices": torch.cat(kv_indices_chunks, dim=0),
                "kv_positions": torch.cat(kv_pos_chunks, dim=0),
                "qo_indptr": torch.tensor(
                    seg_q_offsets, dtype=torch.int32, device=device
                ),
                "kv_indptr": torch.tensor(
                    seg_kv_offsets, dtype=torch.int32, device=device
                ),
                "kv_last_page_len": torch.ones(
                    num_segs, dtype=torch.int32, device=device
                ),
                "scatter_idx": torch.tensor(
                    scatter_q_idx_all, dtype=torch.int64, device=device
                ),
            }

        def add_phase_c_segment(
            q_abs_and_idx: List[Tuple[int, int]],
            visible_slots: List[torch.Tensor],
        ) -> None:
            if not q_abs_and_idx:
                return
            q_abs_and_idx.sort(key=lambda x: x[0])
            seg_qis = [qi for (_ap, qi) in q_abs_and_idx]
            q_packed_src.extend(seg_qis)
            scatter_q_idx_all.extend(seg_qis)
            total_kv = 0
            for slots in visible_slots:
                if slots.numel() == 0:
                    continue
                kv_indices_chunks.append(slots.to(device).to(torch.int32))
                total_kv += int(slots.numel())
            if total_kv <= 0:
                raise RuntimeError("PIC Phase C segment has Q but no visible KV")
            seg_q_offsets.append(seg_q_offsets[-1] + len(seg_qis))
            seg_kv_offsets.append(seg_kv_offsets[-1] + total_kv)

        for req_idx, meta in enumerate(rope_meta):
            global_info = meta["global"]
            seam_for_req = hit_seam_by_req.get(req_idx, [])
            if global_info is None and not seam_for_req:
                continue

            all_segs_in_order: List[Tuple[int, int, torch.Tensor]] = []
            for (s, e, priv, _pub) in meta["local_miss"]:
                all_segs_in_order.append((s, e, priv))
            for (s, e, priv, _entry) in meta["local_hit"]:
                all_segs_in_order.append((s, e, priv))
            all_segs_in_order.sort(key=lambda t: t[0])

            # Build hit-seg priv lookup and seam q-index lookup. Each seam
            # window is planned as a separate FlashInfer segment whose Q tokens
            # are the suffix of its visible KV. A single mixed segment
            # (C1 seam + C3 seam + global) violates FlashInfer causal suffix
            # semantics and lets early seam Q attend to future KV.
            hit_priv = {(hs, he): p for (hs, he, p, _entry) in meta["local_hit"]}
            seam_q_by_abs = {ap: qi for (qi, ap, _rng, _lofs) in seam_for_req}
            seam_info = meta.get("seam") if isinstance(meta, dict) else None
            if seam_info is not None:
                hit_seam = seam_info.get("hit_seam", {}) or {}
                for (s, e), sink_pos in hit_seam.items():
                    priv = hit_priv.get((s, e))
                    if priv is None:
                        continue
                    priv_dev = priv.to(device)
                    if not sink_pos:
                        continue
                    q_entries = [
                        (int(ap), seam_q_by_abs[int(ap)])
                        for ap in sink_pos
                        if int(ap) in seam_q_by_abs
                    ]
                    if not q_entries:
                        continue
                    # Write recomputed K/V for these Q tokens to their real
                    # private slots. Slots are aligned to absolute position
                    # within the hit segment.
                    slot_offsets = torch.tensor(
                        [int(ap) - int(s) for ap, _qi in q_entries],
                        dtype=torch.long,
                        device=device,
                    )
                    kv_dst_slots_chunks.append(
                        priv_dev.to(torch.int64).index_select(0, slot_offsets)
                    )
                    kv_src_qis_list.extend([qi for _ap, qi in q_entries])

                    last_ap = max(ap for ap, _qi in q_entries)
                    local_end = int(last_ap) - int(s) + 1
                    visible_slots: List[torch.Tensor] = []
                    for ss, ee, seg_priv in all_segs_in_order:
                        if ee <= s:
                            visible_slots.append(seg_priv.to(device))
                        elif ss == s and ee == e:
                            visible_slots.append(seg_priv.to(device)[:local_end])
                            break
                    add_phase_c_segment(q_entries, visible_slots)

            if global_info is not None:
                (gs, ge, gp) = global_info
                gseg_len = ge - gs
                gq_slice = global_q_indices[gq_cursor:gq_cursor + gseg_len]
                gq_cursor += gseg_len
                q_entries = [(gs + j, qi) for j, qi in enumerate(gq_slice)]
                # Schedule K/V write: dst = gp, src = gq_slice (real-pos aligned).
                kv_dst_slots_chunks.append(gp.to(device).to(torch.int64))
                kv_src_qis_list.extend(gq_slice)
                visible_slots = [priv.to(device) for _s, _e, priv in all_segs_in_order]
                visible_slots.append(gp.to(device))
                add_phase_c_segment(q_entries, visible_slots)

            # Miss-seg Q rows: each Q row → 1 FlashInfer
            # prefill segment with causal KV slice over prior + own-prefix.
            miss_q_for_req = sorted(
                [
                    (self._pic_q_to_req_abs_dict[qi][1], qi)
                    for qi in (local_miss_q_indices or [])
                    if self._pic_q_to_req_abs_dict.get(qi, (-1, -1))[0] == req_idx
                ],
                key=lambda x: x[0],
            )
            for (abs_pos, qi) in miss_q_for_req:
                visible_slots_m: List[torch.Tensor] = []
                for (s, e, priv) in all_segs_in_order:
                    if e <= abs_pos + 1:
                        visible_slots_m.append(priv.to(device))
                    elif s <= abs_pos < e:
                        local_end = abs_pos - s + 1
                        visible_slots_m.append(priv.to(device)[:local_end])
                        break
                    else:
                        break
                add_phase_c_segment([(abs_pos, qi)], visible_slots_m)

        if not q_packed_src:
            return None

        num_segs = len(seg_q_offsets) - 1
        return {
            "q_packed_src": torch.tensor(q_packed_src, dtype=torch.int64, device=device),
            "kv_dst_slots": torch.cat(kv_dst_slots_chunks, dim=0)
                if kv_dst_slots_chunks else torch.empty(0, dtype=torch.int64, device=device),
            "kv_src_qis": torch.tensor(kv_src_qis_list, dtype=torch.int64, device=device)
                if kv_src_qis_list else torch.empty(0, dtype=torch.int64, device=device),
            "kv_indices": torch.cat(kv_indices_chunks, dim=0),
            "qo_indptr": torch.tensor(seg_q_offsets, dtype=torch.int32, device=device),
            "kv_indptr": torch.tensor(seg_kv_offsets, dtype=torch.int32, device=device),
            "kv_last_page_len": torch.ones(num_segs, dtype=torch.int32, device=device),
            "scatter_idx": torch.tensor(scatter_q_idx_all, dtype=torch.int64, device=device),
        }

    def _init_pic_rope_plans(self, forward_batch: ForwardBatch):
        """Placeholder for parity with _init_pic_prefill_plan: rope plans are
        built inside Phase A / Phase C (per-batch via begin_forward). We only
        need to reset the lazy wrapper-ready flags here."""
        self._pic_rope_local_plan_ready = False
        self._pic_rope_global_plan_ready = False
        self._pic_rope_phase_c_cache = None
        self._pic_rope_phase_b_cache = None
        self._pic_rope_cross_real_cache = None

    def _init_pic_rope_cache(self):
        """Lazily initialize rotary embedding cache for transition_rope mode.

        Finds the rotary_emb on the model and stores cos_sin_cache, is_neox_style,
        and rotary_dim for use by Phase A/B/C derotate/rerotate ops.
        """
        if self._model_runner is None:
            logger.warning("_init_pic_rope_cache: no model_runner, cannot init rope cache")
            return
        model = self._model_runner.model
        # Walk model to find rotary_emb. Common locations:
        # model.model.layers[i].self_attn.rotary_emb (Qwen, Llama, etc.)
        # model.model.rotary_emb (some architectures)
        rotary_emb = None
        rotary_source = None
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "rotary_emb"):
                rotary_emb = inner.rotary_emb
                rotary_source = "model.model.rotary_emb"
            elif hasattr(inner, "layers"):
                fallback = None
                fallback_source = None
                for idx, layer_module in enumerate(inner.layers):
                    # None = the layer itself
                    for attr in ("self_attn", "attention", None):
                        attn_holder = (
                            layer_module
                            if attr is None
                            else getattr(layer_module, attr, None)
                        )
                        if attn_holder is None or not hasattr(attn_holder, "rotary_emb"):
                            continue
                        prefix = f"{attr}." if attr else ""
                        source = f"model.model.layers.{idx}.{prefix}rotary_emb"
                        if fallback is None:
                            fallback = attn_holder.rotary_emb
                            fallback_source = source
                        attn = getattr(attn_holder, "attn", None)
                        layer_id = getattr(attn, "layer_id", None)
                        if layer_id in self.full_attn_layers:
                            rotary_emb = attn_holder.rotary_emb
                            rotary_source = source
                            break
                    if rotary_emb is not None:
                        break
                if rotary_emb is None:
                    rotary_emb = fallback
                    rotary_source = fallback_source
        if rotary_emb is None:
            # Fallback: search all modules
            for name, module in model.named_modules():
                if hasattr(module, "cos_sin_cache") and hasattr(module, "is_neox_style"):
                    rotary_emb = module
                    rotary_source = name
                    break
        if rotary_emb is None:
            logger.error("_init_pic_rope_cache: could not find rotary_emb on model")
            return

        self._pic_rope_cos_sin_cache = rotary_emb.cos_sin_cache
        self._pic_rope_is_neox = rotary_emb.is_neox_style
        self._pic_rope_rotary_dim = rotary_emb.rotary_dim
        logger.info(
            "PIC transition_rope: initialized rope cache from %s "
            "(rotary_dim=%d, is_neox=%s, cache_len=%d, dtype=%s)",
            rotary_source,
            self._pic_rope_rotary_dim,
            self._pic_rope_is_neox,
            self._pic_rope_cos_sin_cache.shape[0],
            self._pic_rope_cos_sin_cache.dtype,
        )

    def _init_pic_prefill_plan(self, forward_batch: ForwardBatch):
        """Build FlashInfer prefill plan for PIC per-segment full-attn.

        Called once per batch in init_forward_metadata. The wrapper + plan is
        reused by all full-attn layers in _forward_extend_pic_full_attn.
        """
        from flashinfer import BatchPrefillWithPagedKVCacheWrapper

        device = forward_batch.input_ids.device
        pic_miss_segments = forward_batch.pic_miss_segments
        batch_size = forward_batch.batch_size

        seg_q_lens = []
        seg_kv_lens = []
        seg_req_indices = []

        for req_idx in range(batch_size):
            miss_segs = pic_miss_segments[req_idx]
            for (start, end) in miss_segs:
                seg_q_lens.append(end - start)
                seg_kv_lens.append(end)
                seg_req_indices.append(req_idx)

            # Recompute mode adds seam tokens (sink from hit segments)
            # to the q tensor. Include them as extra query segments so the
            # FlashInfer plan covers all q rows.
            rope_meta = getattr(forward_batch, "pic_rope_meta", None)
            if rope_meta is not None and req_idx < len(rope_meta):
                meta = rope_meta[req_idx]
                seam_info = meta.get("seam") if isinstance(meta, dict) else None
                if seam_info is not None:
                    hit_seam = seam_info.get("hit_seam", {})
                    total_seam = sum(len(sp) for sp in hit_seam.values())
                    if total_seam > 0:
                        # Find the max KV length for this request's miss segs
                        max_kv = miss_segs[-1][1] if miss_segs else 0
                        seg_q_lens.append(total_seam)
                        seg_kv_lens.append(max_kv)
                        seg_req_indices.append(req_idx)

        num_segs = len(seg_q_lens)
        if num_segs == 0:
            self._pic_has_plan = False
            return

        qo_indptr = torch.zeros(num_segs + 1, dtype=torch.int32, device=device)
        for i, ql in enumerate(seg_q_lens):
            qo_indptr[i + 1] = qo_indptr[i] + ql

        req_to_token = self.req_to_token_pool.req_to_token
        req_pool_indices = forward_batch.req_pool_indices

        kv_indptr = torch.zeros(num_segs + 1, dtype=torch.int32, device=device)
        kv_indices_list = []
        kv_positions_list = []
        for seg_i in range(num_segs):
            req_idx = seg_req_indices[seg_i]
            kv_len = seg_kv_lens[seg_i]
            req_pool_idx = req_pool_indices[req_idx]
            slots = req_to_token[req_pool_idx, :kv_len]
            kv_indices_list.append(slots)
            kv_positions_list.append(
                torch.arange(kv_len, dtype=torch.int64, device=device)
            )
            kv_indptr[seg_i + 1] = kv_indptr[seg_i] + kv_len

        kv_indices = torch.cat(kv_indices_list).to(torch.int32)
        kv_positions = torch.cat(kv_positions_list).to(torch.int64)
        kv_last_page_len = torch.ones(num_segs, dtype=torch.int32, device=device)

        if self._pic_workspace is None:
            self._pic_workspace = torch.empty(
                128 * 1024 * 1024, dtype=torch.uint8, device=device
            )

        # T1: construct wrapper once, reuse across batches. Recreating per
        # batch was the OOM root cause — each rebuild left the prior wrapper's
        # internal FlashInfer state-tensors to be reclaimed asynchronously
        # by Python GC, fragmenting the caching allocator until small allocs
        # could not be satisfied. wrapper.begin_forward (called below in
        # _forward_extend_pic_full_attn) re-plans on the same instance.
        if self._pic_prefill_wrapper is None:
            self._pic_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                float_workspace_buffer=self._pic_workspace,
            )
        self._pic_has_plan = True
        self._pic_plan_ready = False
        self._pic_qo_indptr = qo_indptr
        self._pic_kv_indptr = kv_indptr
        self._pic_kv_indices = kv_indices
        self._pic_kv_positions = kv_positions
        self._pic_kv_last_page_len = kv_last_page_len

    def _forward_extend_pic_full_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """PIC full-attn: use pre-built FlashInfer plan from init_forward_metadata.

        begin_forward is called once in _init_pic_prefill_plan; each layer
        only calls wrapper.forward (the heavy GPU kernel) without re-planning.
        """
        # FlashInfer SM90 only supports head_dim in {64, 128, 256}. For MLA
        # (head_dim=192) or other unsupported dims, fall back to the standard
        # extend path which handles causal attention correctly via the model's
        # native attention backend (fa3/triton).
        _FLASHINFER_SUPPORTED_DIMS = {64, 128, 256}
        if layer.head_dim not in _FLASHINFER_SUPPORTED_DIMS:
            return self._forward_extend_pic_full_attn_native(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )

        cache_loc = forward_batch.out_cache_loc
        if k is not None and v is not None and save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, cache_loc, k, v, layer.k_scale, layer.v_scale
            )

        if not self._pic_has_plan:
            return q.new_zeros(0, layer.tp_q_head_num * layer.head_dim)

        wrapper = self._pic_prefill_wrapper
        if wrapper is None:
            return q.new_zeros(0, layer.tp_q_head_num * layer.head_dim)

        if not self._pic_plan_ready:
            wrapper.begin_forward(
                qo_indptr=self._pic_qo_indptr,
                paged_kv_indptr=self._pic_kv_indptr,
                paged_kv_indices=self._pic_kv_indices,
                paged_kv_last_page_len=self._pic_kv_last_page_len,
                num_qo_heads=layer.tp_q_head_num,
                num_kv_heads=layer.tp_k_head_num,
                head_dim_qk=layer.head_dim,
                page_size=1,
                causal=True,
                q_data_type=q.dtype,
            )
            self._pic_plan_ready = True

        kv_buf = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        _dump_moti2_context_kv(
            layer,
            q,
            kv_buf,
            self._pic_kv_indices,
            forward_batch.positions,
            self._pic_qo_indptr,
            self._pic_kv_indptr,
            self._pic_kv_positions,
        )
        o = wrapper.forward(
            q.view(-1, layer.tp_q_head_num, layer.head_dim),
            kv_buf,
            causal=True,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _forward_extend_pic_full_attn_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Fallback for PIC full-attn when FlashInfer doesn't support the head_dim.

        Delegates to the standard full-attn extend backend (fa3/triton) which
        handles causal attention correctly via the model's native path. For MLA
        models this goes through absorb+decompress → fa3 paged attention.
        """
        return self.full_attn_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: Optional[torch.Tensor] = None,  # For full attention
        k: Optional[torch.Tensor] = None,  # For full attention
        v: Optional[torch.Tensor] = None,  # For full attention
        layer: RadixAttention = None,
        forward_batch: ForwardBatch = None,
        save_kv_cache: bool = True,
        mixed_qkv: Optional[torch.Tensor] = None,  # For linear attention
        a: Optional[torch.Tensor] = None,  # For linear attention
        b: Optional[torch.Tensor] = None,  # For linear attention
        **kwargs,
    ):
        is_linear_attn = not self._is_full_attn(layer, kwargs.get("layer_id"))

        if forward_batch.forward_mode.is_idle():
            if is_linear_attn:
                return mixed_qkv.new_empty(
                    mixed_qkv.shape[0], layer.num_v_heads, layer.head_v_dim
                )
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )
        else:
            return self.forward_extend(
                layer,
                forward_batch,
                save_kv_cache,
                q,
                k,
                v,
                mixed_qkv,
                a,
                b,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: Optional[torch.Tensor],
        mamba_steps_to_track: Optional[torch.Tensor],
        model,
    ):
        """
        Update mamba states after MTP verify using fully fused Triton kernel.

        This replaces the original advanced indexing operations with a single fused
        gather-scatter kernel that also handles masking internally, avoiding:
        - index_elementwise_kernel from tensor[bool_mask]
        - index_select kernel launches
        - nonzero kernel launches
        """
        request_number = last_correct_step_indices.shape[0]

        state_indices_tensor = (
            self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                :request_number
            ]
        )

        mamba_caches = (
            self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        )

        conv_states = mamba_caches.conv[0]
        ssm_states = mamba_caches.temporal
        intermediate_state_cache = mamba_caches.intermediate_ssm
        intermediate_conv_window_cache = mamba_caches.intermediate_conv_window[0]

        # Use fully fused kernel that handles masking internally
        # This avoids separate nonzero() and index_select() calls
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
        # conv intermediate uses the deduplicated sliding-window (overlapping)
        # layout, so it needs the strided-read scatter variant.
        fused_conv_window_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )

        # Track indices used for tracking mamba states for prefix cache
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            # Use fully fused kernel for track scatter operations
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )
