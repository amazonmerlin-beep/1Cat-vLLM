// Attention de vérification spéculative groupée (GQA × jetons) — passe « partition » en CUDA WMMA (sm_70).
//
// Une CTA = (requête r, tête KV h, partition p de `ps` clés). Elle charge les G têtes q du groupe ×
// jusqu'à 32/G jetons de la requête (32 lignes, ligne = j*G + g), lit chaque bloc de 32 clés K puis V
// une seule fois (cache paginé, adresses par block_table), calcule S = Q·Kᵀ et P·V sur cœurs tenseurs
// (fp16 → fp32), softmax en ligne fp32 avec masque causal propre à chaque jeton (jeton j voit les clés
// [0, eff - q_len + j + 1), eff = max(seq_len, q_len)). Sortie : partiel normalisé fp32 + (max, somme)
// par (jeton, tête q, partition) ; la combinaison des partitions est faite par le noyau Triton de
// gqa_verify.py. Aucune allocation, flux courant : compatible capture en graphe CUDA.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/library.h>
#include <torch/torch.h>

using namespace nvcuda;

namespace {

constexpr int D = 256;
constexpr int M = 32;    // lignes (G têtes × jetons)
constexpr int BN = 32;   // clés par bloc
constexpr int LDQ = D + 8;   // demi-mots
constexpr int LDK = D + 8;   // demi-mots
constexpr int LDS = BN + 8;  // flottants
constexpr int LDP = BN + 8;  // demi-mots
constexpr int LDT = D + 8;   // flottants
constexpr int NT = 256;      // 8 warps
constexpr int SMEM_BYTES = M * LDQ * 2 + BN * LDK * 2 + M * LDS * 4 + M * LDP * 2 + M * LDT * 4;

template <int G>
__global__ void __launch_bounds__(NT, 1) gqa_partition_kernel(
    const half* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc,
    const int* __restrict__ bt, const int* __restrict__ seq_lens, const int* __restrict__ qsl,
    float* __restrict__ po, float* __restrict__ pm, float* __restrict__ pl,
    int n_tokens, float scale, float softcap,
    int64_t s_qt, int64_t s_qh,
    int64_t s_kb, int64_t s_ks, int64_t s_kh,
    int64_t s_vb, int64_t s_vs, int64_t s_vh,
    int64_t s_btr,
    int64_t s_pot, int64_t s_poh, int64_t s_pop,
    int64_t s_pmt, int64_t s_pmh,
    int page_size, int ps, int per_token) {
  constexpr int MAXQ = M / G;
  const int r = blockIdx.x, h = blockIdx.y, p = blockIdx.z;
  const int qs = qsl[r];
  const int qlen = qsl[r + 1] - qs;
  if (qlen <= 0) return;
  // per_token : métadonnées persistantes « small-query » du fork (une ligne de block_table et un
  // seq_len de décodage par jeton) ; le seq_len du dernier jeton de la requête vaut eff.
  const int eff = per_token ? seq_lens[qs + qlen - 1] : max(seq_lens[r], qlen);
  const int ctx0 = eff - qlen;
  const int kv_start = p * ps;
  if (kv_start >= eff) return;
  const int kv_end = min(kv_start + ps, eff);

  extern __shared__ __align__(128) unsigned char smem[];
  half* sQ = reinterpret_cast<half*>(smem);
  half* sKV = sQ + M * LDQ;
  float* sS = reinterpret_cast<float*>(sKV + BN * LDK);
  half* sP = reinterpret_cast<half*>(sS + M * LDS);
  float* sT = reinterpret_cast<float*>(sP + M * LDP);

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int* btr = bt + static_cast<int64_t>(per_token ? qs : r) * s_btr;

  // --- Q -> smem (lignes invalides à zéro)
  for (int i = tid; i < M * (D / 8); i += NT) {
    const int row = i / (D / 8), c8 = i % (D / 8);
    const int j = row / G, g = row % G;
    const int tok = qs + j;
    int4 val = make_int4(0, 0, 0, 0);
    if (j < qlen && j < MAXQ && tok < n_tokens)
      val = *reinterpret_cast<const int4*>(q + static_cast<int64_t>(tok) * s_qt +
                                           static_cast<int64_t>(h * G + g) * s_qh + c8 * 8);
    *reinterpret_cast<int4*>(sQ + row * LDQ + c8 * 8) = val;
  }

  // propriété des lignes : 8 fils par ligne ; colonnes entrelacées (col = sub + 8*c)
  const int my_row = tid >> 3;
  const int sub = tid & 7;
  const int my_j = my_row / G;
  const bool row_valid = (my_j < qlen) && (my_j < MAXQ) && (qs + my_j < n_tokens);
  const int row_limit = ctx0 + my_j + 1;
  float m_run = -INFINITY, l_run = 0.f;
  float acc[D / 8];
#pragma unroll
  for (int c = 0; c < D / 8; ++c) acc[c] = 0.f;

  // chargement d'un bloc de 32 clés (K ou V) dans des registres : 4 × int4 par fil
  constexpr int PER = BN * (D / 8) / NT;  // = 4
  int4 reg[PER];
  auto load_block = [&](const half* __restrict__ base, int64_t sb, int64_t ss, int64_t sh, int n0) {
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = tid + u * NT;
      const int kk = i / (D / 8), c8 = i % (D / 8);
      const int n = n0 + kk;
      int4 val = make_int4(0, 0, 0, 0);
      if (n < kv_end) {
        int pg = btr[n / page_size];
        pg = pg < 0 ? 0 : pg;
        const int sl = n % page_size;
        val = *reinterpret_cast<const int4*>(base + static_cast<int64_t>(pg) * sb +
                                             static_cast<int64_t>(sl) * ss +
                                             static_cast<int64_t>(h) * sh + c8 * 8);
      }
      reg[u] = val;
    }
  };
  auto store_block = [&]() {
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = tid + u * NT;
      const int kk = i / (D / 8), c8 = i % (D / 8);
      *reinterpret_cast<int4*>(sKV + kk * LDK + c8 * 8) = reg[u];
    }
  };

  load_block(kc, s_kb, s_ks, s_kh, kv_start);
  for (int n0 = kv_start; n0 < kv_end; n0 += BN) {
    store_block();  // K
    __syncthreads();
    load_block(vc, s_vb, s_vs, s_vh, n0);  // V en vol pendant Q·Kᵀ

    if (warp < 4) {
      const int tm = warp >> 1, tn = warp & 1;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
      wmma::fill_fragment(cf, 0.f);
#pragma unroll
      for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
        wmma::load_matrix_sync(af, sQ + tm * 16 * LDQ + kk, LDQ);
        wmma::load_matrix_sync(bf, sKV + tn * 16 * LDK + kk, LDK);
        wmma::mma_sync(cf, af, bf, cf);
      }
      wmma::store_matrix_sync(sS + tm * 16 * LDS + tn * 16, cf, LDS, wmma::mem_row_major);
    }
    __syncthreads();

    {  // softmax en ligne
      float sv[BN / 8];
      float cmax = -INFINITY;
#pragma unroll
      for (int c = 0; c < BN / 8; ++c) {
        const int col = sub + 8 * c;
        const int n = n0 + col;
        float s = sS[my_row * LDS + col] * scale;
        if (softcap > 0.f) s = softcap * tanhf(s / softcap);
        const bool vis = row_valid && (n < kv_end) && (n < row_limit);
        sv[c] = vis ? s : -INFINITY;
        cmax = fmaxf(cmax, sv[c]);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, off));
      const float m_new = fmaxf(m_run, cmax);
      const float m_safe = (m_new == -INFINITY) ? 0.f : m_new;
      const float alpha = __expf(m_run - m_safe);
      float lsum = 0.f;
#pragma unroll
      for (int c = 0; c < BN / 8; ++c) {
        const float e = __expf(sv[c] - m_safe);
        lsum += e;
        sP[my_row * LDP + sub + 8 * c] = __float2half(e);
      }
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffu, lsum, off);
      l_run = l_run * alpha + lsum;
      m_run = m_new;
#pragma unroll
      for (int c = 0; c < D / 8; ++c) acc[c] *= alpha;
    }
    store_block();  // V (K n'est plus lu : barrière après Q·Kᵀ)
    __syncthreads();
    if (n0 + BN < kv_end) load_block(kc, s_kb, s_ks, s_kh, n0 + BN);  // K suivant en vol pendant P·V

    {  // P·V : 2 × 16 tuiles, 4 par warp
      const int tm = warp >> 2;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int tn = (warp & 3) * 4 + t;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
        wmma::fill_fragment(cf, 0.f);
#pragma unroll
        for (int kk = 0; kk < BN; kk += 16) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
          wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
          wmma::load_matrix_sync(af, sP + tm * 16 * LDP + kk, LDP);
          wmma::load_matrix_sync(bf, sKV + kk * LDK + tn * 16, LDK);
          wmma::mma_sync(cf, af, bf, cf);
        }
        wmma::store_matrix_sync(sT + tm * 16 * LDT + tn * 16, cf, LDT, wmma::mem_row_major);
      }
    }
    __syncthreads();
#pragma unroll
    for (int c = 0; c < D / 8; ++c) acc[c] += sT[my_row * LDT + sub + 8 * c];
    __syncthreads();
  }

  if (row_valid) {
    const int g = my_row % G;
    const int64_t tok = qs + my_j;
    const int64_t qh = h * G + g;
    const float inv = l_run > 0.f ? 1.f / l_run : 0.f;
    float* dst = po + tok * s_pot + qh * s_poh + static_cast<int64_t>(p) * s_pop;
#pragma unroll
    for (int c = 0; c < D / 8; ++c) dst[sub + 8 * c] = acc[c] * inv;
    if (sub == 0) {
      pm[tok * s_pmt + qh * s_pmh + p] = m_run;
      pl[tok * s_pmt + qh * s_pmh + p] = l_run;
    }
  }
}

void gqa_partition(const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& bt,
                   const at::Tensor& seq_lens, const at::Tensor& qsl, at::Tensor& po, at::Tensor& pm,
                   at::Tensor& pl, int64_t num_reqs, int64_t n_parts, int64_t ps, double scale,
                   double softcap, int64_t per_token) {
  TORCH_CHECK(q.is_cuda() && kc.is_cuda() && vc.is_cuda(), "gqa_verify : tenseurs CUDA attendus");
  TORCH_CHECK(q.scalar_type() == at::kHalf && kc.scalar_type() == at::kHalf && vc.scalar_type() == at::kHalf,
              "gqa_verify : q/k/v en fp16");
  TORCH_CHECK(bt.scalar_type() == at::kInt && seq_lens.scalar_type() == at::kInt && qsl.scalar_type() == at::kInt,
              "gqa_verify : métadonnées int32");
  TORCH_CHECK(po.scalar_type() == at::kFloat && pm.scalar_type() == at::kFloat && pl.scalar_type() == at::kFloat);
  TORCH_CHECK(q.dim() == 3 && kc.dim() == 4 && vc.dim() == 4);
  TORCH_CHECK(q.size(2) == D && kc.size(3) == D && vc.size(3) == D, "gqa_verify : head_dim 256");
  TORCH_CHECK(q.stride(2) == 1 && kc.stride(3) == 1 && vc.stride(3) == 1 && po.stride(3) == 1);
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && kc.stride(0) % 8 == 0 && kc.stride(1) % 8 == 0 &&
                  kc.stride(2) % 8 == 0 && vc.stride(0) % 8 == 0 && vc.stride(1) % 8 == 0 && vc.stride(2) % 8 == 0,
              "gqa_verify : pas multiples de 8 (accès 16 octets)");
  TORCH_CHECK(bt.stride(1) == 1 && seq_lens.stride(0) == 1 && qsl.stride(0) == 1);
  const int64_t hkv = kc.size(2);
  TORCH_CHECK(q.size(1) % hkv == 0);
  const int64_t G = q.size(1) / hkv;
  TORCH_CHECK(G == 4, "gqa_verify : seul le groupe G = 4 est compilé");
  TORCH_CHECK(ps % BN == 0 && ps > 0);
  if (num_reqs <= 0 || n_parts <= 0 || q.size(0) == 0) return;
  const c10::cuda::CUDAGuard guard(q.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  static bool attr_done = false;
  if (!attr_done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(gqa_partition_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        SMEM_BYTES));
    attr_done = true;
  }
  dim3 grid(static_cast<unsigned>(num_reqs), static_cast<unsigned>(hkv), static_cast<unsigned>(n_parts));
  gqa_partition_kernel<4><<<grid, NT, SMEM_BYTES, stream>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), reinterpret_cast<const half*>(kc.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(vc.data_ptr<at::Half>()), bt.data_ptr<int>(), seq_lens.data_ptr<int>(),
      qsl.data_ptr<int>(), po.data_ptr<float>(), pm.data_ptr<float>(), pl.data_ptr<float>(),
      static_cast<int>(q.size(0)), static_cast<float>(scale), static_cast<float>(softcap), q.stride(0), q.stride(1),
      kc.stride(0), kc.stride(1), kc.stride(2), vc.stride(0), vc.stride(1), vc.stride(2), bt.stride(0),
      po.stride(0), po.stride(1), po.stride(2), pm.stride(0), pm.stride(1), static_cast<int>(kc.size(1)),
      static_cast<int>(ps), static_cast<int>(per_token));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

TORCH_LIBRARY(gqa_verify, m) {
  m.def(
      "partition(Tensor q, Tensor kc, Tensor vc, Tensor bt, Tensor seq_lens, Tensor qsl, Tensor(a!) po, "
      "Tensor(b!) pm, Tensor(c!) pl, int num_reqs, int n_parts, int ps, float scale, float softcap, int per_token) -> ()");
}

TORCH_LIBRARY_IMPL(gqa_verify, CUDA, m) { m.impl("partition", &gqa_partition); }
