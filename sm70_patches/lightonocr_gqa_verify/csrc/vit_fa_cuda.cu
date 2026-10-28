// Attention de la tour vision (Qwen3-VL, LightOnOCR-3) : FlashAttention avant, non causale, d = 64, en CUDA
// WMMA pour sm_70 (V100). Remplace la SDPA « mem-efficient » (cutlassF sm70, ~27 TFLOPS).
//
// Une CTA = (tuile de 64 requêtes, tête, séquence) ; 4 warps × 16 lignes ; blocs de BN clés.
// Par warp et par bloc : S = Q·Kᵀ (WMMA fp16 → fp32) → mémoire partagée → softmax en ligne fp32 (2 fils par
// ligne) → P fp16 → acc += P·V directement dans les fragments WMMA. Remise à l'échelle paresseuse et exacte :
// max de référence par ligne, l'accumulateur n'est remis à l'échelle (aller-retour en mémoire partagée, un
// fragment à la fois) que si un nouveau max dépasse la référence de TAU (log2) ; P = 2^(s − m_ref) ≤ 2^TAU.
// Variantes : <BN = 32, 4 CTA/SM> (18 Ko de mémoire partagée) et <BN = 64, 2 CTA/SM> (36 Ko).
// Séquences multiples par cu_seqlens (images d'un lot) ; la grille est bornée par la longueur totale (pas de
// synchronisation hôte) et les CTA hors séquence sortent aussitôt.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/library.h>
#include <torch/torch.h>

using namespace nvcuda;

namespace {

constexpr int D = 64;
constexpr int BM = 64;
constexpr int NW = 4;
constexpr int NT = NW * 32;
constexpr int LDH = D + 8;  // demi-mots (72)
constexpr int LDA = 20;     // flottants, tampon 16×16 d'un fragment d'accumulateur

template <int BN>
struct Smem {
  static constexpr int LDF = BN + 4;  // S (flottants)
  static constexpr int LDP = BN + 8;  // P (demi-mots), rangé sur la zone S du warp une fois S lu
  static constexpr int KV = BN * LDH * 2;           // octets par tenseur K ou V
  static constexpr int SW = 16 * LDF * 4;           // octets de la zone S d'un warp
  static constexpr int BYTES = 2 * KV + NW * SW;
  static_assert(2 * KV >= BM * LDH * 2, "Q est d'abord rangé dans la zone K/V");
  static_assert(SW >= 16 * LDP * 2 && SW >= 16 * LDA * 4, "zone S trop petite");
};

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

template <int BN, int MINB>
__global__ void __launch_bounds__(NT, MINB) vit_fa_kernel(
    const half* __restrict__ q, const half* __restrict__ k, const half* __restrict__ v, half* __restrict__ o,
    const int* __restrict__ cu, float scale_log2,
    int64_t s_qt, int64_t s_qh, int64_t s_kt, int64_t s_kh, int64_t s_vt, int64_t s_vh, int64_t s_ot, int64_t s_oh) {
  using SM = Smem<BN>;
  constexpr int LDF = SM::LDF, LDP = SM::LDP;
  constexpr float TAU = 8.f;
  const int qt = blockIdx.x, h = blockIdx.y, sq = blockIdx.z;
  const int start = cu[sq];
  const int L = cu[sq + 1] - start;
  const int q0 = qt * BM;
  if (q0 >= L) return;

  extern __shared__ __align__(128) unsigned char smem[];
  half* sK = reinterpret_cast<half*>(smem);
  half* sV = sK + BN * LDH;
  float* sSw = reinterpret_cast<float*>(smem + 2 * SM::KV + (threadIdx.x >> 5) * SM::SW);
  half* sPw = reinterpret_cast<half*>(sSw);
  float* sAw = sSw;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

  // Q → (zone K/V) → fragments gardés en registres
  half* sQ = sK;
  for (int i = tid; i < BM * (D / 8); i += NT) {
    const int r = i / (D / 8), c8 = i % (D / 8);
    int4 val = make_int4(0, 0, 0, 0);
    if (q0 + r < L)
      val = *reinterpret_cast<const int4*>(q + static_cast<int64_t>(start + q0 + r) * s_qt + h * s_qh + c8 * 8);
    *reinterpret_cast<int4*>(sQ + r * LDH + c8 * 8) = val;
  }
  __syncthreads();
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[D / 16];
#pragma unroll
  for (int kk = 0; kk < D / 16; ++kk) wmma::load_matrix_sync(qf[kk], sQ + warp * 16 * LDH + kk * 16, LDH);

  const int row = lane >> 1, hf = lane & 1;
  float m_ref = -INFINITY, l_run = 0.f;
  bool premier = true;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accf[D / 16];
#pragma unroll
  for (int t = 0; t < D / 16; ++t) wmma::fill_fragment(accf[t], 0.f);

  for (int n0 = 0; n0 < L; n0 += BN) {
    __syncthreads();  // sK/sV (et Q au premier tour) libres
#pragma unroll
    for (int u = 0; u < BN * (D / 8) / NT; ++u) {
      const int i = tid + u * NT;
      const int r = i / (D / 8), c8 = i % (D / 8);
      const int n = n0 + r;
      int4 kv = make_int4(0, 0, 0, 0), vv = make_int4(0, 0, 0, 0);
      if (n < L) {
        kv = *reinterpret_cast<const int4*>(k + static_cast<int64_t>(start + n) * s_kt + h * s_kh + c8 * 8);
        vv = *reinterpret_cast<const int4*>(v + static_cast<int64_t>(start + n) * s_vt + h * s_vh + c8 * 8);
      }
      *reinterpret_cast<int4*>(sK + r * LDH + c8 * 8) = kv;
      *reinterpret_cast<int4*>(sV + r * LDH + c8 * 8) = vv;
    }
    __syncthreads();

    // S_w = Q_w · Kᵀ (16 × BN)
#pragma unroll
    for (int tn = 0; tn < BN / 16; ++tn) {
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
      wmma::fill_fragment(cf, 0.f);
#pragma unroll
      for (int kk = 0; kk < D / 16; ++kk) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
        wmma::load_matrix_sync(bf, sK + tn * 16 * LDH + kk * 16, LDH);
        wmma::mma_sync(cf, qf[kk], bf, cf);
      }
      wmma::store_matrix_sync(sSw + tn * 16, cf, LDF, wmma::mem_row_major);
    }
    __syncwarp();

    const int nvalid = min(BN, L - n0);
    float sv[BN / 2];
    float cmax = -INFINITY;
#pragma unroll
    for (int c = 0; c < BN / 2; ++c) {
      const int col = hf + 2 * c;
      const float s = sSw[row * LDF + col] * scale_log2;
      sv[c] = col < nvalid ? s : -INFINITY;
      cmax = fmaxf(cmax, sv[c]);
    }
    cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
    __syncwarp();  // S lu par tout le warp : la zone sert maintenant de tampon
    const bool besoin = cmax > m_ref + TAU;  // vrai au premier bloc
    if (__any_sync(0xffffffffu, besoin)) {
      const float m_new = besoin ? cmax : m_ref;
      const float alpha = premier ? 0.f : ex2(m_ref - m_new);
      l_run *= alpha;
      if (!premier) {
#pragma unroll
        for (int t = 0; t < D / 16; ++t) {
          wmma::store_matrix_sync(sAw, accf[t], LDA, wmma::mem_row_major);
          __syncwarp();
#pragma unroll
          for (int c = 0; c < 8; ++c) sAw[row * LDA + hf * 8 + c] *= alpha;
          __syncwarp();
          wmma::load_matrix_sync(accf[t], sAw, LDA, wmma::mem_row_major);
          __syncwarp();
        }
      }
      m_ref = m_new;
      premier = false;
    }
    float lsum = 0.f;
#pragma unroll
    for (int c = 0; c < BN / 2; ++c) {
      const float e = ex2(sv[c] - m_ref);
      lsum += e;
      sPw[row * LDP + hf + 2 * c] = __float2half(e);
    }
    lsum += __shfl_xor_sync(0xffffffffu, lsum, 1);
    l_run += lsum;
    __syncwarp();

    // acc_w += P_w · V
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf[BN / 16];
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) wmma::load_matrix_sync(pf[kk], sPw + kk * 16, LDP);
#pragma unroll
    for (int tn = 0; tn < D / 16; ++tn) {
#pragma unroll
      for (int kk = 0; kk < BN / 16; ++kk) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
        wmma::load_matrix_sync(bf, sV + kk * 16 * LDH + tn * 16, LDH);
        wmma::mma_sync(accf[tn], pf[kk], bf, accf[tn]);
      }
    }
    __syncwarp();
  }

  // sortie : un fragment à la fois → tampon → 8 colonnes par fil, écriture 16 octets
  const int qr = q0 + warp * 16 + row;
  const float inv = 1.f / l_run;
#pragma unroll
  for (int t = 0; t < D / 16; ++t) {
    wmma::store_matrix_sync(sAw, accf[t], LDA, wmma::mem_row_major);
    __syncwarp();
    if (qr < L) {
      __align__(16) half vals[8];
#pragma unroll
      for (int c = 0; c < 8; ++c) vals[c] = __float2half(sAw[row * LDA + hf * 8 + c] * inv);
      *reinterpret_cast<int4*>(o + static_cast<int64_t>(start + qr) * s_ot + h * s_oh + t * 16 + hf * 8) =
          *reinterpret_cast<const int4*>(vals);
    }
    __syncwarp();
  }
}

template <int BN, int MINB>
void lancer(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& o, const at::Tensor& cu,
            int64_t max_seqlen, double scale) {
  static bool attr_done = false;
  if (!attr_done) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(vit_fa_kernel<BN, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Smem<BN>::BYTES));
    attr_done = true;
  }
  dim3 grid(static_cast<unsigned>((max_seqlen + BM - 1) / BM), static_cast<unsigned>(q.size(1)),
            static_cast<unsigned>(cu.size(0) - 1));
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  vit_fa_kernel<BN, MINB><<<grid, NT, Smem<BN>::BYTES, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()), reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()), reinterpret_cast<half*>(o.data_ptr<at::Half>()),
      cu.data_ptr<int>(), scale_log2, q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
      o.stride(0), o.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void vit_fa_forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& o,
                    const at::Tensor& cu, int64_t max_seqlen, double scale, int64_t variant) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && o.is_cuda() && cu.is_cuda());
  TORCH_CHECK(q.scalar_type() == at::kHalf && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
              o.scalar_type() == at::kHalf, "vit_fa : fp16 attendu");
  TORCH_CHECK(cu.scalar_type() == at::kInt && cu.dim() == 1 && cu.stride(0) == 1);
  TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && o.dim() == 3, "vit_fa : [T, H, 64]");
  TORCH_CHECK(q.size(2) == D && k.size(2) == D && v.size(2) == D && o.size(2) == D, "vit_fa : d = 64");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1) && q.size(1) == o.size(1), "vit_fa : pas de GQA");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.stride(2) == 1);
  TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0 && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 &&
                  v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0 && o.stride(0) % 8 == 0 && o.stride(1) % 8 == 0,
              "vit_fa : pas multiples de 8");
  if (cu.size(0) - 1 <= 0 || max_seqlen <= 0) return;
  const c10::cuda::CUDAGuard guard(q.device());
  if (variant == 64)
    lancer<64, 2>(q, k, v, o, cu, max_seqlen, scale);
  else
    lancer<32, 4>(q, k, v, o, cu, max_seqlen, scale);
}

}  // namespace

TORCH_LIBRARY(vit_fa, m) {
  m.def(
      "forward(Tensor q, Tensor k, Tensor v, Tensor(a!) o, Tensor cu, int max_seqlen, float scale, int variant=32) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(vit_fa, CUDA, m) { m.impl("forward", &vit_fa_forward); }
