# Patch `attn_gqa_verify` : attention de vérification spéculative groupée (V100, LightOnOCR-3-4B)

## Le problème

Le backend FLASH_ATTN_V100 de 1Cat-vLLM 1.5.1 traite les lots de vérification spéculative (k = 4, donc 5 jetons par requête) par la voie « small-query ». Cette voie déplie chaque jeton vérifié en une ligne de décodage indépendante, avec le noyau `flash_attention_decode_partition_kernel` et une CTA par (jeton, tête q, partition).

Le KV d'une requête est ainsi relu 4 têtes q × 5 jetons = environ 17 fois. Le noyau XQA du fork, qui évite cette relecture, n'est pas utilisé ici (`q_per_kv` ∈ {6, 8} seulement en petit q).

Résultat dans le profil de production du 28/10 (`out/profil_gpu/RAPPORT.md`) : **5,5 ms par couche × 8 couches, soit environ 39 % du temps des noyaux d'un pas de décodage**.

## La solution

Une CTA par (requête, tête KV, partition de 512 clés) :
- elle charge les 4 têtes q du groupe × les jetons de la requête dans un bloc de 32 lignes (ligne = j·4 + g) ;
- elle lit chaque bloc de 32 clés K puis V **une seule fois** (cache paginé, adresses tirées de `block_table`) ;
- `S = Q·Kᵀ` et `P·V` tournent sur **cœurs tenseurs (WMMA fp16 → fp32)** ;
- la softmax est en ligne, en fp32, avec le masque causal propre à chaque jeton spéculatif.

Une passe Triton combine ensuite les partitions (split-K).

Sémantique reprise de `_flash_v100_small_query_prefill_as_decode` :
- jetons de la requête r : `[qsl[r], qsl[r+1])` ;
- `eff = max(seq_lens[r], q_len)` ;
- le jeton j voit les clés `[0, eff − q_len + j + 1)` ;
- échelle `self.scale` (1/16), softcap optionnel ;
- les jetons hors de toute requête (remplissage de graphe CUDA) reçoivent une sortie nulle ;
- cache `kv_cache[num_blocks, 2, 544, 4, 256]` lu par `[:, 0]` et `[:, 1]` avec ses pas réels. La taille de bloc est de **544** : le serveur l'annonce dans son log, « Setting attention block size to 544 tokens ».

Pourquoi CUDA et pas seulement Triton : sur sm_70, Triton 3.6 compile `tl.dot` en FMA (aucun `mma.sync` dans le PTX). La variante Triton (`gqa_verify.py`, backend `triton`) est correcte mais ne gagne que 1,1 à 1,2×.

## Fichiers

| Fichier | Rôle |
|---|---|
| `attn_gqa_verify.py` | Module de patch pour le chargeur `/ext/sitecustomize.py` : `CIBLE`, `appliquer(module)` |
| `gqa_verify.py` | Lanceur, espace de travail persistant, réduction Triton, variante Triton complète |
| `csrc/gqa_verify_cuda.cu` | Passe « partition » CUDA WMMA (op `torch.ops.gqa_verify.partition`) |
| `construire.py` | Compile `build/gqa_verify_cuda.so` dans l'image de production |
| `build/gqa_verify_cuda.so` | Extension compilée pour `ocr/1cat-vllm:1.5.1` (torch 2.10, CUDA 12.8, sm_70) |
| `test_gqa_verify.py` | Test du noyau aux formes de production : référence fp32, noyau du fork, cas limites, chronos |
| `test_patch.py` | Test du branchement : aiguillage, replis, graphe CUDA, lot mixte, vrai module du fork |
| `banc_parties.py` | Banc : passe partition seule, réduction seule, appel complet |

## Ce que le patch remplace

Dans `FlashAttnV100Impl` :

1. `_flash_v100_small_query_prefill_as_decode` : lots uniformes de vérification, **y compris en capture et rejeu de graphe CUDA**.
   - Il lit les tampons persistants du fork (`smallq_query_start_loc`, `smallq_decode_block_table`, `smallq_decode_seq_lens`).
   - À défaut, en eager, il lit `attn_metadata.query_start_loc`, `block_table` et `seq_lens`.
   - Pas de `.item()`, pas d'allocation par appel : l'espace de travail est alloué au premier appel (échauffement eager avant la capture), puis réutilisé.
2. `_run_prefill_prefix_decode_rows` : lignes petit q des lots mixtes (prefill + décodage), en eager. Même logique de sélection des lignes que l'original.

**Repli explicite vers la méthode d'origine** dans les cas suivants :
- groupe ≠ 4 ;
- `head_dim` ≠ 256 ;
- `smallq_decode_max_query_len` > 8 ;
- q, k ou v pas en fp16, ou `kv_cache_dtype` ≠ auto (donc fp8 exclu) ;
- fenêtre glissante ou fenêtre ancrée, alibi, sinks ;
- non causal, DFlash ;
- échelles k/v ≠ 1 ;
- capture de graphe sans métadonnées persistantes ;
- extension `.so` absente ;
- `GQA_VERIFY=0`.

Chaque raison de repli est journalisée une fois : « attn_gqa_verify : repli … ».

## Activation

1. Copier dans `scripts/vllm_ext/patchs/` (monté en `/ext/patchs/`) :
   - `attn_gqa_verify.py` ;
   - `gqa_verify.py` ;
   - **`build/gqa_verify_cuda.so`**, en gardant le sous-répertoire `build/`, ou bien régler `GQA_VERIFY_SO=/chemin/gqa_verify_cuda.so`.
2. `cache/1cat/patchs.json` (vu dans le conteneur comme `/root/.cache/patchs.json`) : `{"actifs": ["attn_gqa_verify"]}`.
3. Redémarrer le serveur. Dans `docker logs ocr-serveur`, on doit voir :
   - `attn_gqa_verify : patch appliqué (cuda=True)` (une fois par processus) ;
   - puis, au premier lot de vérification : `attn_gqa_verify : voie de vérification groupée active (G=4, backend=auto, cuda=True, par_jeton=True)`.

Variables d'environnement, toutes optionnelles :

| Variable | Effet |
|---|---|
| `GQA_VERIFY=0` | Coupe le patch (repli complet) |
| `GQA_VERIFY_BACKEND=auto\|cuda\|triton` | `auto` : CUDA si l'extension est chargée, sinon voie d'origine |
| `GQA_VERIFY_MIXTE=0` | Ne pas toucher aux lots mixtes |
| `GQA_VERIFY_PS=512` | Taille de partition (multiple de 32). 256 et 512 sont équivalents ; 1024 est plus lent |

Recompiler l'extension (si l'image change), avec nice -n 15 et 2 tâches nvcc au plus :
```
cd engines/1cat/patchs
docker run --rm --cpus 3 --entrypoint /usr/bin/nice -v $PWD:/patchs ocr/1cat-vllm:1.5.1 \
  -n 15 /opt/venv/bin/python /patchs/construire.py
```

Relancer les tests dans un conteneur séparé, jamais dans `ocr-serveur`. Il faut environ 1 Go de GPU.
```
docker run --rm --cpus 3 --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 --entrypoint /usr/bin/nice \
  -v $PWD:/patchs:ro ocr/1cat-vllm:1.5.1 -n 15 /opt/venv/bin/python /patchs/test_gqa_verify.py --fork --backend cuda
docker run ... /opt/venv/bin/python /patchs/test_patch.py --vrai
```

## Résultats des tests (28/10, 12:37 UTC)

Conditions : V100 partagé avec la production en cours. Le noyau du fork y prend 10 à 13 ms, contre 5,5 ms dans la trace de production. Les temps absolus sont donc gonflés d'environ 2×, mais le **rapport** entre les deux noyaux est comparable. Les sorties brutes sont dans `out/profil_gpu/tests/test_cuda.json` et `test_triton.json`.

| Cas (page 544, cache non contigu comme en production) | Écart max / moyen vs fp32 | Écart max du fork | Temps nouveau / fork | Gain |
|---|---|---|---|---|
| Production : 48 req. × 5 jetons, contextes 1 100–2 500 | 1,1e-4 / 8e-6 | 1,4e-4 | 2,0–2,5 ms / 10,3–12,8 ms | **×5,1 à ×6,2** |
| 48 req. dont 8 à 1 jeton (prefill fini) | 1,2e-4 / 8e-6 | 1,7e-4 | 1,4 / 10,8 ms | ×7,6 |
| 32 req. × 5 (taille de graphe 160) | 8e-5 / 8e-6 | 1,0e-4 | 1,5 / 7,9 ms | ×5,3 |
| Contextes courts (2 à 40 clés) | 1,0e-3 / 8e-5 (valeurs jusqu'à 3,2 : ≈ 1 ulp) | 9,7e-4 | – | – |
| Frontières de page (543, 544, 545, 1 088…) | 1,4e-4 | 2,0e-4 | – | – |
| Frontières de partition (512, 513, 515, 1 026) | 1,4e-4 | 1,7e-4 | – | – |
| Contexte long (9 000 et 7 777) | 3,4e-5 | 3,3e-5 | – | – |
| Remplissage de graphe (+12 jetons hors requête) | 7,4e-5 ; sortie de remplissage nulle : oui | 9,4e-5 | – | – |
| Requête à 0 jeton au milieu du lot | 1,1e-4 | 1,4e-4 | – | – |

Banc alterné, plus fiable sous contention (`banc_parties.py` et `out/profil_gpu/v1/banc_v1v2.py`). On a fait 10 tours alternés de 10 appels chacun, au format de production :

| Noyau | Médiane | Minimum |
|---|---:|---:|
| Nouveau noyau | 2,05 ms | 1,18 ms |
| Noyau du fork | 12,5 ms | 8,8 ms |
| **Gain** | **×6,1** | **×7,5** |

La réduction Triton ne coûte que 0,04 ms. Une variante v2 (K et V en double tampon, Q·Kᵀ sur 8 warps, 95 Ko de mémoire partagée) n'a rien gagné (médiane 2,04 ms) : elle a été écartée, et c'est v1 qui est livrée.

Autres constats :
- **Par rapport au noyau actuel**, l'écart max est de 2,4e-4 (environ 1 ulp fp16 sur ces magnitudes) et 51 % des valeurs sont identiques au bit près. Ce n'est donc pas identique à l'octet, et il faut valider sur les jeux de qualité. Le nouveau noyau est légèrement plus proche de la référence fp32 que l'ancien.
- La voie « métadonnées par jeton » (tampons persistants du fork) donne une sortie identique au bit près à la voie « par requête », dans tous les cas.
- `test_patch.py --vrai` : **15/15 OK**.
  - Aiguillage : forme gérée → noyau. Groupe 8, q > 8, kv fp8 ou `GQA_VERIFY=0` → voie d'origine.
  - Capture en graphe CUDA puis rejeu après changement du **contenu** des métadonnées persistantes (nouvelles longueurs, nouvelles pages, requêtes à 0 jeton) : égal à la référence ; jetons de remplissage nuls.
  - Lot mixte (2 requêtes en prefill, 4 en petit q) : lignes {1, 2, 4, 5} égales à la référence, lignes de prefill intactes.
  - Vrai module du fork : les deux méthodes sont remplacées, les signatures d'origine sont celles attendues, et `_is_cuda_graph_capturing` et `_record_route` sont présents.

**Gain attendu en production** : hors contention, environ 1 ms par couche au lieu de 5,5, soit environ −35 ms par pas de décodage (sur environ 115 ms de noyaux). D'après le bilan du rapport, cela fait **+15 à +19 % de débit**. À confirmer par une capture de profil après redémarrage : `flash_attention_decode_partition_kernel` doit disparaître des pas de vérification, et le noyau `gqa_partition_kernel` doit apparaître.

## Limites connues

- **Formes gérées** : groupe = 4, `head_dim` = 256, KV fp16, ≤ 8 jetons par requête. Tout le reste retombe sur la voie d'origine.
- **Pas identique à l'octet** au noyau actuel : environ 1 ulp fp16 sur la moitié des valeurs. Il faut valider sur g9/ech9093, avec identité du texte OCR (au niveau actuel, environ 62/64) ou CER égal.
- **Pas encore essayé dans le vrai serveur.** Seuls le vrai module du fork, la capture de graphe et les formes de production ont été testés hors serveur. Au premier démarrage, il faut surveiller les logs (capture des graphes de 5 à 240 jetons) et la mémoire.
- **Espace de travail** : partiels fp32 [256 jetons × 16 têtes × 20 partitions × 256] ≈ 84 Mo, plus 0,6 Mo. Ils sont alloués au premier appel par l'allocateur torch du moteur, donc hors du budget `gpu-memory-utilization` ; cela passe largement à 0,90. Un lot mixte avec plus de 256 jetons petit q allouerait un tampon plus grand (en eager seulement).
- **Lots mixtes** : quelques petits tenseurs hôte→GPU et un `.tolist()` par appel, comme l'original. La réduction Triton se recompile à la première apparition de chaque nouvelle puissance de 2 du nombre de lignes, en environ 1 s, une fois.
- **Softcap** : implémenté (cap·tanh) mais non testé, car le modèle n'en a pas.
- **Occupation** : 1 CTA par SM (75 Ko de mémoire partagée, 8 warps), environ 340 Go/s utiles estimés hors contention. Une version à 2 CTA par SM ou à double tampon pourrait encore diviser le temps par environ 2 (environ −4 ms par pas, +2 à 3 %).
- **Variante Triton** (`GQA_VERIFY_BACKEND=triton`) : correcte mais seulement 1,1 à 1,2× plus rapide (FMA sur sm_70). Elle n'est jamais choisie automatiquement.
- **Le `.so` dépend de l'image** `ocr/1cat-vllm:1.5.1`. Il faut le recompiler avec `construire.py` si l'image change.

---

# Patch `vit_attn_fa` : attention de la tour vision (FlashAttention WMMA sm_70, d = 64)

## Ce qu'il fait

`MMEncoderAttention._forward_sdpa` (module `vllm.model_executor.layers.attention.mm_encoder_attention`) est le chemin TORCH_SDPA de la tour vision Qwen3-VL de LightOnOCR-3-4B : 24 couches, 16 têtes, d = 64, non causale.

Le patch la remplace par le noyau `csrc/vit_fa_cuda.cu` (op `torch.ops.vit_fa.forward`, `build/vit_fa_cuda.so`). C'est une FlashAttention avant sur cœurs tenseurs :
- une CTA par (64 requêtes, tête, séquence), 4 warps × 16 lignes, blocs de 32 clés, 4 CTA par SM ;
- S passe par la mémoire partagée, la softmax est en ligne fp32 avec base 2 ;
- l'accumulateur reste dans les fragments WMMA, avec une **remise à l'échelle paresseuse exacte** : elle n'a lieu que si le max d'une ligne dépasse de 8 (en log2) le max de référence, et P = 2^(s − m_ref) ≤ 256.

**Second gain** : l'original fait `cu_seqlens.tolist()` sur un tenseur GPU dans chaque couche, donc 24 synchronisations par image. Le profil de 13:30 en compte environ 10 ms de trous par pas mixte. Le patch n'en fait aucune : la grille est bornée par la longueur totale, connue de l'hôte, et chaque CTA lit les bornes de sa séquence sur le GPU.

Repli vers l'original dans les cas suivants :
- `VIT_FA=0` ;
- extension absente ;
- dtype autre que fp16 ;
- `head_dim` ≠ 64 ou GQA ;
- lot > 1 sans `cu_seqlens`, ou q et kv de longueurs différentes ;
- `cu_seqlens` qui n'est pas en int32 sur le GPU.

## Activation

1. Copier `vit_attn_fa.py` et `build/vit_fa_cuda.so` (dans `build/`, ou via `VIT_FA_SO`) dans `scripts/vllm_ext/patchs/`.
2. Ajouter `"vit_attn_fa"` à `actifs`.

Au démarrage, le log doit montrer « vit_attn_fa : patch appliqué (extension=True) », puis au premier prefill « attention vision par le noyau FA WMMA sm70 ».

`VIT_FA_VARIANTE=64` choisit les blocs de 64 clés (2 CTA par SM). Cette variante est moins bonne.

## Résultats (28/10, vers 13:50 UTC, GPU partagé avec la production)

Les sorties brutes sont dans `out/profil_gpu/tests/test_vit_fa_v3_32.json`.

| Cas | Écart max / moyen vs fp32 | SDPA actuelle (max / moyen) | Nouveau vs actuel |
|---|---|---|---|
| Colonne de 5 888 patchs | 9,5e-4 / 1,7e-5 | 9,2e-4 / 1,5e-5 | max 9,8e-4 (1 ulp fp16), 55 % identiques |
| Bandeau de 256 | 1,4e-3 / 5,1e-5 | 9,7e-4 / 4,3e-5 | max 2,0e-3 (1 ulp sur des valeurs ≈ 2,5) |
| 5 900 (dernier bloc partiel), 77, 1 | ≤ 1,2e-3 | ≤ 9,7e-4 | ≤ 1 ulp |
| 2 images (5 888 + 1 012), 3 images (64 + 65 + 130) | ≤ 1,6e-3 | ≤ 1,0e-3 | ≤ 1 ulp, pas de fuite entre images |

**Vitesse** : banc alterné `banc_vit.py`, 15 tours, colonne de 5 888 patchs.

| Noyau | Meilleur temps | Médiane |
|---|---:|---:|
| SDPA actuelle | 5,31 ms (26,8 TFLOPS) | 10,7 ms |
| `vit_fa` variante 32 | **4,54 ms** (31,3 TFLOPS) | **8,65 ms** |
| `vit_fa` variante 64 | 5,11 ms | 10,4 ms |

Gain du noyau : **×1,17** au meilleur temps, **×1,23** en médiane. La SDPA au meilleur temps (5,3 ms) correspond à la trace de production (5,2 ms par appel pour une colonne).

`test_vit_patch.py`, sur le vrai module du fork : **10/10 OK**.
- La méthode patchée est égale à l'originale (`__wrapped__`) à 1 ulp près, pour une colonne de 5 888, 2 images, un `cu_seqlens` rembourré de séquences vides, et une entrée 4D sans `cu_seqlens`.
- Replis vérifiés : d = 80, fp32, lot de 2, `VIT_FA=0`.

**Gain attendu en production** : l'attention vision pèse 12,6 % du GPU, et environ 10 ms de trous par pas mixte disparaissent. Cela donne environ **+3 à +4 % de tuiles/s**. C'est à valider par `scripts/valider_ocr.sh` : pas identique à l'octet, à 1 ulp près sur la sortie de la tour vision.

## Limites

- **Gain modeste** sur le noyau lui-même (×1,2). Les allers-retours de S et P par la mémoire partagée, imposés par WMMA, coûtent cher. Il faudrait écrire `mma.sync` m8n8k4 directement, avec la softmax en registres, pour viser ×2 : c'est un travail de plusieurs jours.
- Pas de GQA, d = 64 seulement, fp16 seulement.
- Les séquences multiples sont surdimensionnées en grille (nseq × ⌈S/64⌉ CTA, celles hors séquence sortent aussitôt) : c'est négligeable pour 1 ou 2 images.

---

# Mise à jour d'`attn_gqa_verify` (28/10, 13:55) : lots mixtes sans copies synchrones

Avant, dans les pas mixtes, chacune des 8 couches reconstruisait les métadonnées compactes des lignes petit q, avec 4 `torch.tensor(..., device=cuda)`, donc des copies H2D synchrones. Le profil de 13:30 y attribuait `index_select` 2,9 ms et une partie des `aten::to`/`copy_`.

Désormais, ces métadonnées sont construites **une fois par pas**, à la première couche, et copiées depuis de la mémoire épinglée en `non_blocking`.

Le cache est indexé par l'objet `attn_metadata`, avec un garde-fou : il n'est réutilisé que si `query_start_loc` et `seq_lens` sont identiques. Le test `test_patch.py` couvre ce cas (c') : 8 couches de suite avec le même objet, puis un objet réutilisé avec des `seq_lens` changés, qui doit être reconstruit ; les deux sont OK.

**À recopier dans `scripts/vllm_ext/patchs/`** : la version de production est celle de 12:47.
