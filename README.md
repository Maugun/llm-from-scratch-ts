# TypeScript LLM

Mini LLM pédagogique en TypeScript pour comprendre progressivement le fonctionnement des
LLM modernes. Le but principal est l'apprentissage: le code doit rester simple, lisible,
fortement commenté et vérifiable module par module.

Ce projet ne vise pas la performance ni un usage production.

## Principes

- Avancer un seul module à la fois.
- Présenter le plan du module courant avant d'écrire son code.
- Attendre une validation explicite avant de passer au module suivant.
- Expliquer le rôle théorique, les intuitions mathématiques, les compromis et les limites.
- Garder une attention explicite sur la mémoire et la VRAM.
- Éviter les dépendances inutiles.

## Roadmap pédagogique

1. Tokenizer simple
2. Dataset loader
3. Bigram model
4. Embeddings
5. Self-attention
6. Transformer block
7. Positional encoding
8. Training loop CPU pédagogique
9. Modèle de langage minimal entraînable
10. Text generation
11. Sampling strategies
12. TensorFlow.js / autograd
13. Modèle next-token TensorFlow.js
14. Mini Transformer entraînable + génération greedy
15. Estimation mémoire et taille de modèle
16. Backend GPU @tensorflow/tfjs-node-gpu
17. Pipeline long corpus
18. Entraînement d'un petit vrai modèle
19. Tiny LLM final avec BPE, long corpus et chat playground

## Prérequis

- Node.js 20 LTS ou 24 LTS
- npm 10 ou 11

Le fichier `.nvmrc` indique Node 20, car les bindings natifs TensorFlow.js GPU sont plus fiables
sur une version LTS largement supportée. Node 24 reste utilisable pour les modules sans backend
natif GPU, mais Node 20 est recommandé pour `@tensorflow/tfjs-node-gpu`.

## Installation

```bash
npm install
```

Le module 16 utilise un backend GPU optionnel. `@tensorflow/tfjs-node-gpu` n’est pas installé par
défaut, car il dépend de CUDA/Linux et ne convient pas à toutes les machines. Pour ce module,
installe-le uniquement dans un environnement compatible, depuis WSL ou Linux:

```bash
npm run gpu:install
```

Cette commande installe le backend GPU localement sans l’ajouter aux dépendances versionnées du
projet. Si `node_modules` est nettoyé ou réinstallé côté Windows, il faudra relancer cette commande
côté WSL/Linux avant d’utiliser le backend GPU.

## Scripts

```bash
npm run typecheck
npm run build
npm run lint
npm run format:check
npm test
npm run corpus:clean -- --path data/private/long-corpus.txt
npm run demo:01-tokenizer
npm run demo:02-dataset
npm run demo:03-bigram
npm run demo:04-embeddings
npm run demo:05-attention
npm run demo:06-transformer-block
npm run demo:07-positional-encoding
npm run demo:08-training-loop
npm run demo:09-minimal-lm
npm run demo:10-text-generation
npm run demo:11-sampling
npm run demo:12-tfjs-autograd
npm run demo:13-tfjs-next-token
npm run demo:14-mini-transformer
npm run demo:15-model-sizing
npm run demo:16-tfjs-node-gpu
npm run demo:17-long-corpus
npm run demo:18-small-real-model
npm run demo:18-small-real-model:continue
npm run demo:18-small-real-model:train
npm run demo:19-final-llm
npm run llm:train -- --config data/private/final-llm-config.json
npm run llm:chat -- --config data/private/final-llm-config.json
npm run llm:generate -- --config data/private/final-llm-config.json --prompt "Utilisateur: Bonjour\nAssistant:"
npm run gpu:install
npm run gpu:demo
```

`npm test` accepte temporairement l'absence de tests. Les tests réels seront ajoutés avec
les modules, quand ils apportent une vérification utile.

Les scripts `demo:*` lancent de petits exemples exécutables pour manipuler chaque module
comme dans un cours pratique.

Le script `corpus:clean` prépare un texte extrait d’un livre ou d’un PDF avant entraînement:

```bash
npm run corpus:clean -- --path data/private/long-corpus.txt
```

Par défaut, il écrit le résultat à côté du fichier source en ajoutant `.clean` avant l’extension:

```text
data/private/long-corpus.clean.txt
```

Il remplace les retours à la ligne simples par des espaces, réduit les espaces multiples et nettoie
les espaces avant la ponctuation. Options utiles:

```bash
npm run corpus:clean -- --path data/private/long-corpus.txt --keep-paragraphs
npm run corpus:clean -- --path data/private/long-corpus.txt --fix-hyphenation
npm run corpus:clean -- --path data/private/long-corpus.txt --output data/private/corpus.clean.txt
```

## Structure

```text
src/
  index.ts
  tools/
    corpus-cleaner.ts
  modules/
    01-tokenizer-simple/
    02-dataset-loader/
    03-bigram-model/
    04-embeddings/
    05-self-attention/
    06-transformer-block/
    07-positional-encoding/
    08-training-loop-cpu/
    09-minimal-trainable-language-model/
    10-text-generation/
    11-sampling-strategies/
    12-tensorflow-autograd/
    13-tfjs-next-token-model/
    14-trainable-mini-transformer/
    15-model-sizing-memory-estimator/
    16-tfjs-node-gpu-backend/
    17-long-corpus-pipeline/
    18-small-real-model-training/
    19-final-tiny-llm/
test/
  01-tokenizer-simple/
  02-dataset-loader/
  03-bigram-model/
  04-embeddings/
  05-self-attention/
  06-transformer-block/
  07-positional-encoding/
  08-training-loop-cpu/
  09-minimal-trainable-language-model/
  10-text-generation/
  11-sampling-strategies/
  12-tensorflow-autograd/
  13-tfjs-next-token-model/
  14-trainable-mini-transformer/
  15-model-sizing-memory-estimator/
  16-tfjs-node-gpu-backend/
  17-long-corpus-pipeline/
  18-small-real-model-training/
  19-final-tiny-llm/
data/
  tiny-corpus.txt
  private/      # ignoré par Git, pour les corpus locaux
  cache/        # ignoré par Git, pour les datasets préparés localement
  checkpoints/  # ignoré par Git, pour les modèles sauvegardés localement
```

Les modules sont préfixés par numéro pour rendre l'ordre pédagogique visible dans
l'explorateur de fichiers.

## Workflow pour chaque module

Avant de coder un module, le plan doit couvrir:

1. le but théorique du module;
2. sa place dans un LLM;
3. les concepts mathématiques utiles, expliqués simplement;
4. l'architecture des fichiers;
5. l'impact mémoire et VRAM;
6. les compromis;
7. les tests simples prévus.

Après implémentation, le module doit inclure des explications, des commentaires
pédagogiques et une vérification locale adaptée.
