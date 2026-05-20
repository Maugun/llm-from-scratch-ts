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
13. Mémoire, performance et VRAM
14. Mini pipeline end-to-end

## Prérequis

- Node.js 24 LTS
- npm 11

Le fichier `.nvmrc` indique la ligne Node attendue.

## Installation

```bash
npm install
```

## Scripts

```bash
npm run typecheck
npm run build
npm run lint
npm run format:check
npm test
npm run demo:01-tokenizer
npm run demo:02-dataset
npm run demo:03-bigram
npm run demo:04-embeddings
npm run demo:05-attention
npm run demo:06-transformer-block
npm run demo:07-positional-encoding
npm run demo:08-training-loop
```

`npm test` accepte temporairement l'absence de tests. Les tests réels seront ajoutés avec
les modules, quand ils apportent une vérification utile.

Les scripts `demo:*` lancent de petits exemples exécutables pour manipuler chaque module
comme dans un cours pratique.

## Structure

```text
src/
  index.ts
  modules/
    01-tokenizer-simple/
    02-dataset-loader/
    03-bigram-model/
    04-embeddings/
    05-self-attention/
    06-transformer-block/
    07-positional-encoding/
    08-training-loop-cpu/
test/
  01-tokenizer-simple/
  02-dataset-loader/
  03-bigram-model/
  04-embeddings/
  05-self-attention/
  06-transformer-block/
  07-positional-encoding/
  08-training-loop-cpu/
data/
  tiny-corpus.txt
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
