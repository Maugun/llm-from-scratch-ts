# TypeScript LLM

Mini LLM pedagogique en TypeScript pour comprendre progressivement le fonctionnement des
LLM modernes. Le but principal est l'apprentissage: le code doit rester simple, lisible,
fortement commente et verifiable module par module.

Ce projet ne vise pas la performance ni un usage production.

## Principes

- Avancer un seul module a la fois.
- Presenter le plan du module courant avant d'ecrire son code.
- Attendre une validation explicite avant de passer au module suivant.
- Expliquer le role theorique, les intuitions mathematiques, les compromis et les limites.
- Garder une attention explicite sur la memoire et la VRAM.
- Eviter les dependances inutiles.

## Roadmap pedagogique

1. Tokenizer simple
2. Dataset loader
3. Bigram model
4. Embeddings
5. Self-attention
6. Transformer block
7. Training loop
8. Text generation
9. Sampling strategies
10. Optimisations VRAM
11. Tool calling simple
12. MCP connectors, optionnel

## Prerequis

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
```

`npm test` accepte temporairement l'absence de tests. Les tests reels seront ajoutes avec
les modules, quand ils apportent une verification utile.

Les scripts `demo:*` lancent de petits exemples executables pour manipuler chaque module
comme dans un cours pratique.

## Structure

```text
src/
  index.ts
  modules/
    01-tokenizer-simple/
    02-dataset-loader/
test/
  01-tokenizer-simple/
  02-dataset-loader/
data/
  tiny-corpus.txt
```

Les modules sont prefixes par numero pour rendre l'ordre pedagogique visible dans
l'explorateur de fichiers.

## Workflow pour chaque module

Avant de coder un module, le plan doit couvrir:

1. le but theorique du module;
2. sa place dans un LLM;
3. les concepts mathematiques utiles, expliques simplement;
4. l'architecture des fichiers;
5. l'impact memoire et VRAM;
6. les compromis;
7. les tests simples prevus.

Apres implementation, le module doit inclure des explications, des commentaires
pedagogiques et une verification locale adaptee.
