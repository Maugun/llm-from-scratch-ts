# TypeScript LLM

Projet pédagogique en TypeScript pour comprendre progressivement comment fonctionnent les
LLM modernes.

Le projet part de briques très simples, comme un tokenizer caractère et un modèle bigramme,
puis avance jusqu’à un petit Transformer entraînable avec TensorFlow.js, tokenizer BPE,
checkpoints et playground de génération.

Ce projet ne vise pas la production ni les performances maximales. Le but est d’apprendre,
de manipuler les concepts et de garder le code lisible.

## Ce que tu vas apprendre

- Transformer du texte en tokens numériques.
- Préparer un corpus et des exemples `contexte -> prochain token`.
- Comprendre les embeddings, la self-attention, les blocs Transformer et le positional encoding.
- Observer une boucle d’entraînement, une loss, une perplexité et des gradients.
- Générer du texte avec greedy decoding, température et top-k.
- Préparer un corpus long local, entraîner un petit modèle et sauvegarder des checkpoints.

## Prérequis

- Node.js 20 LTS ou 24 LTS.
- npm 10 ou 11.

Le fichier `.nvmrc` indique Node 20, car les bindings natifs TensorFlow.js GPU sont plus fiables
sur une version LTS largement supportée. Node 24 reste utilisable pour les modules sans backend
natif GPU, mais Node 20 est recommandé pour `@tensorflow/tfjs-node-gpu`.

## Installation

```bash
npm install
```

Vérifier que le projet est sain:

```bash
npm run typecheck
npm test
npm run lint
npm run format:check
npm run build
```

## Démarrage rapide

Lancer les premiers modules:

```bash
npm run demo:01-tokenizer
npm run demo:02-dataset
npm run demo:03-bigram
```

Lancer le mini Transformer pédagogique:

```bash
npm run demo:14-mini-transformer
```

Lancer la démo finale courte:

```bash
npm run demo:19-final-llm
```

Cette dernière commande utilise une config mini versionnée et le petit corpus `data/tiny-corpus.txt`.
Elle est faite pour tester le flux complet rapidement, pas pour obtenir un bon modèle.

## Parcours pédagogique

1. `01-tokenizer-simple`: tokenizer caractère.
2. `02-dataset-loader`: lecture d’un corpus texte et split train/validation.
3. `03-bigram-model`: probabilités simples `token courant -> token suivant`.
4. `04-embeddings`: ids de tokens transformés en vecteurs.
5. `05-self-attention`: self-attention causale CPU.
6. `06-transformer-block`: bloc Transformer CPU.
7. `07-positional-encoding`: ajout d’information de position.
8. `08-training-loop-cpu`: première boucle d’entraînement CPU.
9. `09-minimal-trainable-language-model`: modèle contextuel entraînable fait main.
10. `10-text-generation`: génération greedy.
11. `11-sampling-strategies`: température et top-k.
12. `12-tensorflow-autograd`: introduction à TensorFlow.js et autograd.
13. `13-tfjs-next-token-model`: modèle next-token TensorFlow.js.
14. `14-trainable-mini-transformer`: mini Transformer entraînable.
15. `15-model-sizing-memory-estimator`: estimation paramètres/mémoire.
16. `16-tfjs-node-gpu-backend`: backend GPU optionnel.
17. `17-long-corpus-pipeline`: préparation d’un corpus long privé.
18. `18-small-real-model-training`: entraînement batché sur corpus long.
19. `19-final-tiny-llm`: tiny LLM final avec BPE, checkpoints et playground.

Chaque module contient son propre README avec les explications détaillées.

## Commandes utiles

Qualité du projet:

```bash
npm run typecheck
npm test
npm run lint
npm run format:check
npm run build
```

Démos:

```bash
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
npm run demo:19-final-llm
```

Module final:

```bash
npm run llm:train -- --config data/private/final-llm-config.json
npm run llm:chat -- --config data/private/final-llm-config.json
npm run llm:generate -- --config data/private/final-llm-config.json --prompt "Utilisateur: Bonjour\nAssistant:"
```

`llm:train` continue automatiquement depuis le dernier checkpoint compatible. Pour repartir de
zéro dans une nouvelle version:

```bash
npm run llm:train -- --config data/private/final-llm-config.json --force-train
```

## Utiliser un corpus long privé

Les corpus longs ne sont pas versionnés. Place ton fichier local dans `data/private/`.

Exemple:

```text
data/private/long-corpus.txt
```

Nettoyer un texte extrait d’un livre ou d’un PDF:

```bash
npm run corpus:clean -- --path data/private/long-corpus.txt --keep-paragraphs
```

Par défaut, le résultat est écrit à côté du fichier source avec `.clean` avant l’extension:

```text
data/private/long-corpus.clean.txt
```

Options utiles:

```bash
npm run corpus:clean -- --path data/private/long-corpus.txt --fix-hyphenation
npm run corpus:clean -- --path data/private/long-corpus.txt --output data/private/corpus.clean.txt
```

Les dossiers suivants sont ignorés par Git:

```text
data/private/
data/cache/
data/checkpoints/
```

## Backend GPU optionnel

Le projet fonctionne avec `@tensorflow/tfjs` par défaut.

Le backend `@tensorflow/tfjs-node-gpu` est optionnel, car il dépend de CUDA/Linux et ne convient
pas à toutes les machines. Il est surtout utile pour les modules avancés et les entraînements plus
longs.

Dans un environnement compatible, typiquement WSL2/Linux avec CUDA:

```bash
npm run gpu:install
npm run demo:16-tfjs-node-gpu
```

Cette installation se fait avec `--no-save`: le package GPU n’est pas ajouté aux dépendances
versionnées du projet.

## Configs d’exemple

Le module final fournit plusieurs configs:

```text
src/modules/19-final-tiny-llm/demo-config.mini.example.json
src/modules/19-final-tiny-llm/demo-config.example.json
src/modules/19-final-tiny-llm/demo-config.gpu.example.json
```

Pour entraîner sur ton corpus privé, copie une config dans `data/private/`, adapte les chemins et
les dimensions, puis lance `llm:train`.

## Structure

```text
src/
  index.ts
  tools/
    corpus-cleaner.ts
  modules/
    01-tokenizer-simple/
    ...
    19-final-tiny-llm/
test/
  01-tokenizer-simple/
  ...
  19-final-tiny-llm/
data/
  tiny-corpus.txt
  private/      # ignoré par Git
  cache/        # ignoré par Git
  checkpoints/  # ignoré par Git
```

## Limites

- Les modèles entraînés ici sont très petits.
- Le mode chat n’est pas un assistant fiable.
- Le projet n’inclut pas d’instruction tuning, de RAG ou de garde-fous de production.
- Les résultats dépendent fortement du corpus, de la taille du modèle et du temps d’entraînement.
- Un modèle entraîné sur un seul livre apprend surtout du style et des motifs, pas une compréhension
  robuste du contenu.
