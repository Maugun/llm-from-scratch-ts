# Module 2 — Dataset loader

Ce module lit un fichier `.txt`, encode son contenu avec un tokenizer, puis separe les ids
en deux parties: train et validation.

Il ne construit pas encore de batch, de paire entree/cible, de tenseur ou de modele.

## Pourquoi ce module existe

Un LLM apprend a partir de sequences observees. Avant de parler de probabilites ou de
reseaux de neurones, il faut donc transformer un corpus texte en longue sequence d'ids:

```text
fichier texte -> texte brut -> ids -> train / validation
```

Le module 1 convertissait du texte en nombres. Ce module organise ces nombres comme un
dataset minimal.

## Concepts

- **Fichier texte**: source lisible par un humain, ici un petit `.txt`.
- **Texte brut**: contenu exact du fichier apres lecture UTF-8.
- **Token ids**: sequence numerique produite par le tokenizer.
- **Train split**: partie principale qui servira plus tard a apprendre.
- **Validation split**: petite partie mise de cote pour observer si le modele generalise un peu.

Mathematiquement, le corpus devient une sequence discrete:

```text
x = [id0, id1, id2, id3, ...]
```

Les modules suivants apprendront a exploiter cette sequence, mais ce module se limite a la
preparer.

## Exemple

```ts
import { createCharacterTokenizer, createTokenDataset, loadTextFile } from '../../index.js'

const rawText = await loadTextFile('data/tiny-corpus.txt')
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)

console.info(dataset.totalTokens)
console.info(dataset.trainTokenIds)
console.info(dataset.validationTokenIds)
```

## Mini corpus

Le fichier `data/tiny-corpus.txt` est volontairement court et repetitif. Il n'est pas fait
pour produire un bon modele; il sert a voir clairement comment les motifs textuels deviennent
des motifs dans une sequence d'ids.

## Impact memoire / VRAM

Le loader charge tout le fichier en RAM CPU et garde les ids dans des tableaux JavaScript.
Il ne cree aucun tenseur et n'utilise pas le GPU: la VRAM consommee est donc 0.

Le compromis est assumé: c'est simple et lisible, mais pas adapte a de gros corpus.

## Limites

- Le fichier est lu entierement en memoire.
- Le texte est suppose etre en UTF-8.
- Il n'y a pas de streaming.
- Il n'y a pas encore de batching.
- Il n'y a pas encore de paires `(entree, cible)` pour l'entrainement.
