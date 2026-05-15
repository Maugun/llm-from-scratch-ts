# Module 3 — Bigram model CPU

Ce module construit un modèle bigramme à partir d'une séquence d'ids. Un bigramme regarde
uniquement le token courant pour estimer le token suivant.

Il ne fait pas encore de génération complète, de sampling avancé, d'embeddings, de tenseurs
ou d'entraînement par gradient.

## Pourquoi ce module existe

Les modules précédents transforment un texte en ids. Ce module commence à apprendre une
structure statistique très simple: quelles transitions apparaissent souvent dans le corpus.

```text
token courant -> token suivant
```

Par exemple, si le token `b` est souvent suivi par `o`, alors `P(o | b)` devient élevée.

## Pipeline

Ce module arrive après le tokenizer et le dataset loader:

```text
1. Lire le fichier texte
2. Construire le tokenizer
3. Créer le dataset de token ids
4. Utiliser la partie entraînement du dataset pour compter les bigrammes
5. Normaliser les comptages en probabilités
```

Le modèle bigramme ne repart pas du texte brut. Il utilise la séquence d'ids produite par
les modules précédents, plus précisément la partie principale du dataset réservée à
l'apprentissage. Il compte ensuite les paires consécutives:

```text
[id0, id1, id2] -> (id0, id1), (id1, id2)
```

## Concepts

- **Comptage**: nombre d'apparitions d'une transition `current -> next`.
- **Normalisation**: transformation des comptages en probabilités.
- **Probabilité conditionnelle**: probabilité du prochain token sachant le token courant.
- **Prédiction**: choix du token suivant ayant la probabilité la plus haute.

La formule centrale est:

```text
P(next | current) = count(current, next) / count(current, *)
```

`count(current, *)` signifie: toutes les transitions qui partent de `current`.

Si plusieurs tokens ont exactement la même probabilité maximale, `predictMostLikelyNextToken`
retourne le premier dans l'ordre des ids. Cette règle garde la prédiction déterministe, mais
elle n'est pas une stratégie de génération sophistiquée.

## Exemple

```ts
import { createBigramModel } from './index.js'

const model = createBigramModel([0, 1, 0, 1, 2], 3)

console.info(model.getTransitionCount(0, 1))
console.info(model.getNextTokenProbabilities(0))
console.info(model.predictMostLikelyNextToken(0))
```

Pour lancer une démo exécutable:

```bash
npm run demo:03-bigram
```

La démo affiche d'abord un exemple avec la lettre `l`, puis, dans un terminal interactif,
elle permet de choisir une autre lettre. Appuie sur `ESC` pour quitter.

## Impact mémoire / VRAM

Le modèle utilise uniquement des tableaux JavaScript en RAM CPU. Il ne crée aucun tenseur
et n'utilise pas le GPU: la VRAM consommée est donc 0.

La mémoire dépend de `vocabularySize x vocabularySize`, car on stocke une matrice dense de
comptages et une matrice dense de probabilités. C'est lisible et acceptable avec un petit
vocabulaire caractère, mais ce ne serait pas adapté à un grand vocabulaire moderne.

## Limites

- Le contexte est limité à un seul token.
- Le modèle ne comprend pas la sémantique.
- Il n'y a pas de paramètres appris par gradient.
- Il n'y a pas de smoothing: une transition jamais vue garde une probabilité 0.
- Il n'y a pas encore de sampling ni de génération complète de texte.
