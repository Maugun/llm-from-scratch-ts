# Module 3 — Bigram model CPU

Ce module construit un modele bigramme a partir d'une sequence d'ids. Un bigramme regarde
uniquement le token courant pour estimer le token suivant.

Il ne fait pas encore de generation complete, de sampling avance, d'embeddings, de tenseurs
ou d'entrainement par gradient.

## Pourquoi ce module existe

Les modules precedents transforment un texte en ids. Ce module commence a apprendre une
structure statistique tres simple: quelles transitions apparaissent souvent dans le corpus.

```text
token courant -> token suivant
```

Par exemple, si le token `b` est souvent suivi par `o`, alors `P(o | b)` devient elevee.

## Pipeline

Ce module arrive apres le tokenizer et le dataset loader:

```text
1. Lire le fichier texte
2. Construire le tokenizer
3. Creer le dataset de token ids
4. Utiliser la partie entrainement du dataset pour compter les bigrammes
5. Normaliser les comptages en probabilites
```

Le modele bigramme ne repart pas du texte brut. Il utilise la sequence d'ids produite par
les modules precedents, plus precisement la partie principale du dataset reservee a
l'apprentissage. Il compte ensuite les paires consecutives:

```text
[id0, id1, id2] -> (id0, id1), (id1, id2)
```

## Concepts

- **Comptage**: nombre d'apparitions d'une transition `current -> next`.
- **Normalisation**: transformation des comptages en probabilites.
- **Probabilite conditionnelle**: probabilite du prochain token sachant le token courant.
- **Prediction**: choix du token suivant ayant la probabilite la plus haute.

La formule centrale est:

```text
P(next | current) = count(current, next) / count(current, *)
```

`count(current, *)` signifie: toutes les transitions qui partent de `current`.

Si plusieurs tokens ont exactement la meme probabilite maximale, `predictMostLikelyNextToken`
retourne le premier dans l'ordre des ids. Cette regle garde la prediction deterministe, mais
elle n'est pas une strategie de generation sophistiquee.

## Exemple

```ts
import { createBigramModel } from './index.js'

const model = createBigramModel([0, 1, 0, 1, 2], 3)

console.info(model.getTransitionCount(0, 1))
console.info(model.getNextTokenProbabilities(0))
console.info(model.predictMostLikelyNextToken(0))
```

Pour lancer une demo executable:

```bash
npm run demo:03-bigram
```

La demo affiche d'abord un exemple avec la lettre `l`, puis, dans un terminal interactif,
elle permet de choisir une autre lettre. Appuie sur `ESC` pour quitter.

## Impact memoire / VRAM

Le modele utilise uniquement des tableaux JavaScript en RAM CPU. Il ne cree aucun tenseur
et n'utilise pas le GPU: la VRAM consommee est donc 0.

La memoire depend de `vocabularySize x vocabularySize`, car on stocke une matrice dense de
comptages et une matrice dense de probabilites. C'est lisible et acceptable avec un petit
vocabulaire caractere, mais ce ne serait pas adapte a un grand vocabulaire moderne.

## Limites

- Le contexte est limite a un seul token.
- Le modele ne comprend pas la semantique.
- Il n'y a pas de parametres appris par gradient.
- Il n'y a pas de smoothing: une transition jamais vue garde une probabilite 0.
- Il n'y a pas encore de sampling ni de generation complete de texte.
