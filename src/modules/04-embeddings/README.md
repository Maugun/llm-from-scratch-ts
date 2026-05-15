# Module 4 — Embeddings CPU

Ce module associe chaque token id a un vecteur de nombres. Il reste volontairement CPU-only:
pas de TensorFlow.js, pas de tenseurs, pas de gradients et pas encore d'entrainement.

## Pourquoi ce module existe

Un id de token est une etiquette arbitraire. Le nombre `12` ne veut pas dire que le token est
naturellement proche du token `13`. Pour donner au modele une representation manipulable, on
remplace donc chaque id par un vecteur:

```text
token id -> ligne dans une matrice d'embeddings -> vecteur
```

## Pipeline

Ce module arrive apres le tokenizer et le dataset loader:

```text
1. Lire le fichier texte
2. Construire le tokenizer
3. Creer le dataset de token ids
4. Creer une table d'embeddings
5. Transformer des ids en vecteurs
```

Le module ne modifie pas encore les vecteurs par apprentissage. Il cree seulement une table
initialisee de facon deterministe pour rendre le mecanisme observable.

## Concepts

- **Embedding**: vecteur associe a un token.
- **Table d'embeddings**: matrice de shape `vocabularySize x embeddingDimension`.
- **Lookup**: operation `embeddingTable[tokenId]`.
- **Sequence embeddee**: sequence d'ids transformee en sequence de vecteurs.
- **Similarite cosinus**: mesure d'angle entre deux vecteurs, utile pour l'inspection.

`embeddingDimension` est le nombre de valeurs dans chaque vecteur. Avec une dimension 4, un
token devient par exemple `[0.01, -0.02, 0.00, 0.03]`. Plus cette dimension est grande, plus
le modele a de place pour representer des nuances, mais plus la table consomme de memoire.
Dans ce module, on garde une petite dimension pour que les vecteurs restent lisibles.

La similarite cosinus aide a comparer deux vecteurs, mais elle ne prouve pas que le modele
comprend leur sens. Dans ce module, les vecteurs sont initialises, pas appris.

Intuitivement, elle regarde si deux vecteurs pointent dans une direction proche:

```text
1    -> meme direction
0    -> directions independantes ou orthogonales
-1   -> directions opposees
```

Son interet ici est surtout pedagogique: elle donne une premiere facon de manipuler les
vecteurs comme des objets geometriques. Plus tard, apres entrainement, des tokens utilises
dans des contextes proches pourraient finir avec des vecteurs plus proches. Dans ce module,
ce n'est pas encore le cas: les valeurs sont seulement initialisees.

## Exemple

```ts
import { createEmbeddingTable } from './index.js'

const table = createEmbeddingTable({
    vocabularySize: 20,
    embeddingDimension: 4,
    seed: 123,
})

console.info(table.getEmbedding(0))
console.info(table.embedSequence([0, 1, 2]))
```

Pour lancer une demo executable:

```bash
npm run demo:04-embeddings
```

La demo affiche d'abord un exemple avec `le`, puis, dans un terminal interactif, elle permet
de saisir une ou plusieurs lettres du vocabulaire. Appuie sur `ENTREE` pour valider et sur
`ESC` pour quitter.

## Impact memoire / VRAM

La table est stockee en RAM CPU dans un `number[][]`. La VRAM consommee est donc 0.

La memoire augmente avec:

```text
vocabularySize x embeddingDimension
```

Avec un petit vocabulaire caractere et une dimension 4 dans la demo, l'impact est minuscule.

## Limites

- Les vecteurs ne sont pas appris.
- Les similarites ne portent pas encore de sens linguistique fiable.
- Il n'y a pas de TensorFlow.js.
- Il n'y a pas de gradients.
- Il n'y a pas encore de self-attention.
