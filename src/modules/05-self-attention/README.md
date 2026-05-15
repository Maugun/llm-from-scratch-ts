# Module 5 — Self-attention causale CPU

Ce module transforme une sequence de vecteurs en nouveaux vecteurs contextualises. Un
embedding isole represente un token seul; la self-attention permet a chaque position de
regarder les positions precedentes pour melanger de l'information de contexte.

Il reste volontairement CPU-only: pas de TensorFlow.js, pas de tenseurs, pas de gradients,
pas de multi-head attention et pas encore de transformer block complet.

## Pourquoi ce module existe

Dans un LLM, le sens d'un token depend fortement de son contexte. La lettre ou le mot courant
ne suffit pas toujours: on veut savoir ce qui vient avant. La self-attention repond a cette
question:

```text
Pour chaque position, quelles positions precedentes sont utiles ?
```

Elle produit alors un nouveau vecteur pour chaque position, construit comme une somme
ponderee des informations accessibles.

## Pourquoi parler de positions ?

Jusqu'ici, on manipulait surtout des sequences:

```text
[id0, id1, id2]
```

ou:

```text
[embedding0, embedding1, embedding2]
```

Mais dans un LLM, chaque element de la sequence occupe une **position**:

```text
position 0 -> premier token
position 1 -> deuxieme token
position 2 -> troisieme token
```

La self-attention calcule une nouvelle representation **pour chaque position**. La position
1 ne remplace pas la position 0: elle produit son propre vecteur contextualise en regardant
les positions autorisees.

Avec un masque causal:

```text
position 0 peut regarder: 0
position 1 peut regarder: 0, 1
position 2 peut regarder: 0, 1, 2
```

Cette contrainte est essentielle pour un LLM autoregressif: quand le modele predit le token
suivant, il ne doit pas tricher en regardant les tokens futurs.

Important: dans ce module, on ne cree pas encore de **positional encoding**. On utilise les
positions pour appliquer le masque causal et organiser les calculs d'attention. L'ajout
d'une information de position dans les vecteurs eux-memes pourra venir plus tard, quand on
assemblera un bloc Transformer plus complet.

## Pipeline

Ce module arrive apres les embeddings:

```text
1. Lire le fichier texte
2. Construire le tokenizer
3. Creer le dataset de token ids
4. Transformer les ids en embeddings
5. Appliquer la self-attention causale sur ces vecteurs
```

## Concepts

- **Query (Q)**: ce que la position cherche.
- **Key (K)**: ce que chaque position annonce pour etre retrouvee.
- **Value (V)**: l'information qui sera vraiment copiee ou melangee.
- **Score d'attention**: compatibilite entre une query et une key.
- **Softmax**: transformation des scores en poids qui somment a 1.
- **Masque causal**: interdiction de regarder les positions futures.

La formule centrale est:

```text
score(i, j) = dot(query_i, key_j) / sqrt(attentionDimension)
weights_i = softmax(scores_i)
output_i = somme_j weights_i[j] * value_j
```

La division par `sqrt(attentionDimension)` evite que les scores deviennent trop grands quand
les vecteurs ont beaucoup de dimensions. Le softmax resterait mathematiquement valide, mais
il deviendrait souvent trop extreme.

## Q, K, V avec une analogie dev

Une bonne analogie est un petit moteur de recherche interne:

```text
Query = la requete de recherche
Key   = l'index ou les metadonnees qui permettent de retrouver une entree
Value = le contenu que l'on recupere si l'entree est jugee pertinente
```

Chaque position fabrique donc trois versions de son embedding:

```text
embedding_i -> query_i
embedding_i -> key_i
embedding_i -> value_i
```

Pour produire la sortie de la position `i`:

1. `query_i` demande: "qu'est-ce qui m'est utile dans le contexte ?"
2. On compare `query_i` aux `key_j` des positions autorisees.
3. Ces comparaisons donnent des scores.
4. Le softmax transforme les scores en poids.
5. On melange les `value_j` avec ces poids.

On ne melange pas les keys: les keys servent a etre trouvees. Ce sont les values qui portent
l'information recuperee.

Version courte:

```text
Q cherche
K permet de comparer
V transporte l'information
```

Pourquoi trois projections au lieu d'un seul vecteur ? Parce que "servir a etre retrouve",
"chercher quelque chose" et "fournir du contenu" sont trois roles differents. C'est le meme
genre de separation que dans une application ou un objet peut avoir un champ d'indexation et
un champ de contenu.

## Exemple concret avec `llm`

Prenons une sequence de trois caracteres:

```text
position 0 -> "l"
position 1 -> "l"
position 2 -> "m"
```

Chaque position a son propre embedding, puis ses propres projections:

```text
"l" position 0 -> Q0, K0, V0
"l" position 1 -> Q1, K1, V1
"m" position 2 -> Q2, K2, V2
```

Meme si les deux premiers caracteres sont tous les deux `"l"`, ils occupent deux positions
differentes. Dans ce module, comme on n'a pas encore de positional encoding, leurs embeddings
de depart seront identiques. Mais leur **sortie contextualisee** peut differer, car la
position 0 et la position 1 n'ont pas acces au meme contexte.

Avec le masque causal:

```text
position 0 peut regarder: position 0
position 1 peut regarder: position 0, position 1
position 2 peut regarder: position 0, position 1, position 2
```

Donc pour la position 0:

```text
output0 = 1.0 * V0
```

Elle n'a pas de passe et ne peut pas regarder le futur. Son vecteur contextualise est donc
exactement sa value.

Pour la position 1:

```text
score(1, 0) = dot(Q1, K0) / sqrt(attentionDimension)
score(1, 1) = dot(Q1, K1) / sqrt(attentionDimension)
weights1 = softmax([score(1, 0), score(1, 1)])
output1 = weights1[0] * V0 + weights1[1] * V1
```

Cette fois, le deuxieme `"l"` peut melanger sa propre information avec celle du premier
`"l"`. Son vecteur contextualise peut donc etre different de sa simple value.

Pour la position 2:

```text
output2 = weights2[0] * V0 + weights2[1] * V1 + weights2[2] * V2
```

Le `"m"` peut regarder toute la sequence disponible jusque-la: les deux `"l"` precedents et
lui-meme.

Point important: les keys ne sont pas melangees dans la sortie. Les keys servent a calculer
les poids. Ce sont les values qui sont melangees.

## Pourquoi Q, K et V sont initialises aleatoirement ?

Les matrices `queryWeights`, `keyWeights` et `valueWeights` sont les parametres qui
transforment un embedding en query, key et value.

Dans un vrai Transformer, elles sont generalement initialisees avec de petites valeurs
aleatoires, puis modifiees pendant l'entrainement. Au depart, elles ne savent rien faire de
special. La training loop les ajuste progressivement pour produire de meilleures queries,
keys et values.

Dans ce module, il n'y a pas encore d'entrainement. Les matrices restent donc aleatoires,
mais avec une seed deterministe:

```text
meme seed -> memes matrices -> memes sorties de demo
```

Le but n'est pas encore d'obtenir une attention intelligente. Le but est de voir le mecanisme:

```text
embedding -> Q/K/V -> scores -> softmax -> somme ponderee des values
```

## Exemple

```ts
import { createSelfAttention } from './index.js'

const attention = createSelfAttention({
    embeddingDimension: 4,
    attentionDimension: 4,
    seed: 123,
})

const result = attention.applyCausalSelfAttention([
    [0.1, 0.2, 0.3, 0.4],
    [0.2, 0.1, 0.0, 0.3],
])

console.info(result.attentionWeights)
console.info(result.outputVectors)
```

Pour lancer une demo executable:

```bash
npm run demo:05-attention
```

La demo affiche d'abord un exemple avec `llm`, puis, dans un terminal interactif, elle permet
de saisir une ou plusieurs lettres du vocabulaire. Appuie sur `ENTREE` pour valider et sur
`ESC` pour quitter.

## Impact memoire / VRAM

Tout est stocke en tableaux JavaScript CPU. La VRAM consommee est donc 0.

Le cout principal vient de la matrice de scores:

```text
sequenceLength x sequenceLength
```

Ici les sequences de demo sont minuscules, donc le cout est negligeable. Dans un vrai LLM,
cette croissance quadratique est une des raisons pour lesquelles l'attention devient chere.

## Limites

- Une seule tete d'attention.
- Projections deterministes, pas apprises.
- Pas de layer norm.
- Pas de connexion residuelle.
- Pas de feed-forward network.
- Pas encore de training loop.
