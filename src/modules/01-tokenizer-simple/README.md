# Module 1 — Tokenizer simple caractère

Ce module transforme un texte en ids numériques, puis reconstruit le texte depuis ces ids.
Il utilise volontairement une règle très simple: un token correspond à un caractère.

## Pourquoi ce module existe

Un LLM ne manipule pas directement du texte brut. Les calculs d'un modèle se font sur des
nombres. Le tokenizer convertit donc une chaîne lisible par un humain en séquence discrète
d'entiers utilisable par les prochains modules.

## Concepts

- **Token**: unité discrète manipulée par le modèle. Ici, un caractère.
- **Vocabulaire**: liste finie des tokens connus.
- **Id**: position numérique d'un token dans le vocabulaire.
- **Encodage**: conversion `texte -> ids`.
- **Décodage**: conversion `ids -> texte`.

## Exemple

```ts
import { createCharacterTokenizer } from './index.js'

const tokenizer = createCharacterTokenizer('hello world')

const tokenIds = tokenizer.encode('hello')
console.info(tokenIds)

const text = tokenizer.decode(tokenIds)
console.info(text)
```

Dans cet exemple, le texte `hello` devient une séquence d'entiers, puis cette séquence est
reconvertie en texte. Le modèle ne verrait que les entiers; nous gardons le décodage pour
pouvoir revenir à une forme lisible.

Pour lancer une démo exécutable:

```bash
npm run demo:01-tokenizer
```

## Impact mémoire / VRAM

Ce module utilise uniquement des tableaux et des `Map` en RAM CPU. Il ne crée aucun tenseur,
donc il n'utilise pas de VRAM. La mémoire augmente avec le nombre de caractères uniques et
la longueur des textes encodés.

## Note Unicode

Le tokenizer utilise `Array.from`, qui parcourt les points de code Unicode. C'est mieux que
`split('')`, mais cela ne correspond pas toujours à ce qu'un humain perçoit comme un seul
caractère affiché. Certains symboles complexes, comme des emoji combinés ou des caractères
avec accents décomposés, peuvent être formés de plusieurs éléments Unicode.

Pour ce premier module, cette limite est acceptable: l'objectif est de comprendre le passage
du texte vers des ids, pas de construire un tokenizer universel.

## Limites

- Le vocabulaire dépend du texte donné à `createCharacterTokenizer`.
- Un caractère absent du vocabulaire provoque une erreur.
- Les caractères sont moins efficaces que les sous-mots utilisés par les LLM modernes.
- Aucun token spécial comme `<unk>`, `<pad>`, `<bos>` ou `<eos>` n'est ajouté dans ce module.
