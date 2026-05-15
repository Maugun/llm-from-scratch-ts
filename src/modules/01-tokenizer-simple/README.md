# Module 1 — Tokenizer simple caractere

Ce module transforme un texte en ids numeriques, puis reconstruit le texte depuis ces ids.
Il utilise volontairement une regle tres simple: un token correspond a un caractere.

## Pourquoi ce module existe

Un LLM ne manipule pas directement du texte brut. Les calculs d'un modele se font sur des
nombres. Le tokenizer convertit donc une chaine lisible par un humain en sequence discrete
d'entiers utilisable par les prochains modules.

## Concepts

- **Token**: unite discrete manipulee par le modele. Ici, un caractere.
- **Vocabulaire**: liste finie des tokens connus.
- **Id**: position numerique d'un token dans le vocabulaire.
- **Encodage**: conversion `texte -> ids`.
- **Decodage**: conversion `ids -> texte`.

## Exemple

```ts
import { createCharacterTokenizer } from './index.js'

const tokenizer = createCharacterTokenizer('hello world')

const tokenIds = tokenizer.encode('hello')
console.info(tokenIds)

const text = tokenizer.decode(tokenIds)
console.info(text)
```

Dans cet exemple, le texte `hello` devient une sequence d'entiers, puis cette sequence est
reconvertie en texte. Le modele ne verrait que les entiers; nous gardons le decodage pour
pouvoir revenir a une forme lisible.

Pour lancer une demo executable:

```bash
npm run demo:01-tokenizer
```

## Impact memoire / VRAM

Ce module utilise uniquement des tableaux et des `Map` en RAM CPU. Il ne cree aucun tenseur,
donc il n'utilise pas de VRAM. La memoire augmente avec le nombre de caracteres uniques et
la longueur des textes encodes.

## Note Unicode

Le tokenizer utilise `Array.from`, qui parcourt les points de code Unicode. C'est mieux que
`split('')`, mais cela ne correspond pas toujours a ce qu'un humain percoit comme un seul
caractere affiche. Certains symboles complexes, comme des emoji combines ou des caracteres
avec accents decomposes, peuvent etre formes de plusieurs elements Unicode.

Pour ce premier module, cette limite est acceptable: l'objectif est de comprendre le passage
du texte vers des ids, pas de construire un tokenizer universel.

## Limites

- Le vocabulaire depend du texte donne a `createCharacterTokenizer`.
- Un caractere absent du vocabulaire provoque une erreur.
- Les caracteres sont moins efficaces que les sous-mots utilises par les LLM modernes.
- Aucun token special comme `<unk>`, `<pad>`, `<bos>` ou `<eos>` n'est ajoute dans ce module.
