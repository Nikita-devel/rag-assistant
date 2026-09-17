# Un assistant qui cite ses sources — et qui sait se taire

> Étude de cas · assistant de recherche documentaire pour une PME industrielle
> **Démonstration en ligne :** https://assistant-droit-travail-405627799203.europe-west9.run.app
> **Version web de cette étude :** https://claude.ai/artifact/CUxT52d6B6VDNeKQ38XVyS

Un assistant de recherche sur le Code du travail et la convention collective de
la métallurgie (IDCC 3248), conçu pour une entreprise de tôlerie. Chaque
affirmation renvoie à l'article exact. Quand la réponse n'est pas dans les
textes, le modèle de langage n'est même pas appelé.

2 204 articles · 2 634 extraits indexés · réponse en 3 à 4 secondes ·
Python, FastAPI, Chroma.

## Le problème

Une PME industrielle accumule des règles qu'elle ne lit jamais : le Code du
travail, une convention collective de plusieurs centaines d'articles, des
procédures internes. Chaque semaine, quelqu'un passe une heure à chercher ce que
dit exactement la règle sur les heures supplémentaires, l'astreinte ou le temps
d'habillage.

Un chatbot qui répond de mémoire est pire qu'inutile dans ce contexte : une
réponse fausse énoncée avec assurance sur le temps de travail coûte de l'argent
réel, et se découvre aux prud'hommes. L'objectif n'a donc jamais été « paraître
compétent ». Il était **vérifiable**.

## Deux contraintes de conception

**Chaque affirmation porte son article.** Les extraits arrivent numérotés, le
modèle doit citer par numéro, et ces numéros sont ensuite résolus en références
réelles. Le lecteur n'a pas à faire confiance au système : il clique et lit
l'article L3141-3 lui-même.

**Pas de contexte, pas d'appel au modèle.** Quand la recherche n'est pas
confiante, le modèle de langage n'est pas invoqué du tout — et non pas invoqué
avec une consigne polie lui demandant de refuser.

> Une consigne dans un prompt est une demande. Ne pas appeler le modèle est une
> garantie.

## Ce qui a été mesuré

13 questions dont les bonnes réponses ont été vérifiées article par article dans
le corpus, plus 4 questions hors sujet qui doivent être refusées.

| Stratégie | Recall@1 | Recall@5 | MRR | Refus corrects |
|---|---|---|---|---|
| Vectoriel seul (e5-small) | 0,23 | 0,54 | 0,34 | 0 / 4 |
| Lexical seul (BM25) | 0,15 | 0,62 | 0,35 | 1 / 4 |
| Hybride (fusion RRF) | 0,31 | 0,62 | 0,44 | 0 / 4 |
| **Hybride + reclassement (cross-encoder)** | **0,54** | **0,85** | **0,60** | **4 / 4** |

La recherche vectorielle seule retrouve l'article dans 54 % des cas. Le texte
juridique est dense en formules presque identiques, et un bi-encodeur les
projette toutes dans un cône étroit. La recherche lexicale rattrape les termes
exacts — « astreinte », « habillage », un numéro d'article — mais rate les
paraphrases. Fusionner les deux, puis reclasser les candidats avec un
cross-encodeur, porte le chiffre à 85 %.

La dernière colonne est celle que les démonstrations passent généralement sous
silence. Le seuil de refus n'est pas choisi à la main : il est calibré sur les
données, en pondérant une réponse fausse deux fois plus lourd qu'un refus
injustifié.

## Ce qu'il a fallu casser pour y arriver

**1. L'évaluation était cassée avant le système.** La première version comparait
« Article L3141-3 » à l'étiquette « L3141-3 » et notait chaque stratégie à 0,00 —
un bug de mesure qui ressemblait trait pour trait à un échec total. Un benchmark
qu'on n'a pas testé n'est pas un benchmark.

**2. La recherche n'était pas déterministe.** La base vectorielle ne promet aucun
ordre de lecture, cet ordre devenait l'index lexical, et l'index départageait les
ex æquo partout en aval. Deux exécutions sur un index identique donnaient des
scores différents — et un seuil calibré différent. Un tri par empreinte de
contenu a réglé le problème ; un test le garde fermé.

**3. Le seuil de refus avait été deviné.** Fixé à la main, il refusait des
questions parfaitement valides : « le temps d'habillage est-il payé » obtenait
−2,61 et partait au refus. Seules les erreurs d'un côté étaient mesurées. L'outil
de calibration mesure désormais les deux.

**4. Les questions sans accents cassaient tout.** « Quel delai pour prevenir un
salarie » faisait passer le bon article du rang 1 au rang 5 : la recherche
lexicale replie les accents des deux côtés, l'embedder voit *delai* et *délai*
comme deux mots distincts. La table de restauration est générée depuis le corpus
lui-même et ne garde que les correspondances sans ambiguïté — ce qui écarte
*sur*/*sûr* et *cote*/*côte*.

## Et pour votre entreprise ?

Cette démonstration publique tourne sur une infrastructure cloud, sur des textes
juridiques publics. Pour des documents internes — vos procédures, vos accords
d'entreprise, vos fiches techniques — le même système s'installe sur vos propres
machines : aucune donnée ne quitte l'entreprise. C'est une ligne de
configuration, pas une réécriture.

Je suis développeur indépendant et je conçois ce type d'outils internes sur
mesure pour les PME industrielles.

---

Corpus : données ouvertes publiées par la DILA (Code du travail ; convention
collective de la métallurgie, IDCC 3248). Démonstration technique — ne constitue
pas un conseil juridique.
Code source et benchmark : https://github.com/Nikita-devel/rag-assistant
