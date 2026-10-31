# JCKouassi · Assistant juridique

**Une initiative de Kouassi Jean-Charles, juriste.**

Assistant d'information juridique en protection des données personnelles et en droit des affaires, pour la
**Côte d'Ivoire**, le **Maroc**, l'espace **OHADA**, la **CEDEAO** et l'**Union africaine**. Il répond à partir
des textes officiels, article par article, et cite ses sources.

> Information juridique, pas un conseil juridique. Vérifiez toujours le texte officiel en vigueur.

## Fonctionnalités

- **Question juridique** : réponse rédigée à partir des seuls articles retrouvés, au choix en *synthèse* ou en
  *note juridique* (réponse, textes applicables, analyse, points de vigilance), avec la liste des articles
  consultés, lisibles en entier et copiables pour la citation.
- **Aide à la formulation** : thèmes avec questions types, composition guidée (thème, pays, ce que l'on
  cherche, situation) et bouton « Améliorer ma question » qui reformule en termes juridiques et signale les
  informations manquantes. Après chaque réponse : la question telle que comprise, les précisions utiles et
  trois questions de suivi en un clic.
- **Citation directe** : « Que prévoit l'article 309 de l'AUSCGIE ? » va chercher l'article exact.
- **Consulter un article** : texte intégral d'un article, avec navigation vers l'article précédent ou suivant.
- **Bibliothèque** : liste des textes intégrés, avec leur source.
- **Périmètre** : filtre par domaine et par pays ou espace. Une question sur la Côte d'Ivoire interroge aussi
  les textes OHADA, CEDEAO et Union africaine qui s'y appliquent.

## Textes intégrés

**Côte d'Ivoire** : loi n° 2013-450 relative à la protection des données à caractère personnel ; décisions
n° 2023-0936, 2023-0937 et 2023-0938 de l'ARTCI.

**Maroc** : loi n° 09-08 ; code de commerce (version consolidée 2019) ; Dahir des obligations et contrats ;
loi n° 5-96 (version d'origine de 1997) ; loi n° 49-16 sur les baux commerciaux ; loi n° 53-95 sur les
juridictions de commerce.

**OHADA** : Actes uniformes relatifs au droit commercial général ; au droit des sociétés commerciales et du
groupement d'intérêt économique ; aux sûretés ; aux procédures simplifiées de recouvrement et aux voies
d'exécution (2023) ; aux procédures collectives d'apurement du passif ; au droit de l'arbitrage ; à la
médiation ; aux contrats de transport de marchandises par route ; au droit des sociétés coopératives ; au droit
comptable et à l'information financière ; au système comptable des entités à but non lucratif (partie
législative).

**CEDEAO** : Acte additionnel A/SA.1/01/10 relatif à la protection des données à caractère personnel.

**Union africaine** : Convention sur la cybersécurité et la protection des données à caractère personnel
(Convention de Malabo).

## Comment ça marche

1. Chaque texte (dossier `corpus/`) est découpé article par article, avec son livre, son titre et son chapitre.
2. La question est comparée aux articles par une recherche par mots-clés (BM25), avec des équivalences de
   vocabulaire (« ARTCI » / « Commission nationale », « SAS » / « société par actions simplifiée »…) et un
   renforcement du texte visé par la question.
3. Un modèle open source (via l'API Groq) rédige la réponse à partir des seuls articles retrouvés.

## Ajouter un texte

Déposer un fichier `.txt` dans `corpus/donnees/` ou `corpus/affaires/`, commençant par :

```
# titre: Intitulé complet du texte
# label: Nom court utilisé dans les citations
# pays: Côte d'Ivoire | Maroc | OHADA | CEDEAO | Union africaine
# domaine: donnees | affaires
# source: Origine du texte
```

## Déploiement

Streamlit Community Cloud, fichier principal `streamlit_app.py`. Secrets :

```
GROQ_API_KEY = "..."
GROQ_MODEL = "openai/gpt-oss-120b"
```

Conçu et développé par Kouassi Jean-Charles · JCKouassi, Juriste 2.0.
