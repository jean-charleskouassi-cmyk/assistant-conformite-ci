# -*- coding: utf-8 -*-
"""
JCKouassi · Assistant juridique
Une initiative de Kouassi Jean-Charles.

Données personnelles et droit des affaires : Côte d'Ivoire, Maroc, OHADA,
CEDEAO et Union africaine.

- Le corpus est dans corpus/ : un fichier texte par texte juridique, avec un
  en-tête (titre, label, pays, domaine, source).
- Chaque texte est découpé article par article, puis indexé par mots-clés (BM25).
- Les réponses sont rédigées par un modèle open source (API Groq) à partir des
  seuls articles retrouvés, avec leurs références.
- La clé Groq est lue dans les Secrets de l'application, jamais dans le code.
"""

import json
import re
import unicodedata
from pathlib import Path

import streamlit as st
from groq import Groq
from rank_bm25 import BM25Okapi

ICI = Path(__file__).parent
CORPUS_DIR = ICI / "corpus"
LOGO_H = ICI / "assets" / "logo_horizontal_sombre.png"
LOGO_ICONE = ICI / "assets" / "logo_icone.png"

AUTEUR = "Kouassi Jean-Charles"
CONTACT = "nshinobi701@gmail.com"
MAX_CHUNK = 1800
TOP_K = 7
DEFAULT_MODEL = "openai/gpt-oss-120b"

ESPACES = ["Côte d'Ivoire", "Maroc", "OHADA", "CEDEAO", "Union africaine"]
DOMAINES = {"donnees": "Données personnelles", "affaires": "Droit des affaires"}
# Textes communautaires applicables quand la question vise un pays.
APPLICABLES = {"Côte d'Ivoire": ["OHADA", "CEDEAO", "Union africaine"], "Maroc": ["Union africaine"]}

SYSTEME = """Tu es l'assistant juridique de JCKouassi, conçu par Kouassi Jean-Charles. Tu informes sur la \
protection des données personnelles et le droit des affaires en Côte d'Ivoire, au Maroc, dans l'espace \
OHADA (les Actes uniformes s'appliquent notamment en Côte d'Ivoire, pas au Maroc), dans la CEDEAO et \
dans l'Union africaine.

Règles impératives :
1. Appuie-toi UNIQUEMENT sur les extraits fournis. N'affirme aucune règle de droit qui n'y figure pas.
   Les extraits sont trouvés automatiquement et certains peuvent être hors sujet : IGNORE-les. Ne cite
   jamais un article sans rapport direct avec la question, même pour « illustrer ».
2. Cite chaque règle avec sa référence exacte entre crochets, telle qu'elle figure dans l'en-tête de \
l'extrait, par exemple [Loi 09-08 (Maroc), art. 4].
3. Si les extraits ne suffisent pas, ne dis JAMAIS que le texte manque ou n'est « pas fourni » : la \
bibliothèque contient les textes complets. Dis que les articles retrouvés ne traitent pas ce point et \
indique, d'après le plan fourni, les articles à lire dans l'onglet « Consulter un article ». N'invente \
jamais un numéro d'article, un délai ou un montant.
4. Si plusieurs pays ou espaces sont concernés, distingue nettement ce que prévoit chacun.
5. Si l'en-tête d'un extrait indique « VERSION D'ORIGINE » ou « OCR », signale qu'il faut vérifier le \
texte officiel en vigueur avant toute citation définitive.
6. Si la question est pratique (« comment créer… », « que faire si… »), réponds par étapes concrètes et
   numérotées, chacune rattachée à l'article qui la fonde.
7. Si la question ne précise pas le pays, traite séparément la Côte d'Ivoire (droit OHADA) et le Maroc
   quand les extraits le permettent, et invite à préciser le pays.
8. Écris en français juridique clair, sans jargon inutile.
9. Termine par une phrase rappelant qu'il s'agit d'une information juridique et non d'un conseil, \
à faire valider par un professionnel."""

FORMATS = {
    "Synthèse": ("Commence par la réponse directe en deux ou trois phrases. Donne ensuite ses fondements "
                 "textuels, puis, si c'est utile, une courte liste « À faire » avec les démarches concrètes."),
    "Note juridique": ("Présente ta réponse comme une courte note juridique, avec ces intertitres : "
                       "**Réponse**, **Textes applicables**, **Analyse** (principe, conditions, exceptions, "
                       "sanctions le cas échéant), **Points de vigilance**."),
}

STOP = set("""a au aux avec ce ces cette dans de des du elle en est et etre il ils je la le les leur \
leurs lui ma mais me meme mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son \
sur ta te tes toi ton tu un une vos votre vous y quel quelle quels quelles comment est-ce peut \
peuvent doit doivent faut sont ont cas loi article articles selon""".split())


# ------------------------------------------------------------------ recherche

def normaliser(texte):
    texte = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in texte if not unicodedata.combining(c))


SUFFIXES = ("issements", "issement", "ations", "ation", "ements", "ement", "atrices", "atrice",
            "ables", "able", "ibles", "ible",
            "ateurs", "ateur", "euses", "euse", "ances", "ance", "ences", "ence", "ites", "ite",
            "ives", "ive", "ifs", "if", "aux", "ees", "ee", "es", "er", "ez", "e", "s", "x")


def raciner(mot):
    """Racinisation très simple du français : déclarer / déclaration -> declar."""
    if len(mot) <= 4 or mot.isdigit():
        return mot
    for suf in SUFFIXES:
        if mot.endswith(suf) and len(mot) - len(suf) >= 4:
            mot = mot[: -len(suf)]
            break
    return mot[:-2] + "c" if mot.endswith("qu") else mot


def tokeniser(texte):
    return [raciner(m) for m in re.findall(r"[a-z0-9]+", normaliser(texte))
            if m not in STOP and len(m) > 1]


ARTICLE_RE = re.compile(
    r"(?m)^[\W_]{0,4}(?:Article|ARTICLE|Art\.)\s+"
    r"(premier|Premier|PREMIER|1er|\d+(?:\s*[-‑]\s*\d+)?(?:\s*bis|\s*ter)?)\b")
TITRE_RE = re.compile(
    r"(?m)^\s*((?i:livre|titre|chapitre|section))\s+"
    r"(?:[0-9]+(?:-[0-9]+)?|[IVXLC]+|(?i:premier|première|preliminaire|préliminaire|unique|bis))"
    r"(?:\s*(?:[-:–—.]|$)|\s+[A-ZÀ-ÖØ-Þ(«])[^\n]{0,120}")


def lire_document(chemin):
    brut = chemin.read_text(encoding="utf-8")
    meta, lignes = {}, brut.split("\n")
    i = 0
    while i < len(lignes) and lignes[i].startswith("# ") and ":" in lignes[i]:
        cle, val = lignes[i][2:].split(":", 1)
        meta[cle.strip()] = val.strip()
        i += 1
    meta.setdefault("label", chemin.stem)
    meta.setdefault("titre", meta["label"])
    meta.setdefault("pays", "")
    meta.setdefault("domaine", chemin.parent.name)
    meta.setdefault("source", "")
    return meta, "\n".join(lignes[i:]).strip()


def cle_article(num):
    if num.lower() in ("premier", "1er"):
        return (1, 0)
    parts = re.findall(r"\d+", num)
    return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0) if parts else (0, 0)


def decouper(meta, texte):
    """Découpe un texte en passages : un par article, les longs articles en plusieurs parties."""
    reperes = list(ARTICLE_RE.finditer(texte))
    retenus, precedent = [], (0, -1)
    for m in reperes:
        k = cle_article(m.group(1))
        if k > precedent:            # sinon : article d'un autre texte cité en note
            retenus.append(m)
            precedent = k
    if len(retenus) < 5:                 # décision, note… : découpage en extraits
        retenus = []
    if not retenus:
        blocs = [("", texte)]
    else:
        blocs = []
        if retenus[0].start() > 300:
            blocs.append(("", texte[:retenus[0].start()]))
        for n, m in enumerate(retenus):
            fin = retenus[n + 1].start() if n + 1 < len(retenus) else len(texte)
            num = re.sub(r"\s+", " ", m.group(1))
            num = "premier" if num.lower() in ("premier", "1er") else num.replace(" ", "")
            blocs.append((num, texte[m.start():fin]))
    niveaux = ["livre", "titre", "chapitre", "section"]
    plan, passages = {}, []
    for num, bloc in blocs:
        contexte = " > ".join(plan[n] for n in niveaux if n in plan)
        for t in TITRE_RE.finditer(bloc):      # intitulés valables pour la suite
            niv = t.group(1).lower()
            plan[niv] = t.group(0).strip()
            for inf in niveaux[niveaux.index(niv) + 1:]:
                plan.pop(inf, None)
        corps = bloc.strip()
        if len(corps) < 25:
            continue
        if num:
            ref = f"{meta['label']}, art. {num}"
        elif retenus:
            ref = f"{meta['label']}, préambule et dispositions liminaires"
        else:
            ref = meta["label"]
        morceaux = [corps[k:k + MAX_CHUNK] for k in range(0, len(corps), MAX_CHUNK - 150)]
        for j, morceau in enumerate(morceaux):
            suffixe = ""
            if len(morceaux) > 1:
                suffixe = f" (partie {j + 1})" if num else f" (extrait {j + 1})"
            passages.append({"ref": ref + suffixe, "texte": morceau, "num": num, "partie": j,
                             "label": meta["label"], "titre": meta["titre"], "pays": meta["pays"],
                             "domaine": meta["domaine"], "contexte": contexte})
    return passages


@st.cache_resource(show_spinner="Chargement de la bibliothèque juridique…")
def charger_corpus():
    passages, textes = [], []
    for chemin in sorted(CORPUS_DIR.glob("*/*")):
        if chemin.suffix.lower() not in (".txt", ".md"):
            continue
        meta, texte = lire_document(chemin)
        p = decouper(meta, texte)
        passages.extend(p)
        textes.append({**meta, "articles": len({x["num"] for x in p if x["num"]}), "passages": len(p)})
    # Les intitulés (titre, chapitre, section) sont répétés pour peser davantage.
    index = BM25Okapi([tokeniser(" ".join([p["ref"], p["contexte"], p["contexte"], p["contexte"],
                                           p["texte"]])) for p in passages])
    # Table des articles : label -> liste ordonnée de (numéro, [indices des passages])
    articles = {}
    for i, p in enumerate(passages):
        if not p["num"]:
            continue
        lst = articles.setdefault(p["label"], [])
        if lst and lst[-1][0] == p["num"]:
            lst[-1][1].append(i)
        else:
            lst.append((p["num"], [i]))
    plans = {lab: plan_texte(lab, passages, articles, 2) for lab in articles}
    return passages, index, textes, articles, plans


def plan_texte(label, passages, articles, niveaux=2):
    """Plan d'un texte : [(intitulé, premier article, dernier article)], d'après livres / titres."""
    out, cur, premier, dernier = [], None, None, None
    for n, idx in articles.get(label, []):
        parts = [x for x in passages[idx[0]]["contexte"].split(" > ") if x]
        tete = " > ".join(parts[:niveaux])
        if tete != cur:
            if cur is not None:
                out.append((cur, premier, dernier))
            cur, premier = tete, n
        dernier = n
    if cur is not None:
        out.append((cur, premier, dernier))
    return out


def plan_resume(label, plans, niveaux=1):
    """Plan compact d'un texte, au premier niveau, pour les consignes du modèle."""
    vus, lignes = set(), []
    for intitule, a, b in plans.get(label, []):
        tete = intitule.split(" > ")[0] if niveaux == 1 else intitule
        if not tete or tete in vus:
            continue
        vus.add(tete)
        fin = b
        for i2, a2, b2 in plans[label]:
            if i2.split(" > ")[0] == tete:
                fin = b2
        lignes.append(f"art. {a} à {fin} : {tete[:70]}")
    return f"{label} — " + " ; ".join(lignes) if lignes else ""


EXPANSIONS = {
    r"\bsarl\b": " société à responsabilité limitée",
    r"\bsas\b": " société par actions simplifiée",
    r"\bsa\b": " société anonyme",
    r"\bsnc\b": " société en nom collectif",
    r"\bgie\b": " groupement d'intérêt économique",
    r"\brccm\b": " registre du commerce et du crédit mobilier",
    r"\bminimum\b|\bminimal\b": " au moins",
    r"\bdpo\b": " correspondant à la protection des données délégué",
    r"\bcndp\b": " commission nationale",
    r"\bartci\b": " autorité de protection",
    r"autorit[ée] de (?:protection|contr[ôo]le)|\bautorit[ée]\b": " commission nationale",
    r"\bcommission nationale\b": " autorité de protection",
    r"\bbail commercial\b": " fonds de commerce locataire bailleur preneur",
    r"\bd[ée]clar": " déclaration préalable récépissé",
    r"\bautoris": " autorisation préalable",
    r"\bfaillite\b": " redressement judiciaire liquidation des biens cessation des paiements",
    r"\bsaisie\b": " saisie voie d'exécution titre exécutoire",
}
ESPACES_CITES = {
    "Maroc": r"maroc|marocain",
    "Côte d'Ivoire": r"c[oô]te d.ivoire|ivoirien|\bci\b|artci",
    "OHADA": r"ohada|acte uniforme|\bau[a-z]{1,6}\b",
    "CEDEAO": r"cedeao|ecowas",
    "Union africaine": r"union africaine|malabo",
}
# Abréviations courantes -> label du texte, pour « article 309 AUSCGIE » etc.
ALIAS_TEXTES = {
    r"\bauscgie\b|soci[ée]t[ée]s commerciales": "AUSCGIE", r"\baudcg\b|droit commercial g[ée]n[ée]ral": "AUDCG",
    r"\baus\b|s[ûu]ret[ée]s": "(AUS)", r"\baupsrve\b|voies d.ex[ée]cution|recouvrement": "AUPSRVE",
    r"\baupcap\b|proc[ée]dures collectives": "AUPCAP", r"\baua\b|arbitrage": "(AUA)",
    r"\baum\b|m[ée]diation": "(AUM)", r"\bauctmr\b|transport de marchandises": "AUCTMR",
    r"\bauscoop\b|coop[ée]rative": "AUSCOOP", r"\baudcif\b|droit comptable": "AUDCIF",
    r"\bsycebnl\b|but non lucratif": "SYCEBNL", r"09-08": "Loi 09-08", r"2013-450": "Loi 2013-450",
    r"5-96": "Loi 5-96", r"49-16": "Loi 49-16", r"53-95": "Loi 53-95",
    r"code de commerce": "Code de commerce", r"\bdoc\b|obligations et (?:des )?contrats": "DOC",
    r"malabo": "Malabo", r"cedeao|a/sa\.?1": "CEDEAO",
}


# Notions qui désignent un texte précis : on renforce le score de ses articles.
NOTIONS_TEXTES = {
    r"cr[ée]e[rz]? (?:ma |une |sa )?soci[ée]t[ée]|constitu|immatricul|statuts": ["AUSCGIE", "AUDCG", "Loi 5-96", "Code de commerce"],
    r"\bsas\b|actions simplifi[ée]e": ["AUSCGIE"],
    r"\bsarl\b|responsabilit[ée] limit[ée]e": ["AUSCGIE", "Loi 5-96"],
    r"\bsa\b|soci[ée]t[ée] anonyme|conseil d.administration": ["AUSCGIE"],
    r"\bgie\b|groupement d.int[ée]r[êe]t": ["AUSCGIE"],
    r"coop[ée]rative": ["AUSCOOP"],
    r"injonction de payer|saisie|voies d.ex[ée]cution": ["AUPSRVE", "Loi 53-95"],
    r"redressement judiciaire|liquidation|cessation des paiements|conciliation|sauvegarde": ["AUPCAP", "Code de commerce"],
    r"bail commercial|fonds de commerce|rccm|registre du commerce|entreprenant": ["AUDCG", "Code de commerce", "Loi 49-16"],
    r"hypoth[èe]que|gage|nantissement|cautionnement|s[ûu]ret[ée]": ["(AUS)", "DOC"],
    r"arbitr": ["(AUA)"], r"m[ée]diat": ["(AUM)"], r"transport": ["AUCTMR"],
    r"comptab|[ée]tats financiers": ["AUDCIF", "SYCEBNL"],
}


ANALYSE = """Tu es documentaliste juridique (OHADA, Côte d'Ivoire, Maroc, CEDEAO, Union africaine).
Pour la question posée, réponds UNIQUEMENT par un objet JSON de cette forme :
{"question_precise": "la question reformulée en une phrase juridique claire et précise, en français",
 "mots_cles": "termes juridiques précis tels qu'ils figurent dans les textes, synonymes compris (20 mots au plus)",
 "espaces": ["Côte d'Ivoire" | "Maroc" | "OHADA" | "CEDEAO" | "Union africaine"],
 "articles": [{"texte": "AUSCGIE", "num": "10"}],
 "precisions": ["information manquante formulée comme une courte question à l'utilisateur"],
 "suivis": ["question de suivi utile"]}
- "question_precise" : garde le sens de l'utilisateur ; n'ajoute un pays que s'il est indiqué ou évident.
- "espaces" : seulement ceux que la question vise explicitement ou implicitement ; liste vide si rien n'est précisé.
- "articles" : au plus 6 articles centraux (par exemple les premiers articles du titre qui traite le sujet,
  d'après le plan fourni). Valeurs possibles pour "texte" : AUSCGIE, AUDCG, AUS,
  AUPSRVE, AUPCAP, AUA, AUM, AUCTMR, AUSCOOP, AUDCIF, SYCEBNL, Loi 09-08, Loi 2013-450, Loi 5-96, Loi 49-16,
  Loi 53-95, Code de commerce, DOC, Malabo, CEDEAO. Liste vide si tu n'es pas sûr.
- "precisions" : au plus 3 informations qui manquent pour bien répondre (pays, forme de société, qualité de la
  personne, montant…) ; liste vide si la question est assez précise.
- "suivis" : 3 questions de suivi courtes et concrètes qu'un juriste poserait ensuite."""


def analyser_question(client, question, precedente="", plan_txt=""):
    """Reformule la question en vocabulaire juridique (modèle rapide). Ne bloque jamais."""
    modele = st.secrets["GROQ_MODEL_RAPIDE"] if "GROQ_MODEL_RAPIDE" in st.secrets else "openai/gpt-oss-20b"
    contenu = ((f"Plan des textes disponibles (appuie-toi dessus pour choisir les articles) :\n{plan_txt}\n\n"
                if plan_txt else "")
               + (f"Question précédente (contexte) : {precedente}\n" if precedente else "")
               + f"Question : {question}")
    try:
        rep = client.chat.completions.create(
            model=modele, temperature=0, max_tokens=700,
            messages=[{"role": "system", "content": ANALYSE}, {"role": "user", "content": contenu}])
        brut = rep.choices[0].message.content or ""
        m = re.search(r"\{.*\}", brut, re.S)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}

    def liste(cle, n):
        v = data.get(cle, [])
        return [x for x in v if x][:n] if isinstance(v, list) else []

    return {"question_precise": str(data.get("question_precise", ""))[:500],
            "mots_cles": str(data.get("mots_cles", ""))[:400],
            "espaces": [e for e in liste("espaces", 5) if e in ESPACES],
            "articles": [a for a in liste("articles", 6) if isinstance(a, dict)],
            "precisions": [str(x)[:160] for x in liste("precisions", 3)],
            "suivis": [str(x)[:200] for x in liste("suivis", 3)]}


def articles_suggeres(suggestions, passages, articles):
    """Va chercher le texte réel des articles suggérés (ignorés s'ils n'existent pas)."""
    trouves = []
    for sug in suggestions:
        frag, num = str(sug.get("texte", "")), str(sug.get("num", "")).strip()
        if not frag or not num:
            continue
        cle = f"({frag})" if frag in ("AUS", "AUA", "AUM") else frag
        for lab, liste in articles.items():
            if cle in lab:
                for n, idx in liste:
                    if n == num:
                        trouves.append(passages[idx[0]])
    return trouves


def articles_cites(question, passages, articles):
    """Repère « art. 309 de l'AUSCGIE » et renvoie les passages de cet article."""
    nums = re.findall(r"\bart(?:icle|\.)?\s*(\d+(?:-\d+)?|premier|1er)\b", question, re.I)
    if not nums:
        return []
    labels = [lab for lab in articles
              if any(re.search(m, question, re.I) and frag in lab for m, frag in ALIAS_TEXTES.items())]
    trouves = []
    for lab in labels:
        for num in nums:
            num = "premier" if num.lower() in ("premier", "1er") else num
            for n, idx in articles[lab]:
                if n == num:
                    trouves.extend(passages[i] for i in idx)
    return trouves


def rechercher(question, passages, index, articles, domaines, espaces, k=TOP_K, analyse=None):
    """Articles cités explicitement, puis meilleurs passages ; chaque espace pertinent a sa part."""
    analyse = analyse or {}
    exacts = articles_cites(question, passages, articles)
    for p in articles_suggeres(analyse.get("articles", []), passages, articles):
        if p not in exacts and p["domaine"] in domaines and p["pays"] in espaces:
            exacts.append(p)
    requete = question + " " + analyse.get("mots_cles", "")
    for motif, ajout in EXPANSIONS.items():
        if re.search(motif, question, re.I):
            requete += ajout
    cites = [e for e, motif in ESPACES_CITES.items() if re.search(motif, question, re.I) and e in espaces]
    cites += [e for e in analyse.get("espaces", []) if e in espaces and e not in cites]
    if cites:
        etendu = set(cites)
        for e in cites:
            etendu.update(a for a in APPLICABLES.get(e, []) if a in espaces)
        espaces = list(etendu)
    for motif in ESPACES_CITES.values():
        requete = re.sub(motif, " ", requete, flags=re.I)
    if len(espaces) > 1:
        k = max(k, 8)
    scores = index.get_scores(tokeniser(requete))
    vises = [frag for motif, frags in NOTIONS_TEXTES.items() if re.search(motif, question, re.I)
             for frag in frags]
    if vises:
        scores = [sc * 1.6 if any(f in passages[i]["label"] for f in vises) else sc
                  for i, sc in enumerate(scores)]
    ordre = [i for i in sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
             if scores[i] > 0 and passages[i]["domaine"] in domaines and passages[i]["pays"] in espaces]
    choisis = []
    if ordre:
        meilleur = scores[ordre[0]]
        par_espace = {}
        for i in ordre:
            par_espace.setdefault(passages[i]["pays"], []).append(i)
        seuil = 0.6 if cites else 0.4      # sans pays précisé, on garde plus d'espaces
        pertinents = [e for e, lst in par_espace.items()
                      if scores[lst[0]] >= seuil * meilleur or e in cites]
        part = max(2, k // max(1, len(pertinents)))
        for e in pertinents:
            choisis.extend(par_espace[e][:part])
        for i in ordre:
            if len(choisis) >= k:
                break
            if i not in choisis:
                choisis.append(i)
        choisis = sorted(choisis[:k], key=lambda i: scores[i], reverse=True)
    resultat = list(exacts[:5])
    for i in choisis:                   # 9 passages au plus : reste dans les quotas gratuits de Groq
        if passages[i] not in resultat and len(resultat) < 9:
            resultat.append(passages[i])
    return resultat


# ------------------------------------------------------------------ appels au modèle (quotas)

def flux_modele(client, messages, modele):
    """Lance la génération ; si le modèle est saturé (quota) ou indisponible, bascule sur le modèle de secours."""
    secours = st.secrets["GROQ_MODEL_SECOURS"] if "GROQ_MODEL_SECOURS" in st.secrets else "openai/gpt-oss-20b"
    derniere = None
    for m in ([modele, secours] if secours != modele else [modele]):
        try:
            return client.chat.completions.create(model=m, messages=messages, stream=True,
                                                  temperature=0.2), m
        except Exception as e:  # noqa: BLE001
            derniere = e
            if not any(x in type(e).__name__ for x in ("RateLimit", "NotFound", "InternalServer", "APIStatus",
                                                         "ServiceUnavailable")):
                raise
    raise derniere


def ecrire_flux(flux):
    def generer():
        for bloc in flux:
            if bloc.choices and bloc.choices[0].delta.content:
                yield bloc.choices[0].delta.content
    return st.write_stream(generer())


def message_erreur(e):
    nom = type(e).__name__
    if "RateLimit" in nom:
        return "Le service est très sollicité (limite de requêtes atteinte). Réessayez dans une minute."
    return f"Le service de génération est momentanément indisponible ({nom}). Réessayez dans un instant."


def textes_probables(question, passages, index, domaines, espaces, n=5):
    """Textes les plus probablement concernés (pour n'envoyer que leur plan à l'analyse)."""
    scores = index.get_scores(tokeniser(question))
    vus = []
    for i in sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)[:60]:
        q = passages[i]
        if scores[i] > 0 and q["domaine"] in domaines and q["pays"] in espaces and q["label"] not in vus:
            vus.append(q["label"])
    for motif, frags in NOTIONS_TEXTES.items():
        if re.search(motif, question, re.I):
            for lab in {q["label"] for q in passages}:
                if any(f in lab for f in frags) and lab not in vus:
                    vus.insert(0, lab)
    return vus[:n]


# ------------------------------------------------------------------ équivalences Maroc / OHADA

EQUIVALENCES = [
    {"theme": "Société à responsabilité limitée (SARL)",
     "maroc": [("Loi 5-96", r"responsabilit[ée] limit")], "ohada": [("AUSCGIE", r"responsabilit[ée] limit")]},
    {"theme": "Société en nom collectif",
     "maroc": [("Loi 5-96", r"nom collectif")], "ohada": [("AUSCGIE", r"nom collectif")]},
    {"theme": "Société en commandite simple",
     "maroc": [("Loi 5-96", r"commandite simple")], "ohada": [("AUSCGIE", r"commandite simple")]},
    {"theme": "Société en commandite par actions",
     "maroc": [("Loi 5-96", r"commandite par actions")], "ohada": None,
     "note": "L'AUSCGIE ne prévoit pas de société en commandite par actions : pas d'équivalent direct en droit OHADA."},
    {"theme": "Société en participation",
     "maroc": [("Loi 5-96", r"participation")], "ohada": [("AUSCGIE", r"Société en participation")]},
    {"theme": "Société anonyme et société par actions simplifiée",
     "maroc": None, "ohada": [("AUSCGIE", r"Société anonyme|actions simplifi")],
     "note": "La loi marocaine n° 17-95 sur les sociétés anonymes n'est pas encore intégrée dans la bibliothèque."},
    {"theme": "Groupement d'intérêt économique",
     "maroc": None, "ohada": [("AUSCGIE", r"Groupement d.int")],
     "note": "Le droit marocain des groupements d'intérêt économique n'est pas encore intégré dans la bibliothèque."},
    {"theme": "Règles communes aux sociétés commerciales",
     "maroc": [("Loi 5-96", r"Dispositions générales|Responsabilit[ée] Civile|publicit|infractions|Dispositions diverses"),
               ("DOC", r"association")],
     "ohada": [("AUSCGIE", r"Constitution de la soci|Fonctionnement de la soci|responsabilit[ée] civile|Formalit|"
                           r"Nullit|Dissolution|Transformation|Fusion|Liens de droit|Capital variable|pénales")]},
    {"theme": "Statut du commerçant et registre du commerce",
     "maroc": [("Code de commerce", r"LIVRE PREMIER")],
     "ohada": [("AUDCG", r"Livre [1-5] |Champ d.application")]},
    {"theme": "Bail commercial",
     "maroc": [("Loi 49-16", None)], "ohada": [("AUDCG", r"Titre 1 - Bail")]},
    {"theme": "Fonds de commerce",
     "maroc": [("Code de commerce", r"LIVRE II:")], "ohada": [("AUDCG", r"Titre 2 - Fonds de commerce")]},
    {"theme": "Intermédiaires de commerce (agence, courtage, commission)",
     "maroc": [("Code de commerce", r"AGENCE COMMERCIALE|COURTAGE|COMMISSION")],
     "ohada": [("AUDCG", r"Intermédiaires de commerce")]},
    {"theme": "Vente commerciale",
     "maroc": [("DOC", r"De la Vente")], "ohada": [("AUDCG", r"Vente commerciale|livrer les marchandises")]},
    {"theme": "Transport de marchandises",
     "maroc": [("Code de commerce", r"LE TRANSPORT")], "ohada": [("AUCTMR", None)]},
    {"theme": "Sûretés : nantissement, gage, cautionnement, hypothèque",
     "maroc": [("Code de commerce", r"NANTISSEMENT"), ("DOC", r"Cautionnement|Nantissement")],
     "ohada": [("(AUS)", None)]},
    {"theme": "Difficultés de l'entreprise et procédures collectives",
     "maroc": [("Code de commerce", r"Livre V")], "ohada": [("AUPCAP", None)]},
    {"theme": "Effets de commerce",
     "maroc": [("Code de commerce", r"EFFETS DE COMMERCE")], "ohada": None,
     "note": "Les effets de commerce (lettre de change, billet à ordre, chèque) ne relèvent d'aucun Acte uniforme "
             "OHADA ; en Côte d'Ivoire, ils relèvent notamment de la réglementation UEMOA, non intégrée ici."},
    {"theme": "Contrats bancaires, crédit-bail et domiciliation",
     "maroc": [("Code de commerce", r"CREDIT|BANCAIRES|DOMICILIATION")], "ohada": None,
     "note": "Aucun Acte uniforme OHADA ne traite ces contrats : ils relèvent du droit national ou de l'UEMOA."},
    {"theme": "Juridictions de commerce, recouvrement et voies d'exécution",
     "maroc": [("Loi 53-95", None)], "ohada": [("AUPSRVE", None)]},
    {"theme": "Arbitrage et médiation",
     "maroc": None, "ohada": [("(AUA)", None), ("(AUM)", None)],
     "note": "La loi marocaine n° 95-17 sur l'arbitrage et la médiation n'est pas encore intégrée dans la bibliothèque."},
    {"theme": "Droit comptable",
     "maroc": None, "ohada": [("AUDCIF", None), ("SYCEBNL", None)],
     "note": "Le droit comptable marocain n'est pas intégré dans la bibliothèque."},
    {"theme": "Sociétés coopératives",
     "maroc": None, "ohada": [("AUSCOOP", None)],
     "note": "La loi marocaine sur les coopératives n'est pas intégrée dans la bibliothèque."},
]


def _correspond(p, entrees):
    return any(frag in p["label"] and (rx is None or re.search(rx, p["contexte"], re.I)) for frag, rx in entrees)


def theme_equivalence(p, source):
    for t in EQUIVALENCES:
        if t.get(source) and _correspond(p, t[source]):
            return t
    return None


def equivalents(p, source, passages, index, n=5):
    """Articles de l'autre système qui traitent le même point que le passage p."""
    cible = "ohada" if source == "maroc" else "maroc"
    theme = theme_equivalence(p, source)
    if theme is not None and theme.get(cible) is None:
        return theme, []
    if theme is not None:
        candidats = {i for i, q in enumerate(passages) if _correspond(q, theme[cible])}
    else:
        pays_cible = "OHADA" if cible == "ohada" else "Maroc"
        candidats = {i for i, q in enumerate(passages) if q["pays"] == pays_cible and q["domaine"] == "affaires"}
    corps = re.sub(r"^\W*Article\s+\S+", "", p["texte"])[:1400]
    mots = list(dict.fromkeys(tokeniser(p["contexte"] + " " + corps)))[:90]
    scores = index.get_scores(mots)
    resultats, vus = [], set()
    for i in sorted(candidats, key=lambda i: scores[i], reverse=True):
        cle = (passages[i]["label"], passages[i]["num"])
        if cle in vus or not passages[i]["num"]:
            continue
        vus.add(cle)
        resultats.append((passages[i], scores[i]))
        if len(resultats) >= n:
            break
    return theme, resultats


# ------------------------------------------------------------------ rédaction de contrats

PARTIES = [("p1_nom", "Nom ou dénomination", "ex. Société Atlas Digital SARL"),
           ("p1_info", "Forme, siège ou adresse, représentant, n° RC ou CIN",
            "ex. SARL au capital de 100 000 DH, siège à Kénitra, RC 12345, représentée par M. X, gérant"),
           ("p2_nom", "Nom ou dénomination", "ex. M. Youssef Benali"),
           ("p2_info", "Forme, siège ou adresse, représentant, n° RC ou CIN", "ex. consultant indépendant, Rabat")]

# Pour chaque contrat : rôles des parties, champs propres, et articles d'ancrage par droit applicable.
# « socle » = articles toujours fournis (numéros vérifiés dans la bibliothèque) ;
# « reserve » = plages d'articles parmi lesquelles le moteur choisit les plus pertinents.
CONTRATS = {
    "Contrat de prestation de services": {
        "roles": ("Le client", "Le prestataire"),
        "champs": [("objet", "Objet de la mission", "area", "ex. refonte du site internet et formation de l'équipe"),
                   ("livrables", "Livrables et délais", "area", "ex. maquette sous 15 jours, mise en ligne sous 45 jours"),
                   ("duree", "Durée ou date de fin", "text", "ex. 3 mois à compter de la signature"),
                   ("prix", "Prix et modalités de paiement", "text", "ex. 30 000 DH HT, 40 % à la commande, solde à la livraison"),
                   ("donnees", "Le prestataire traite des données personnelles pour le client", "check", ""),
                   ("litiges", "Règlement des litiges", "choice",
                    ["Tribunal compétent", "Médiation, puis tribunal", "Médiation, puis arbitrage"])],
        "requete": "louage ouvrage services prix exécution délai responsabilité inexécution force majeure résiliation",
        "droits": {
            "Maroc": {"socle": [("DOC", ["230", "723", "724", "263", "264", "268", "269"])],
                      "reserve": [("DOC", (723, 780))],
                      "donnees": [("Loi 09-08", ["23", "25", "26"])]},
            "Côte d'Ivoire (OHADA)": {"socle": [("(AUM)", ["2"]), ("(AUA)", ["2", "3-1", "4"])], "reserve": [],
                                      "donnees": [("Loi 2013-450", ["39", "40", "41"])],
                                      "note": "Le droit commun ivoirien des contrats (Code civil) n'est pas intégré dans "
                                              "la bibliothèque : les clauses générales sont rédigées sans référence "
                                              "légale et doivent être vérifiées."},
        },
    },
    "Accord de confidentialité (NDA)": {
        "roles": ("La partie divulgatrice", "La partie destinataire"),
        "champs": [("sens", "Type d'accord", "choice", ["Unilatéral", "Mutuel (les deux parties échangent)"]),
                   ("projet", "Projet ou contexte des échanges", "area", "ex. étude d'un partenariat commercial"),
                   ("duree_conf", "Durée de l'obligation de confidentialité", "text", "ex. 5 ans après la fin des échanges"),
                   ("penalite", "Pénalité forfaitaire en cas de violation (facultatif)", "text", "ex. 50 000 DH"),
                   ("donnees", "Les informations échangées comprennent des données personnelles", "check", ""),
                   ("litiges", "Règlement des litiges", "choice",
                    ["Tribunal compétent", "Médiation, puis tribunal", "Médiation, puis arbitrage"])],
        "requete": "obligation contractuelle inexécution dommages intérêts confidentialité secret",
        "droits": {
            "Maroc": {"socle": [("DOC", ["230", "263", "264"])], "reserve": [],
                      "donnees": [("Loi 09-08", ["23", "25", "26"])]},
            "Côte d'Ivoire (OHADA)": {"socle": [("(AUM)", ["2"]), ("(AUA)", ["3-1", "4"])], "reserve": [],
                                      "donnees": [("Loi 2013-450", ["39", "40", "41"])],
                                      "note": "Le droit commun ivoirien des contrats (Code civil) n'est pas intégré dans "
                                              "la bibliothèque : les clauses générales doivent être vérifiées."},
        },
    },
    "Statuts de SARL": {
        "roles": None,
        "champs": [("denomination", "Dénomination sociale", "text", "ex. Atlas Digital"),
                   ("objet", "Objet social", "area", "ex. conseil en transformation numérique, formation"),
                   ("siege", "Siège social", "text", "ex. 12, avenue Mohammed V, Kénitra"),
                   ("capital", "Capital social et valeur d'une part", "text", "ex. 100 000 DH, parts de 100 DH"),
                   ("associes", "Associés et apports (une ligne par associé)", "area",
                    "ex. Amina Alaoui — 60 000 DH en numéraire — 600 parts\nKarim Idrissi — 40 000 DH en numéraire — 400 parts"),
                   ("gerance", "Gérant(s) et durée du mandat", "text", "ex. Amina Alaoui, pour une durée indéterminée"),
                   ("duree_soc", "Durée de la société", "text", "99 ans"),
                   ("cession", "Cession de parts à des tiers", "choice",
                    ["Avec agrément des associés", "Libre entre associés, agrément pour les tiers"])],
        "requete": "statuts capital parts sociales apports gérance gérant assemblée décisions cession agrément",
        "droits": {
            "Maroc": {"socle": [("Loi 5-96", ["44", "45", "46", "50", "51", "53", "58", "62", "71"])],
                      "reserve": [("Loi 5-96", (44, 87))],
                      "note": "La loi n° 5-96 de la bibliothèque est la version d'origine de 1997 : elle a été modifiée "
                              "depuis, notamment sur le capital minimum. Vérifiez chaque règle dans la version en vigueur."},
            "Côte d'Ivoire (OHADA)": {"socle": [("AUSCGIE", ["10", "13", "309", "310", "311", "312", "313", "317",
                                                             "318", "323", "324", "334"])],
                                      "reserve": [("AUSCGIE", (309, 384))]},
        },
    },
    "Bail commercial": {
        "roles": ("Le bailleur", "Le preneur"),
        "champs": [("local", "Adresse et description du local", "area", "ex. local de 80 m² au rez-de-chaussée, 5 rue X, Casablanca"),
                   ("activite", "Activité exercée dans les lieux", "text", "ex. restauration rapide"),
                   ("duree", "Durée du bail et date d'effet", "text", "ex. 3 ans à compter du 1er janvier 2027"),
                   ("loyer", "Loyer, charges et révision", "text", "ex. 8 000 DH par mois, charges en sus"),
                   ("depot", "Dépôt de garantie", "text", "ex. deux mois de loyer"),
                   ("travaux", "Répartition des travaux et réparations", "choice",
                    ["Selon la loi (grosses réparations au bailleur)", "À préciser dans le contrat"])],
        "requete": "bail écrit durée loyer révision renouvellement réparations résiliation congé indemnité",
        "droits": {
            "Maroc": {"socle": [("Loi 49-16", ["premier", "3", "4", "5", "6", "7"])],
                      "reserve": [("Loi 49-16", (1, 38))]},
            "Côte d'Ivoire (OHADA)": {"socle": [("AUDCG", ["101", "103", "104", "105", "106", "112", "123", "133"])],
                                      "reserve": [("AUDCG", (101, 134))]},
        },
    },
}

REDACTION = """Tu es juriste rédacteur de contrats (droit marocain et droit OHADA). Rédige un PROJET d'acte complet, \
prêt à être relu par un professionnel.

Règles :
1. Rédige en français juridique clair, en articles numérotés (« Article 1 – Objet », « Article 2 – … »), avec un \
titre, la désignation des parties, un préambule court si utile, et un bloc de signatures (lieu, date, nombre \
d'exemplaires).
2. Utilise les informations fournies. Pour toute information manquante, écris [À COMPLÉTER : ce qu'il faut \
indiquer]. N'invente aucun nom, montant, date, numéro ou adresse.
3. Appuie les clauses imposées ou encadrées par la loi sur les articles fournis, en les citant entre parenthèses, \
par exemple (art. 3 de la loi n° 49-16). Ne cite AUCUN article qui ne figure pas dans les extraits. N'invente \
aucune règle légale (délai, montant minimum, majorité) qui n'y figure pas.
4. Inclus les clauses usuelles adaptées au contrat (durée, prix, obligations, responsabilité, force majeure, \
confidentialité, résiliation, règlement des litiges, droit applicable, élection de domicile).
5. Après le contrat, écris exactement la ligne ===VIGILANCE=== puis 3 à 6 points de vigilance pour la personne \
qui va signer (formalités, mentions obligatoires, points à faire vérifier), chacun rattaché à un article fourni \
quand c'est possible.
6. N'ajoute aucun commentaire avant le titre du contrat."""


def extraits_contrat(type_c, droit, avec_donnees, passages, index, articles, n_max=10):
    """Articles d'ancrage d'un contrat : socle vérifié + articles les plus pertinents de la réserve."""
    conf = CONTRATS[type_c]["droits"][droit]
    labels = list(articles)

    def label_de(frag):
        return next((lab for lab in labels if frag in lab), None)

    choisis = []

    def ajouter(lab, num):
        lst = dict(articles.get(lab, []))
        if num in lst and (lab, num) not in [(p["label"], p["num"]) for p in choisis]:
            idx = lst[num]
            p = dict(passages[idx[0]])
            p["texte"] = p["texte"] + "".join(passages[i]["texte"][150:] for i in idx[1:])
            choisis.append(p)

    for frag, nums in conf["socle"] + (conf.get("donnees", []) if avec_donnees else []):
        lab = label_de(frag)
        for num in nums:
            ajouter(lab, num)
    scores = index.get_scores(tokeniser(CONTRATS[type_c]["requete"]))
    for frag, (debut, fin) in conf.get("reserve", []):
        lab = label_de(frag)
        cands = [(num, idx) for num, idx in articles.get(lab, []) if debut <= cle_article(num)[0] <= fin]
        for num, idx in sorted(cands, key=lambda c: max(scores[i] for i in c[1]), reverse=True):
            if len(choisis) >= n_max:
                break
            ajouter(lab, num)
    return choisis[:n_max + 3]


def demande_contrat(type_c, droit, valeurs):
    conf = CONTRATS[type_c]
    lignes = [f"Contrat demandé : {type_c}", f"Droit applicable : {droit}"]
    if conf["roles"]:
        for (cle, _, _), role in zip([PARTIES[0], PARTIES[2]], conf["roles"]):
            pref = cle[:2]
            lignes.append(f"{role} : {valeurs.get(pref + '_nom') or '[non renseigné]'} — "
                          f"{valeurs.get(pref + '_info') or '[non renseigné]'}")
    for cle, libelle, genre, _ in conf["champs"]:
        v = valeurs.get(cle)
        if genre == "check":
            v = "oui" if v else "non"
        lignes.append(f"{libelle} : {v or '[non renseigné]'}")
    if valeurs.get("ville"):
        lignes.append(f"Lieu de signature : {valeurs['ville']}")
    if valeurs.get("precisions"):
        lignes.append(f"Précisions du rédacteur : {valeurs['precisions']}")
    return "\n".join(lignes)


def _xml(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def markdown_vers_docx(md, titre_doc="Projet de contrat"):
    """Convertit le markdown simple du contrat en fichier Word (.docx), sans dépendance externe."""
    import io
    import zipfile

    def runs(texte):
        out = []
        for k, morceau in enumerate(re.split(r"\*\*", texte)):
            if not morceau:
                continue
            gras = "<w:b/>" if k % 2 else ""
            for bout in re.split(r"(\[À COMPLÉTER[^\]]*\])", morceau):
                if not bout:
                    continue
                hl = "<w:highlight w:val=\"yellow\"/>" if bout.startswith("[À COMPLÉTER") else ""
                out.append(f"<w:r><w:rPr>{gras}{hl}</w:rPr><w:t xml:space=\"preserve\">{_xml(bout)}</w:t></w:r>")
        return "".join(out)

    paras = []
    for ligne in md.replace("\r", "").split("\n"):
        s = ligne.strip()
        if not s:
            continue
        if s in ("---", "***", "___"):
            continue
        style = None
        if s.startswith("# "):
            style, s = "Title", s[2:]
        elif s.startswith("## "):
            style, s = "Heading1", s[3:]
        elif s.startswith("### ") or re.match(r"^\**Article\s+\w+", s):
            style, s = "Heading2", s.lstrip("#").strip()
            s = s.strip("*")
        elif re.match(r"^[-*•]\s+", s):
            s = "• " + re.sub(r"^[-*•]\s+", "", s)
            style = "Liste"
        ppr = f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""
        paras.append(f"<w:p>{ppr}{runs(s)}</w:p>")
    corps = "".join(paras)
    document = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
                f"{corps}<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/><w:pgMar w:top=\"1418\" w:right=\"1418\" "
                "w:bottom=\"1418\" w:left=\"1418\" w:header=\"709\" w:footer=\"709\" w:gutter=\"0\"/></w:sectPr>"
                "</w:body></w:document>")

    def style(sid, nom, taille, gras=False, couleur="1D2433", avant=0, apres=120, centre=False):
        return (f"<w:style w:type=\"paragraph\" w:styleId=\"{sid}\"><w:name w:val=\"{nom}\"/>"
                "<w:basedOn w:val=\"Normal\"/><w:pPr>" + (f"<w:jc w:val=\"center\"/>" if centre else "") +
                f"<w:spacing w:before=\"{avant}\" w:after=\"{apres}\"/><w:keepNext/></w:pPr><w:rPr>"
                + ("<w:b/>" if gras else "") + f"<w:color w:val=\"{couleur}\"/><w:sz w:val=\"{taille}\"/></w:rPr></w:style>")

    styles = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
              "<w:styles xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\">"
              "<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii=\"Calibri\" w:hAnsi=\"Calibri\" w:cs=\"Calibri\"/>"
              "<w:sz w:val=\"22\"/><w:lang w:val=\"fr-FR\"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr>"
              "<w:spacing w:after=\"120\" w:line=\"276\" w:lineRule=\"auto\"/><w:jc w:val=\"both\"/></w:pPr></w:pPrDefault>"
              "</w:docDefaults><w:style w:type=\"paragraph\" w:default=\"1\" w:styleId=\"Normal\"><w:name w:val=\"Normal\"/>"
              "</w:style>"
              + style("Title", "Title", 34, True, "0D1B3D", 0, 320, True)
              + style("Heading1", "heading 1", 26, True, "0D1B3D", 240, 120)
              + style("Heading2", "heading 2", 23, True, "8A6A1F", 200, 80)
              + "<w:style w:type=\"paragraph\" w:styleId=\"Liste\"><w:name w:val=\"Liste\"/><w:basedOn w:val=\"Normal\"/>"
                "<w:pPr><w:ind w:left=\"357\"/><w:spacing w:after=\"60\"/></w:pPr></w:style></w:styles>")
    types = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
             "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
             "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
             "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
             "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document.main+xml\"/><Override PartName=\"/word/styles.xml\" ContentType=\"application/"
             "vnd.openxmlformats-officedocument.wordprocessingml.styles+xml\"/><Override PartName=\"/docProps/core.xml\" "
             "ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/></Types>")
    rels = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
            "officeDocument\" Target=\"word/document.xml\"/><Relationship Id=\"rId2\" Type=\"http://schemas."
            "openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>"
            "</Relationships>")
    doc_rels = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
                "styles\" Target=\"styles.xml\"/></Relationships>")
    core = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" "
            "xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:title>" + _xml(titre_doc) + "</dc:title>"
            "<dc:creator>JCKouassi · Assistant juridique</dc:creator></cp:coreProperties>")
    tampon = io.BytesIO()
    with zipfile.ZipFile(tampon, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("word/document.xml", document)
        z.writestr("word/_rels/document.xml.rels", doc_rels)
        z.writestr("word/styles.xml", styles)
        z.writestr("docProps/core.xml", core)
    return tampon.getvalue()


# ------------------------------------------------------------------ interface

st.set_page_config(page_title="JCKouassi · Assistant juridique",
                   page_icon=str(LOGO_ICONE) if LOGO_ICONE.exists() else "⚖️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');
:root { --nuit:#070c1a; --nuit2:#0f1a33; --nuit3:#16244a; --bord:#243456; --or:#d6a94a; --or-clair:#f3d08a;
        --bleu:#3b6ef6; --bleu-clair:#9fc0ff; --texte:#e7edf8; --doux:#a9b5cc; }
html, body, .stMarkdown, p, li, label, textarea, input { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, h4, .jck-titre { font-family: 'Sora', 'Inter', sans-serif !important; color: var(--texte); letter-spacing: -.01em; }
h3 { color: var(--or-clair) !important; }
.stApp { background: radial-gradient(1200px 600px at 75% -10%, rgba(59,110,246,.20), transparent 60%),
                     radial-gradient(900px 500px at -10% 110%, rgba(214,169,74,.07), transparent 60%), var(--nuit); }
.block-container { padding-top: 1.2rem; max-width: 1180px; }

/* Bandeau de marque */
.jck-hero { position: relative; overflow: hidden; display: flex; align-items: center; gap: 26px;
            padding: 26px 30px; margin-bottom: 14px; border-radius: 20px; border: 1px solid var(--bord);
            background: radial-gradient(600px 260px at 85% 0%, rgba(59,110,246,.35), transparent 70%),
                        linear-gradient(135deg, #0b1430 0%, #070c1a 100%); }
.jck-hero::after { content: ""; position: absolute; left: 30px; right: 30px; bottom: 0; height: 2px;
                   background: linear-gradient(90deg, transparent, var(--or), var(--or-clair), var(--or), transparent); }
.jck-hero img { width: 104px; height: 104px; flex-shrink: 0; filter: drop-shadow(0 0 22px rgba(59,110,246,.45)); }
.jck-marque { font-family: 'Sora', sans-serif; font-weight: 600; font-size: .78rem; letter-spacing: .38em;
              color: var(--or-clair); text-transform: uppercase; }
.jck-titre { font-size: 2.1rem; font-weight: 800; line-height: 1.1; margin: 4px 0 6px; color: #fff; }
.jck-sous { color: var(--doux); font-size: .96rem; line-height: 1.5; }
.jck-sous b { color: var(--or-clair); font-weight: 600; }
.jck-chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.jck-chip { font-size: .78rem; padding: 4px 11px; border-radius: 999px; border: 1px solid var(--bord);
            color: var(--bleu-clair); background: rgba(59,110,246,.10); }

/* Barre latérale */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b1430 0%, #070c1a 100%);
                                   border-right: 1px solid var(--bord); }
section[data-testid="stSidebar"] h4 { color: var(--or-clair); font-size: .9rem; letter-spacing: .08em; text-transform: uppercase; }

/* Onglets */
div[data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--bord); }
button[data-baseweb="tab"] { font-family: 'Sora', sans-serif; font-weight: 600; color: var(--doux);
                             padding: 10px 16px; border-radius: 10px 10px 0 0; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--or-clair); background: rgba(214,169,74,.08); }
div[data-baseweb="tab-highlight"] { background-color: var(--or) !important; }

/* Boutons */
.stButton button, .stFormSubmitButton button { border-radius: 12px; border: 1px solid var(--bord);
    background: var(--nuit2); color: var(--texte); transition: border-color .15s, background .15s; text-align: left; }
.stButton button:hover { border-color: var(--or); background: var(--nuit3); color: #fff; }
.stFormSubmitButton button, .stButton button[kind="primary"] { background: linear-gradient(90deg, #c9972f, #f3d08a);
    color: #0b1220; border: none; font-weight: 700; }
.stFormSubmitButton button:hover { filter: brightness(1.06); color: #0b1220; }

/* Saisie */
.stTextArea textarea, .stTextInput input { background: var(--nuit2) !important; border: 1px solid var(--bord) !important;
    border-radius: 12px !important; color: var(--texte) !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--or) !important; box-shadow: 0 0 0 1px var(--or) !important; }

/* Messages et sources */
div[data-testid="stChatMessage"] { background: rgba(15,26,51,.65); border: 1px solid var(--bord); border-radius: 16px; padding: 14px 16px; }
div[data-testid="stExpander"] details { border: 1px solid var(--bord); border-radius: 12px; background: rgba(15,26,51,.7); }
div[data-testid="stExpander"] summary { color: var(--bleu-clair); font-weight: 500; }
div[data-testid="stExpander"] summary:hover { color: var(--or-clair); }
.jck-ctx { color: var(--doux); font-size: .82rem; margin-bottom: 6px; border-left: 2px solid var(--or); padding-left: 10px; }
.jck-label { color: var(--or-clair); font-family: 'Sora', sans-serif; font-weight: 600; font-size: .95rem; margin: 10px 0 6px; }

/* Pied de page */
.jck-pied { margin-top: 2.6rem; padding: 18px 0 6px; border-top: 1px solid var(--bord); color: var(--doux);
            font-size: .85rem; text-align: center; }
.jck-pied b { color: var(--or-clair); }
.jck-pied span { color: var(--bleu-clair); letter-spacing: .3em; font-family: 'Sora', sans-serif; font-size: .72rem; }

/* ---------- Refonte : chiffres clés, onglets en pilules, cartes ---------- */
.jck-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 0 0 18px; }
.jck-stat { position: relative; overflow: hidden; padding: 14px 16px; border-radius: 16px; border: 1px solid var(--bord);
            background: linear-gradient(160deg, rgba(22,36,74,.75), rgba(11,20,48,.75)); backdrop-filter: blur(6px); }
.jck-stat::before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
                    background: linear-gradient(180deg, var(--or-clair), var(--or)); }
.jck-stat b { display: block; font-family: 'Sora', sans-serif; font-size: 1.55rem; font-weight: 800; color: #fff; line-height: 1.1; }
.jck-stat span { color: var(--doux); font-size: .8rem; letter-spacing: .02em; }
@media (max-width: 760px) { .jck-stats { grid-template-columns: repeat(2, 1fr); }
                            .jck-hero { flex-direction: column; align-items: flex-start; padding: 22px; }
                            .jck-titre { font-size: 1.7rem; } }

div[data-baseweb="tab-list"] { gap: 6px; border-bottom: none !important; padding: 6px; border-radius: 16px;
    background: rgba(15,26,51,.7); border: 1px solid var(--bord); flex-wrap: wrap; }
button[data-baseweb="tab"] { border-radius: 11px !important; padding: 8px 14px !important; }
button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(90deg, rgba(201,151,47,.22), rgba(243,208,138,.12)) !important;
    box-shadow: inset 0 0 0 1px rgba(214,169,74,.45); }
button[data-baseweb="tab"]:hover { color: #fff; background: rgba(59,110,246,.10); }
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }
div[data-baseweb="tab-panel"] { padding-top: 18px; }

/* Étapes « comment ça marche » */
.jck-etapes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 18px; }
.jck-etape { padding: 14px 16px; border-radius: 14px; border: 1px dashed var(--bord); background: rgba(15,26,51,.45); }
.jck-etape i { font-style: normal; display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 50%;
               background: rgba(214,169,74,.15); color: var(--or-clair); font-family: 'Sora', sans-serif; font-weight: 700;
               font-size: .85rem; margin-right: 8px; }
.jck-etape b { color: #fff; font-weight: 600; }
.jck-etape p { color: var(--doux); font-size: .84rem; margin: 6px 0 0; }
@media (max-width: 760px) { .jck-etapes { grid-template-columns: 1fr; } }

/* Cartes à bordure (formulaires, résultats) */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; border-color: var(--bord) !important;
    background: linear-gradient(160deg, rgba(15,26,51,.78), rgba(7,12,26,.6)); }
div[data-testid="stForm"] { border: 1px solid var(--bord) !important; border-radius: 16px !important;
    background: linear-gradient(160deg, rgba(15,26,51,.78), rgba(7,12,26,.6)); padding: 18px 18px 8px !important; }
.jck-section { font-family: 'Sora', sans-serif; font-weight: 700; color: #fff; font-size: 1.05rem; margin: 2px 0 8px;
               display: flex; align-items: center; gap: 10px; }
.jck-section::before { content: ""; width: 18px; height: 2px; background: var(--or); border-radius: 2px; }
.jck-badge { display: inline-block; font-size: .72rem; padding: 3px 9px; border-radius: 999px; margin: 0 6px 6px 0;
             border: 1px solid rgba(214,169,74,.45); color: var(--or-clair); background: rgba(214,169,74,.08); }
.st-key-papier_contrat, .st-key-papier_flux { background: #fbfaf6; border-radius: 14px; padding: 28px 34px 18px;
    box-shadow: 0 18px 50px rgba(0,0,0,.45); border-top: 4px solid var(--or); }
.st-key-papier_contrat *, .st-key-papier_flux * { color: #1d2433; }
.st-key-papier_contrat h1, .st-key-papier_contrat h2, .st-key-papier_contrat h3,
.st-key-papier_flux h1, .st-key-papier_flux h2, .st-key-papier_flux h3 { color: #0d1b3d !important; font-family: 'Sora', sans-serif !important; }
.st-key-papier_contrat h1, .st-key-papier_flux h1 { text-align: center; font-size: 1.45rem; }
.st-key-papier_contrat h2, .st-key-papier_flux h2 { font-size: 1.05rem; margin-top: 1rem; }
.st-key-papier_contrat h3, .st-key-papier_flux h3 { font-size: .98rem; color: #8a6a1f !important; }
@media (max-width: 760px) { .st-key-papier_contrat, .st-key-papier_flux { padding: 18px 16px 10px; } }

/* Barre latérale épurée */
section[data-testid="stSidebar"] .stRadio label, section[data-testid="stSidebar"] .stMultiSelect label { color: var(--doux); }
.jck-side-carte { margin-top: 10px; padding: 12px 14px; border-radius: 14px; border: 1px solid var(--bord);
                  background: rgba(15,26,51,.6); font-size: .82rem; color: var(--doux); line-height: 1.5; }
.jck-side-carte b { color: var(--or-clair); }

/* ---------- Animations ---------- */
@keyframes jck-monte { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes jck-apparait { from { opacity: 0; } to { opacity: 1; } }
@keyframes jck-flotte { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
@keyframes jck-halo { 0%, 100% { filter: drop-shadow(0 0 16px rgba(59,110,246,.35)); }
                      50% { filter: drop-shadow(0 0 30px rgba(214,169,74,.55)); } }
@keyframes jck-balaye { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes jck-aurore { 0% { transform: translate(0, 0) scale(1); } 50% { transform: translate(-4%, 3%) scale(1.08); }
                        100% { transform: translate(0, 0) scale(1); } }
@keyframes jck-pop { 0% { opacity: 0; transform: scale(.8); } 70% { transform: scale(1.06); } 100% { opacity: 1; transform: none; } }
@keyframes jck-reflet { 0% { left: -60%; } 60%, 100% { left: 130%; } }
@keyframes jck-glisse { from { opacity: 0; transform: translateX(-12px); } to { opacity: 1; transform: none; } }

/* Bandeau */
.jck-hero { animation: jck-monte .7s cubic-bezier(.2,.7,.2,1) both; }
.jck-hero > * { position: relative; z-index: 1; }
.jck-hero::before { content: ""; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(380px 200px at 85% 0%, rgba(214,169,74,.18), transparent 70%);
    animation: jck-aurore 14s ease-in-out infinite; }
.jck-hero::after { background-size: 200% 100%; animation: jck-balaye 5s linear infinite; }
.jck-hero img { animation: jck-flotte 6s ease-in-out infinite, jck-halo 6s ease-in-out infinite; }
.jck-marque { animation: jck-apparait .9s .15s both; }
.jck-titre { background: linear-gradient(90deg, #ffffff 0%, #ffffff 40%, var(--or-clair) 50%, #ffffff 60%, #ffffff 100%);
    background-size: 250% 100%; -webkit-background-clip: text; background-clip: text; color: transparent;
    animation: jck-monte .8s .2s both, jck-balaye 7s 1s linear infinite; }
.jck-sous { animation: jck-apparait 1s .35s both; }
.jck-chip { animation: jck-pop .5s both; transition: transform .2s, border-color .2s, color .2s; }
.jck-chip:nth-child(1) { animation-delay: .45s; } .jck-chip:nth-child(2) { animation-delay: .55s; }
.jck-chip:nth-child(3) { animation-delay: .65s; } .jck-chip:nth-child(4) { animation-delay: .75s; }
.jck-chip:nth-child(5) { animation-delay: .85s; }
.jck-chip:hover { transform: translateY(-2px); border-color: var(--or); color: var(--or-clair); }
.jck-stat, .jck-etape { animation: jck-monte .6s cubic-bezier(.2,.7,.2,1) both; transition: transform .2s, border-color .2s; }
.jck-stat:nth-child(2), .jck-etape:nth-child(2) { animation-delay: .1s; }
.jck-stat:nth-child(3), .jck-etape:nth-child(3) { animation-delay: .2s; }
.jck-stat:nth-child(4) { animation-delay: .3s; }
.jck-stat:hover, .jck-etape:hover { transform: translateY(-3px); border-color: rgba(214,169,74,.6); }
.st-key-papier_contrat { animation: jck-monte .6s both; }

/* Onglets : le contenu apparaît à chaque changement */
div[data-baseweb="tab-panel"] { animation: jck-monte .45s cubic-bezier(.2,.7,.2,1) both; }
button[data-baseweb="tab"] { transition: color .2s, background .2s; }
div[data-baseweb="tab-highlight"] { transition: all .35s cubic-bezier(.2,.7,.2,1) !important; }

/* Boutons : léger soulèvement, reflet doré sur le bouton principal */
.stButton button, .stFormSubmitButton button { transition: transform .18s, box-shadow .18s, border-color .15s, background .15s; }
.stButton button:hover, .stFormSubmitButton button:hover { transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(0,0,0,.35), 0 0 0 1px rgba(214,169,74,.25); }
.stButton button:active, .stFormSubmitButton button:active { transform: translateY(0); }
.stFormSubmitButton button, .stButton button[kind="primary"] { position: relative; overflow: hidden; }
.stFormSubmitButton button::after, .stButton button[kind="primary"]::after { content: ""; position: absolute; top: 0;
    left: -60%; width: 40%; height: 100%; transform: skewX(-20deg);
    background: linear-gradient(90deg, transparent, rgba(255,255,255,.55), transparent);
    animation: jck-reflet 3.5s ease-in-out infinite; }

/* Saisie : halo à la mise au point */
.stTextArea textarea, .stTextInput input { transition: border-color .2s, box-shadow .3s !important; }
.stTextArea textarea:focus, .stTextInput input:focus { box-shadow: 0 0 0 1px var(--or), 0 0 18px rgba(214,169,74,.25) !important; }

/* Réponses, sources, encadrés */
div[data-testid="stChatMessage"] { animation: jck-monte .5s cubic-bezier(.2,.7,.2,1) both; }
div[data-testid="stExpander"] { animation: jck-glisse .45s both; }
div[data-testid="stExpander"] details { transition: border-color .2s, background .2s; }
div[data-testid="stExpander"] details:hover { border-color: rgba(214,169,74,.55); }
div[data-testid="stAlert"] { animation: jck-monte .4s both; }
section[data-testid="stSidebar"] img { animation: jck-apparait .8s both; }

/* Respect du réglage « réduire les animations » */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .jck-titre { color: #fff; background: none; }
}
</style>
""", unsafe_allow_html=True)

passages, index, textes, articles, plans = charger_corpus()
espaces_dispo = [e for e in ESPACES if any(t["pays"] == e for t in textes)]
labels_textes = [t["label"] for t in textes if t["label"] in articles]

import base64

with st.sidebar:
    if LOGO_H.exists():
        logo_b64 = base64.b64encode(LOGO_H.read_bytes()).decode()
        st.markdown(f"<img src='data:image/png;base64,{logo_b64}' alt='JCKouassi, Juriste 2.0' "
                    f"style='width:100%;border-radius:14px;margin:4px 0 14px;border:1px solid #243456'>",
                    unsafe_allow_html=True)
    st.markdown("#### Périmètre de recherche")
    choix_domaine = st.radio("Domaine", ["Les deux", "Données personnelles", "Droit des affaires"])
    domaines = {"Les deux": ["donnees", "affaires"], "Données personnelles": ["donnees"],
                "Droit des affaires": ["affaires"]}[choix_domaine]
    choix_espaces = st.multiselect("Pays et espaces juridiques", espaces_dispo, default=espaces_dispo)
    st.markdown("#### Réponse")
    format_rep = st.radio("Format", list(FORMATS), horizontal=True)
    st.divider()
    if st.button("Nouvelle conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    nb_articles = sum(t["articles"] for t in textes)
    st.markdown(f"<div class='jck-side-carte'><b>{len(textes)} textes</b> · "
                f"{nb_articles:,} articles indexés".replace(",", " ") +
                f"<br>Une initiative de <b>{AUTEUR}</b>, juriste.</div>", unsafe_allow_html=True)

# En-tête : bandeau de marque
icone_b64 = base64.b64encode(LOGO_ICONE.read_bytes()).decode() if LOGO_ICONE.exists() else ""
st.markdown(
    f"""<div class='jck-hero'>
      {f"<img src='data:image/png;base64,{icone_b64}' alt='Emblème JCKouassi'>" if icone_b64 else ""}
      <div>
        <div class='jck-marque'>JCKouassi · Juriste 2.0</div>
        <div class='jck-titre'>Assistant juridique</div>
        <div class='jck-sous'>Données personnelles et droit des affaires, réponses sourcées article par article.<br>
          Une initiative de <b>{AUTEUR}</b>.</div>
        <div class='jck-chips'><span class='jck-chip'>Côte d'Ivoire</span><span class='jck-chip'>Maroc</span>
          <span class='jck-chip'>OHADA</span><span class='jck-chip'>CEDEAO</span><span class='jck-chip'>Union africaine</span></div>
      </div></div>""",
    unsafe_allow_html=True)

nb_art_total = sum(t["articles"] for t in textes)
st.markdown(
    "<div class='jck-stats'>"
    f"<div class='jck-stat'><b>{len(textes)}</b><span>textes officiels</span></div>"
    f"<div class='jck-stat'><b>{nb_art_total:,}</b><span>articles consultables</span></div>".replace(",", " ") +
    f"<div class='jck-stat'><b>{len(espaces_dispo)}</b><span>pays et espaces juridiques</span></div>"
    f"<div class='jck-stat'><b>{len(CONTRATS)}</b><span>modèles de contrats guidés</span></div>"
    "</div>", unsafe_allow_html=True)

onglet_q, onglet_eq, onglet_ctr, onglet_art, onglet_biblio, onglet_apropos = st.tabs(
    [":material/forum: Question juridique", ":material/compare_arrows: Équivalences Maroc ↔ OHADA",
     ":material/edit_document: Rédiger un contrat", ":material/article: Consulter un article",
     ":material/local_library: Bibliothèque", ":material/info: À propos"])


def afficher_sources(sources, cle):
    for n, s in enumerate(sources):
        with st.expander(f"[{n + 1}] {s['ref']}"):
            if s.get("contexte"):
                st.markdown(f"<div class='jck-ctx'>{s['contexte']}</div>", unsafe_allow_html=True)
            st.markdown(s["texte"].replace("\n", "  \n"))
            st.code(s["ref"], language=None)


# --- Onglet 1 : questions
THEMES = {
    "Créer une société": ("créer une société", "la création d'une société", [
        "Quelles sont les étapes pour créer une SARL en Côte d'Ivoire ?",
        "Quelles sont les conditions de constitution d'une SAS en droit OHADA ?",
        "Comment créer une SARL au Maroc ?"]),
    "Bail commercial": ("conclure ou renouveler un bail commercial", "le bail commercial", [
        "Quelle est la durée d'un bail commercial en droit OHADA et comment le renouveler ?",
        "Dans quels cas le bailleur peut-il refuser le renouvellement du bail commercial au Maroc ?"]),
    "Recouvrer une créance": ("recouvrer une créance", "le recouvrement de créances", [
        "Comment fonctionne la procédure d'injonction de payer dans l'espace OHADA ?",
        "Quelles saisies un créancier peut-il pratiquer contre son débiteur en droit OHADA ?"]),
    "Garanties et sûretés": ("constituer une garantie (sûreté)", "les sûretés", [
        "Comment constituer une hypothèque en droit OHADA ?",
        "Quelles sont les règles du cautionnement en droit OHADA ?"]),
    "Entreprise en difficulté": ("traiter les difficultés d'une entreprise", "les procédures collectives", [
        "Quelles sont les conditions d'ouverture du redressement judiciaire en droit OHADA ?",
        "Comment se déroule la procédure de conciliation en droit OHADA ?"]),
    "Données personnelles": ("traiter des données personnelles", "la protection des données personnelles", [
        "Faut-il déclarer un fichier clients à l'autorité de protection, au Maroc et en Côte d'Ivoire ?",
        "Quels sont les droits des personnes concernées en Côte d'Ivoire ?",
        "Peut-on transférer des données personnelles vers l'étranger depuis le Maroc ?"]),
    "Arbitrage et médiation": ("recourir à l'arbitrage ou à la médiation", "l'arbitrage et la médiation", [
        "Comment désigner un arbitre selon l'Acte uniforme relatif au droit de l'arbitrage ?",
        "Comment se déroule une médiation selon l'Acte uniforme relatif à la médiation ?"]),
}
PAYS_FORMULES = {"Côte d'Ivoire (droit OHADA)": "en Côte d'Ivoire", "Maroc": "au Maroc",
                 "Comparer Côte d'Ivoire et Maroc": "en Côte d'Ivoire et au Maroc (comparaison)",
                 "Autre pays de l'OHADA": "dans l'espace OHADA"}
BESOINS = {
    "Les étapes à suivre": "Quelles sont les étapes à suivre pour {v} {p} ?",
    "Les conditions à remplir": "Quelles sont les conditions à remplir pour {v} {p} ?",
    "Les délais à respecter": "Quels sont les délais à respecter pour {v} {p} ?",
    "Les formalités et documents": "Quelles formalités et quels documents faut-il pour {v} {p} ?",
    "Les sanctions prévues": "Quelles sanctions sont prévues en matière de {n} {p} ?",
    "Les droits et obligations": "Quels sont les droits et obligations des parties en matière de {n} {p} ?",
}


def poser(q):
    st.session_state.question_exemple = q


def soumettre():
    texte = st.session_state.get("saisie", "").strip()
    if texte:
        st.session_state.question_a_traiter = texte
        st.session_state.saisie = ""
        st.session_state.pop("suggestion", None)


def composer():
    v, n, _ = THEMES[st.session_state.f_theme]
    q = BESOINS[st.session_state.f_besoin].format(v=v, n=n, p=PAYS_FORMULES[st.session_state.f_pays])
    situation = st.session_state.get("f_situation", "").strip()
    st.session_state.saisie = q + (f" Ma situation : {situation}" if situation else "")


def utiliser_suggestion():
    sug = st.session_state.pop("suggestion", None)
    if sug and sug.get("question_precise"):
        st.session_state.saisie = sug["question_precise"]


def boutons_suivi(suivis, cle):
    if suivis:
        st.markdown("<div class='jck-label'>Pour aller plus loin</div>", unsafe_allow_html=True)
        for j, q in enumerate(suivis):
            st.button(q, key=f"suivi_{cle}_{j}", on_click=poser, args=(q,), width="stretch")


with onglet_q:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "GROQ_API_KEY" not in st.secrets:
        st.error("La clé GROQ_API_KEY n'est pas configurée dans les Secrets de l'application.")
    else:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        modele = st.secrets["GROQ_MODEL"] if "GROQ_MODEL" in st.secrets else DEFAULT_MODEL

        def ameliorer():
            texte = st.session_state.get("saisie", "").strip()
            if texte:
                st.session_state.suggestion = analyser_question(client, texte)

        # Points de départ par thème
        if not st.session_state.messages:
            st.markdown(
                "<div class='jck-etapes'>"
                "<div class='jck-etape'><i>1</i><b>Posez votre question</b><p>Indiquez le pays, votre situation "
                "et ce que vous cherchez : étapes, conditions, délais.</p></div>"
                "<div class='jck-etape'><i>2</i><b>L'assistant cherche les articles</b><p>Il retrouve les articles "
                "pertinents dans les textes officiels intégrés.</p></div>"
                "<div class='jck-etape'><i>3</i><b>Réponse sourcée</b><p>Chaque règle est citée avec son article, "
                "lisible en entier sous la réponse.</p></div></div>", unsafe_allow_html=True)
            st.markdown("<div class='jck-label'>Par où commencer ? Choisissez un thème</div>",
                        unsafe_allow_html=True)
            theme = st.selectbox("Thème", list(THEMES), label_visibility="collapsed", key="theme_depart")
            cols = st.columns(len(THEMES[theme][2]))
            for j, q in enumerate(THEMES[theme][2]):
                cols[j].button(q, key=f"ex_{theme}_{j}", on_click=poser, args=(q,), width="stretch")

        # Historique
        for n, m in enumerate(st.session_state.messages):
            avatar = str(LOGO_ICONE) if m["role"] == "assistant" and LOGO_ICONE.exists() else None
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])
                if m.get("precise"):
                    st.caption(f"Question comprise comme : {m['precise']}")
                if m.get("sources"):
                    st.markdown("<div class='jck-label'>Articles consultés</div>", unsafe_allow_html=True)
                    afficher_sources(m["sources"], f"h{n}")
                if m.get("precisions"):
                    st.caption("Pour une réponse plus ciblée, précisez : " + " · ".join(m["precisions"]))
                boutons_suivi(m.get("suivis"), f"h{n}")

        # Aide à la formulation
        with st.expander("Besoin d'aide pour formuler votre question ?"):
            st.markdown("Une bonne question indique **le pays**, **la situation** (forme de société, qualité "
                        "des parties, montant…) et **ce que vous cherchez** (étapes, conditions, délais, "
                        "sanctions). Composez-la en trois choix :")
            g1, g2, g3 = st.columns(3)
            g1.selectbox("Thème", list(THEMES), key="f_theme")
            g2.selectbox("Pays", list(PAYS_FORMULES), key="f_pays")
            g3.selectbox("Vous cherchez", list(BESOINS), key="f_besoin")
            st.text_input("Votre situation en une phrase (facultatif)", key="f_situation",
                          placeholder="ex. deux associés, capital de 1 million de FCFA, activité de commerce")
            st.button("Composer ma question", on_click=composer)

        # Saisie
        st.text_area("Votre question", key="saisie", height=110, label_visibility="collapsed",
                     placeholder="Posez votre question juridique, ou citez un article : « Que prévoit "
                                 "l'article 309 de l'AUSCGIE ? »")
        b1, b2, _ = st.columns([2, 2, 3])
        b1.button("Rechercher et répondre", type="primary", on_click=soumettre, width="stretch")
        b2.button("Améliorer ma question", on_click=ameliorer, width="stretch",
                  help="Reformule votre question en termes juridiques et signale ce qui manque.")

        sug = st.session_state.get("suggestion")
        if sug:
            with st.container(border=True):
                if sug.get("question_precise"):
                    st.markdown(f"**Version proposée :** {sug['question_precise']}")
                if sug.get("precisions"):
                    st.markdown("**Pour une réponse plus précise, indiquez :**\n" +
                                "\n".join(f"- {x}" for x in sug["precisions"]))
                if not sug.get("question_precise") and not sug.get("precisions"):
                    st.markdown("Votre question est déjà claire : vous pouvez lancer la recherche.")
                s1, s2, _ = st.columns([2, 1, 4])
                if sug.get("question_precise"):
                    s1.button("Utiliser cette version", on_click=utiliser_suggestion, width="stretch")
                s2.button("Ignorer", on_click=lambda: st.session_state.pop("suggestion", None))

        question = st.session_state.pop("question_a_traiter", None) or st.session_state.pop("question_exemple", None)

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            analyse = {}
            if not choix_espaces:
                reponse, trouves = "Sélectionnez au moins un pays ou espace juridique dans le panneau de gauche.", []
                st.warning(reponse)
            else:
                precedente = next((m["content"] for m in reversed(st.session_state.messages[:-1])
                                   if m["role"] == "user"), "")
                with st.spinner("Analyse de la question et recherche des articles…"):
                    plan_dispo = "\n".join(
                        plan_resume(lab, plans) for lab in
                        textes_probables(question, passages, index, domaines, choix_espaces) if lab in plans)
                    analyse = analyser_question(client, question, precedente, plan_dispo)
                    trouves = rechercher(question + " " + analyse.get("question_precise", ""), passages, index,
                                         articles, domaines, choix_espaces, analyse=analyse)
                contexte = "\n\n".join(
                    f"[{n + 1}] {p['ref']} — {p['titre']}\n{p['texte']}" for n, p in enumerate(trouves)
                ) or "(Aucun extrait pertinent trouvé dans le corpus.)"
                concernes = list(dict.fromkeys(p["label"] for p in trouves if p["label"] in plans))[:3]
                if concernes:
                    contexte += "\n\nPlan des textes concernés :\n" + "\n".join(
                        plan_resume(lab, plans) for lab in concernes)
                precise = analyse.get("question_precise", "")
                historique = [{"role": m["role"], "content": m["content"][:1500]}
                              for m in st.session_state.messages[-5:-1]]
                messages = [{"role": "system", "content": SYSTEME + "\n\n" + FORMATS[format_rep]},
                            *historique,
                            {"role": "user", "content": f"Extraits des textes :\n\n{contexte}\n\n"
                                                        f"Question : {question}"
                                                        + (f"\n(Reformulation juridique : {precise})" if precise else "")}]
                with st.chat_message("assistant", avatar=str(LOGO_ICONE) if LOGO_ICONE.exists() else None):
                    try:
                        flux, utilise = flux_modele(client, messages, modele)
                        reponse = ecrire_flux(flux)
                        if utilise != modele:
                            st.caption("Réponse rédigée par le modèle de secours (forte affluence).")
                    except Exception as e:  # noqa: BLE001
                        reponse = message_erreur(e)
                        st.warning(reponse)
                    if precise:
                        st.caption(f"Question comprise comme : {precise}")
                    if trouves:
                        st.markdown("<div class='jck-label'>Articles consultés</div>", unsafe_allow_html=True)
                        afficher_sources(trouves, "dernier")
                    if analyse.get("precisions"):
                        st.caption("Pour une réponse plus ciblée, précisez : " + " · ".join(analyse["precisions"]))
                    boutons_suivi(analyse.get("suivis"), "dernier")
            st.session_state.messages.append({
                "role": "assistant", "content": reponse,
                "sources": [{"ref": s["ref"], "texte": s["texte"], "contexte": s["contexte"]} for s in trouves],
                "precise": analyse.get("question_precise", ""), "precisions": analyse.get("precisions", []),
                "suivis": analyse.get("suivis", [])})

# --- Onglet Équivalences Maroc ↔ OHADA
COMPARAISON = """Tu es juriste comparatiste (droit marocain et droit uniforme OHADA). On te donne un article source
et les articles de l'autre système qui traitent le même point. À partir de ces seuls textes, rédige :
**Correspondance** (quel article correspond le mieux, et pourquoi), **Points communs**, **Différences** (conditions,
délais, montants, majorités, sanctions), **Conséquence pratique** (ce que cela change pour une entreprise qui opère
dans les deux espaces). Cite les références exactes entre crochets. Si aucun article ne correspond vraiment,
dis-le. Termine par une phrase rappelant qu'il s'agit d'une information et non d'un conseil juridique."""

with onglet_eq:
    st.markdown("Choisissez un article de droit marocain ou de droit OHADA : l'assistant retrouve les articles qui "
                "traitent le même point dans l'autre système, puis peut rédiger une comparaison.")
    sens = st.radio("Sens", ["Du droit marocain vers l'OHADA", "De l'OHADA vers le droit marocain"],
                    horizontal=True, label_visibility="collapsed", key="eq_sens")
    source = "maroc" if sens.startswith("Du droit marocain") else "ohada"
    pays_src = "Maroc" if source == "maroc" else "OHADA"
    labels_src = [t["label"] for t in textes
                  if t["pays"] == pays_src and t["domaine"] == "affaires" and t["label"] in articles]
    e1, e2 = st.columns([3, 1])
    texte_eq = e1.selectbox("Texte source", labels_src, key=f"eq_texte_{source}")
    nums_eq = [n for n, _ in articles.get(texte_eq, [])]
    num_eq = e2.selectbox("Article", nums_eq, key=f"eq_num_{texte_eq}") if nums_eq else None
    if num_eq:
        idx_src = dict(articles[texte_eq])[num_eq]
        p_src = dict(passages[idx_src[0]])
        p_src["texte"] = passages[idx_src[0]]["texte"] + "".join(passages[i]["texte"][150:] for i in idx_src[1:])
        theme, equiv = equivalents(p_src, source, passages, index)
        st.markdown(f"<div class='jck-label'>Article source · {texte_eq}, art. {num_eq}</div>",
                    unsafe_allow_html=True)
        with st.expander("Lire l'article source", expanded=True):
            if p_src["contexte"]:
                st.markdown(f"<div class='jck-ctx'>{p_src['contexte']}</div>", unsafe_allow_html=True)
            st.markdown(p_src["texte"].replace("\n", "  \n"))
        if theme:
            st.markdown(f"**Thème :** {theme['theme']}")
        else:
            st.caption("Thème non répertorié dans la table de correspondance : recherche dans tout le droit "
                       + ("OHADA." if source == "maroc" else "marocain des affaires."))
        if theme and theme.get("note") and not equiv:
            st.info(theme["note"])
        if equiv:
            st.markdown("<div class='jck-label'>Articles équivalents probables</div>", unsafe_allow_html=True)
            top = equiv[0][1] or 1
            for q, sc in equiv:
                force = "forte" if sc >= 0.75 * top else ("moyenne" if sc >= 0.5 * top else "faible")
                with st.expander(f"{q['ref']} · correspondance {force}"):
                    if q["contexte"]:
                        st.markdown(f"<div class='jck-ctx'>{q['contexte']}</div>", unsafe_allow_html=True)
                    st.markdown(q["texte"].replace("\n", "  \n"))
                    st.code(q["ref"], language=None)
            st.caption("Correspondances proposées automatiquement à partir du thème et du vocabulaire des "
                       "articles : vérifiez-les avant de les citer.")
            if "GROQ_API_KEY" in st.secrets and st.button("Comparer avec l'IA", type="primary", key="eq_comparer"):
                client_eq = Groq(api_key=st.secrets["GROQ_API_KEY"])
                modele_eq = st.secrets["GROQ_MODEL"] if "GROQ_MODEL" in st.secrets else DEFAULT_MODEL
                contenu = (f"Article source : [{p_src['ref']}]\n{p_src['texte'][:2500]}\n\n"
                           "Articles de l'autre système :\n\n" +
                           "\n\n".join(f"[{q['ref']}]\n{q['texte'][:1500]}" for q, _ in equiv[:4]))
                with st.container(border=True):
                    try:
                        flux, utilise = flux_modele(client_eq, [{"role": "system", "content": COMPARAISON},
                                                                {"role": "user", "content": contenu}], modele_eq)
                        ecrire_flux(flux)
                        if utilise != modele_eq:
                            st.caption("Comparaison rédigée par le modèle de secours (forte affluence).")
                    except Exception as e:  # noqa: BLE001
                        st.warning(message_erreur(e))

# --- Onglet Rédiger un contrat
def surligner(md):
    return re.sub(r"\[(À COMPLÉTER[^\]]*)\]", r":orange-background[\1]", md)


with onglet_ctr:
    st.markdown("<div class='jck-section'>Rédiger un projet de contrat</div>", unsafe_allow_html=True)
    st.markdown("Remplissez le formulaire : l'assistant rédige un projet complet, appuyé sur les articles de la "
                "bibliothèque, avec les informations manquantes surlignées. Le projet se télécharge en Word.")
    k1, k2 = st.columns([3, 2])
    type_c = k1.selectbox("Contrat", list(CONTRATS), key="ctr_type")
    droit = k2.radio("Droit applicable", list(CONTRATS[type_c]["droits"]), horizontal=True, key="ctr_droit")
    conf_c = CONTRATS[type_c]
    conf_d = conf_c["droits"][droit]
    bases = list(dict.fromkeys(f for f, _ in conf_d["socle"] + conf_d.get("donnees", [])))
    noms_bases = [next((t["label"] for t in textes if f in t["label"]), f) for f in bases]
    st.markdown("".join(f"<span class='jck-badge'>{_xml(n.split(' (')[0] if n.startswith('Acte') else n)}</span>"
                        for n in noms_bases), unsafe_allow_html=True)
    if conf_d.get("note"):
        st.info(conf_d["note"], icon=":material/info:")

    with st.form(f"form_{type_c}"):
        valeurs = {}
        if conf_c["roles"]:
            colonnes = st.columns(2)
            for col, role, (cn, ln, pn), (ci, li, pi) in zip(colonnes, conf_c["roles"],
                                                             [PARTIES[0], PARTIES[2]], [PARTIES[1], PARTIES[3]]):
                col.markdown(f"<div class='jck-label'>{role}</div>", unsafe_allow_html=True)
                valeurs[cn] = col.text_input(ln, placeholder=pn, key=f"{type_c}_{cn}")
                valeurs[ci] = col.text_area(li, placeholder=pi, height=80, key=f"{type_c}_{ci}")
        st.markdown("<div class='jck-label'>Contenu</div>", unsafe_allow_html=True)
        for cle, libelle, genre, aide in conf_c["champs"]:
            k = f"{type_c}_{cle}"
            if genre == "area":
                valeurs[cle] = st.text_area(libelle, placeholder=aide, height=90, key=k)
            elif genre == "text":
                valeurs[cle] = st.text_input(libelle, placeholder=aide, key=k)
            elif genre == "check":
                valeurs[cle] = st.checkbox(libelle, key=k)
            else:
                valeurs[cle] = st.selectbox(libelle, aide, key=k)
        f1, f2 = st.columns([1, 2])
        valeurs["ville"] = f1.text_input("Lieu de signature", placeholder="ex. Kénitra", key=f"{type_c}_ville")
        valeurs["precisions"] = f2.text_input("Autres précisions (facultatif)", key=f"{type_c}_precisions",
                                              placeholder="ex. clause de non-concurrence de 12 mois, paiement par virement")
        lancer = st.form_submit_button("Rédiger le projet de contrat", type="primary", icon=":material/edit_document:")

    if lancer:
        if "GROQ_API_KEY" not in st.secrets:
            st.error("La clé GROQ_API_KEY n'est pas configurée dans les Secrets de l'application.")
        else:
            client_c = Groq(api_key=st.secrets["GROQ_API_KEY"])
            modele_c = st.secrets["GROQ_MODEL"] if "GROQ_MODEL" in st.secrets else DEFAULT_MODEL
            extraits = extraits_contrat(type_c, droit, bool(valeurs.get("donnees")), passages, index, articles)
            contexte_c = "\n\n".join(
                f"[{p['ref']}]\n{p['texte'][:800]}{'…' if len(p['texte']) > 800 else ''}" for p in extraits)
            if conf_d.get("note"):
                contexte_c += f"\n\nAvertissement : {conf_d['note']}"
            messages_c = [{"role": "system", "content": REDACTION},
                          {"role": "user", "content": f"{demande_contrat(type_c, droit, valeurs)}\n\n"
                                                      f"Articles de référence :\n\n{contexte_c}"}]
            fait = False
            with st.container(key="papier_flux"):
                try:
                    with st.spinner("Rédaction du projet en cours…"):
                        flux, utilise = flux_modele(client_c, messages_c, modele_c)
                    zone = st.empty()
                    texte_c = ""
                    for bloc in flux:
                        if bloc.choices and bloc.choices[0].delta.content:
                            texte_c += bloc.choices[0].delta.content
                            zone.markdown(surligner(texte_c.split("===VIGILANCE===")[0]))
                    corps_c, _, vigil = texte_c.partition("===VIGILANCE===")
                    st.session_state.contrat = {"type": type_c, "droit": droit, "corps": corps_c.strip(),
                                                "vigilance": vigil.strip(), "secours": utilise != modele_c,
                                                "sources": [{"ref": p["ref"], "texte": p["texte"],
                                                             "contexte": p["contexte"]} for p in extraits]}
                    fait = True
                except Exception as e:  # noqa: BLE001
                    st.warning(message_erreur(e))
            if fait:
                st.rerun()

    res = st.session_state.get("contrat")
    if res and res["corps"]:
        st.markdown(f"<div class='jck-label'>Projet · {res['type']} · {res['droit']}</div>", unsafe_allow_html=True)
        with st.container(key="papier_contrat"):
            st.markdown(surligner(res["corps"]))
        if res.get("secours"):
            st.caption("Projet rédigé par le modèle de secours (forte affluence).")
        d1, d2, d3 = st.columns([2, 2, 3])
        nom_fichier = re.sub(r"[^a-z0-9]+", "_", normaliser(res["type"])).strip("_")
        d1.download_button("Télécharger en Word", data=markdown_vers_docx(res["corps"], res["type"]),
                           file_name=f"{nom_fichier}.docx", icon=":material/download:", type="primary",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           on_click="ignore", width="stretch")
        d2.download_button("Texte brut (.md)", data=res["corps"].encode("utf-8"), file_name=f"{nom_fichier}.md",
                           mime="text/markdown", on_click="ignore", width="stretch")
        if d3.button("Effacer ce projet", key="ctr_effacer"):
            st.session_state.pop("contrat", None)
            st.rerun()
        if res["vigilance"]:
            with st.container(border=True):
                st.markdown("<div class='jck-section'>Points de vigilance</div>", unsafe_allow_html=True)
                st.markdown(res["vigilance"])
        st.markdown("<div class='jck-label'>Articles utilisés pour la rédaction</div>", unsafe_allow_html=True)
        afficher_sources(res["sources"], "contrat")
        st.caption("Projet généré automatiquement à titre d'aide à la rédaction : il ne remplace pas la relecture "
                   "d'un professionnel du droit. Certains actes (cession d'immeuble, statuts dans certains pays) "
                   "exigent l'intervention d'un notaire ou d'un avocat.")

# --- Onglet 2 : consultation directe d'un article
with onglet_art:
    st.markdown("Affichez le texte intégral d'un article, puis naviguez vers l'article précédent ou suivant.")
    c1, c2 = st.columns([3, 1])
    texte_choisi = c1.selectbox("Texte", labels_textes, key="texte_art")
    liste = articles.get(texte_choisi, [])
    numeros = [n for n, _ in liste]
    if "pos_art" not in st.session_state or st.session_state.get("pos_texte") != texte_choisi:
        st.session_state.pos_art, st.session_state.pos_texte = 0, texte_choisi

    def aller_article():
        saisie = st.session_state.get("num_art", "").strip()
        cible = "premier" if saisie.lower() in ("1", "1er", "premier") and "premier" in numeros else saisie
        if cible in numeros:
            st.session_state.pos_art = numeros.index(cible)
            st.session_state.art_introuvable = ""
        elif saisie:
            st.session_state.art_introuvable = saisie

    c2.text_input("Article n°", placeholder="ex. 309 ou 2-1", key="num_art", on_change=aller_article)
    if st.session_state.get("art_introuvable"):
        st.warning(f"L'article {st.session_state.art_introuvable} n'a pas été trouvé dans ce texte.")
    if liste:
        pos = min(st.session_state.pos_art, len(liste) - 1)
        num, idx = liste[pos]
        p0 = passages[idx[0]]
        st.markdown(f"### Article {num}")
        st.markdown(f"<div class='jck-ctx'>{p0['titre']}<br>{p0['contexte']}</div>", unsafe_allow_html=True)
        corps = passages[idx[0]]["texte"]
        for i in idx[1:]:                       # recoller les parties sans le chevauchement
            corps += passages[i]["texte"][150:]
        st.markdown(corps.replace("\n", "  \n"))
        st.code(f"{texte_choisi}, art. {num}", language=None)
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("← Article précédent", disabled=pos == 0, width="stretch"):
            st.session_state.pos_art = pos - 1
            st.rerun()
        if b2.button("Article suivant →", disabled=pos >= len(liste) - 1, width="stretch"):
            st.session_state.pos_art = pos + 1
            st.rerun()

# --- Onglet 3 : bibliothèque
with onglet_biblio:
    st.markdown("Textes intégrés, découpés article par article.")
    for espace in ESPACES:
        lot = [t for t in textes if t["pays"] == espace]
        if not lot:
            continue
        st.markdown(f"#### {espace}")
        st.dataframe(
            [{"Texte": t["titre"], "Domaine": DOMAINES.get(t["domaine"], t["domaine"]),
              "Articles": t["articles"] or "—", "Source": t["source"]} for t in lot],
            hide_index=True, width="stretch")
        for t in lot:
            if plans.get(t["label"]):
                with st.expander(f"Plan — {t['label']}"):
                    st.markdown("\n".join(f"- **art. {a} à {b}** · {i}" for i, a, b in plans[t["label"]]))

# --- Onglet 4 : à propos
with onglet_apropos:
    a, b = st.columns([2, 1])
    with a:
        st.markdown(f"""
### Une initiative de {AUTEUR}

Cet assistant a été conçu et développé par **{AUTEUR}**, juriste, pour rendre les textes de la protection
des données personnelles et du droit des affaires plus faciles à interroger, en Côte d'Ivoire, au Maroc et
dans l'espace OHADA.

**Méthode.** Chaque texte officiel est découpé article par article, avec son livre, son titre et son
chapitre. Pour chaque question, le moteur retrouve les articles pertinents, puis un modèle de langage open
source rédige une réponse **à partir de ces seuls articles**, en citant leurs références. Les articles
consultés sont affichés sous chaque réponse et peuvent être lus en entier dans l'onglet « Consulter un
article ».

**Limites.** L'outil fournit une information juridique, pas un conseil. Certains textes ont été obtenus par
reconnaissance de caractères (OCR) ou sont des versions non consolidées : vérifiez toujours le texte officiel
en vigueur avant de vous en prévaloir. La jurisprudence n'est pas encore intégrée, en dehors de décisions
de l'ARTCI.

**Contact** : {CONTACT}
""")
    with b:
        if LOGO_H.exists():
            st.image(str(LOGO_H), width=300)

st.markdown(
    f"<div class='jck-pied'><span>JCKOUASSI · JURISTE 2.0</span><br>Conçu et développé par <b>{AUTEUR}</b> · "
    f"Information juridique, pas un conseil juridique.</div>", unsafe_allow_html=True)
