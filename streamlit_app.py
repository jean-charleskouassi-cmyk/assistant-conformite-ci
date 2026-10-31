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
TITRE_RE = re.compile(r"(?mi)^\s*(livre|titre|chapitre|section)\b[^\n]{0,120}")


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
            model=modele, temperature=0, max_tokens=900,
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
    if st.button("Nouvelle conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    nb_articles = sum(t["articles"] for t in textes)
    st.caption(f"{len(textes)} textes · {nb_articles:,} articles indexés".replace(",", " "))
    st.caption(f"Une initiative de **{AUTEUR}**")

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

onglet_q, onglet_art, onglet_biblio, onglet_apropos = st.tabs(
    ["Question juridique", "Consulter un article", "Bibliothèque des textes", "À propos"])


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
                        plan_resume(t["label"], plans) for t in textes
                        if t["domaine"] in domaines and t["pays"] in choix_espaces and t["label"] in plans)
                    analyse = analyser_question(client, question, precedente, plan_dispo)
                    trouves = rechercher(question + " " + analyse.get("question_precise", ""), passages, index,
                                         articles, domaines, choix_espaces, analyse=analyse)
                contexte = "\n\n".join(
                    f"[{n + 1}] {p['ref']} — {p['titre']}\n{p['texte']}" for n, p in enumerate(trouves)
                ) or "(Aucun extrait pertinent trouvé dans le corpus.)"
                concernes = list(dict.fromkeys(p["label"] for p in trouves if p["label"] in plans))
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
                        flux = client.chat.completions.create(model=modele, messages=messages,
                                                              stream=True, temperature=0.2)

                        def generer():
                            for bloc in flux:
                                if bloc.choices and bloc.choices[0].delta.content:
                                    yield bloc.choices[0].delta.content

                        reponse = st.write_stream(generer())
                    except Exception as e:
                        nom = type(e).__name__
                        reponse = ("Le service est très sollicité (limite de requêtes atteinte). "
                                   "Réessayez dans une minute." if "RateLimit" in nom else
                                   f"Le service de génération est momentanément indisponible ({nom}). "
                                   "Réessayez dans un instant.")
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
