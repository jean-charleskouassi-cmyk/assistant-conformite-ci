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

import re
import unicodedata
from pathlib import Path

import streamlit as st
from groq import Groq
from rank_bm25 import BM25Okapi

ICI = Path(__file__).parent
CORPUS_DIR = ICI / "corpus"
LOGO_H = ICI / "assets" / "logo_horizontal.png"
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
2. Cite chaque règle avec sa référence exacte entre crochets, telle qu'elle figure dans l'en-tête de \
l'extrait, par exemple [Loi 09-08 (Maroc), art. 4].
3. Si les extraits ne suffisent pas, dis-le et indique quel texte consulter. N'invente jamais un numéro \
d'article, un délai ou un montant.
4. Si plusieurs pays ou espaces sont concernés, distingue nettement ce que prévoit chacun.
5. Si l'en-tête d'un extrait indique « VERSION D'ORIGINE » ou « OCR », signale qu'il faut vérifier le \
texte officiel en vigueur avant toute citation définitive.
6. Écris en français juridique clair, sans jargon inutile.
7. Termine par une phrase rappelant qu'il s'agit d'une information juridique et non d'un conseil, \
à faire valider par un professionnel."""

FORMATS = {
    "Synthèse": "Réponds en quelques paragraphes courts : la réponse d'abord, puis ses fondements textuels.",
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
    return passages, index, textes, articles


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


def rechercher(question, passages, index, articles, domaines, espaces, k=TOP_K):
    """Articles cités explicitement, puis meilleurs passages ; chaque espace pertinent a sa part."""
    exacts = articles_cites(question, passages, articles)
    requete = question
    for motif, ajout in EXPANSIONS.items():
        if re.search(motif, question, re.I):
            requete += ajout
    cites = [e for e, motif in ESPACES_CITES.items() if re.search(motif, question, re.I) and e in espaces]
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
        pertinents = [e for e, lst in par_espace.items()
                      if scores[lst[0]] >= 0.6 * meilleur or e in cites]
        part = max(2, k // max(1, len(pertinents)))
        for e in pertinents:
            choisis.extend(par_espace[e][:part])
        for i in ordre:
            if len(choisis) >= k:
                break
            if i not in choisis:
                choisis.append(i)
        choisis = sorted(choisis[:k], key=lambda i: scores[i], reverse=True)
    resultat = list(exacts)
    for i in choisis:
        if passages[i] not in resultat and len(resultat) < k + len(exacts):
            resultat.append(passages[i])
    return resultat


# ------------------------------------------------------------------ interface

st.set_page_config(page_title="JCKouassi · Assistant juridique",
                   page_icon=str(LOGO_ICONE) if LOGO_ICONE.exists() else "⚖️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, p, li, label { font-family: 'Inter', system-ui, sans-serif; }
h1, h2, h3, .jck-titre { font-family: 'Source Serif 4', Georgia, serif !important; color: #14284b; }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
.jck-entete { display: flex; align-items: center; gap: 18px; padding: 6px 0 14px;
              border-bottom: 2px solid #c9972f; margin-bottom: 8px; }
.jck-titre { font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin: 0; }
.jck-sous { color: #4a5670; font-size: .95rem; margin-top: 2px; }
.jck-sous b { color: #9c7322; }
.jck-pied { margin-top: 2.5rem; padding-top: 12px; border-top: 1px solid #d7dce6; color: #5c6780;
            font-size: .85rem; text-align: center; }
.jck-pied b { color: #14284b; }
.jck-ref { font-weight: 600; color: #14284b; }
.jck-ctx { color: #6b7590; font-size: .82rem; margin-bottom: 4px; }
div[data-testid="stExpander"] details { border-color: #d7dce6; background: #ffffff; }
</style>
""", unsafe_allow_html=True)

passages, index, textes, articles = charger_corpus()
espaces_dispo = [e for e in ESPACES if any(t["pays"] == e for t in textes)]
labels_textes = [t["label"] for t in textes if t["label"] in articles]

if LOGO_H.exists():
    st.logo(str(LOGO_H), icon_image=str(LOGO_ICONE) if LOGO_ICONE.exists() else None, size="large")

with st.sidebar:
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

# En-tête
col_logo, col_titre = st.columns([1, 9], vertical_alignment="center")
with col_logo:
    if LOGO_ICONE.exists():
        st.image(str(LOGO_ICONE), width=84)
with col_titre:
    st.markdown(
        f"<div class='jck-titre'>Assistant juridique</div>"
        f"<div class='jck-sous'>Données personnelles et droit des affaires · Côte d'Ivoire, Maroc, OHADA, "
        f"CEDEAO, Union africaine<br>Une initiative de <b>{AUTEUR}</b> · Juriste 2.0</div>",
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
with onglet_q:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "GROQ_API_KEY" not in st.secrets:
        st.error("La clé GROQ_API_KEY n'est pas configurée dans les Secrets de l'application.")
    else:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        modele = st.secrets["GROQ_MODEL"] if "GROQ_MODEL" in st.secrets else DEFAULT_MODEL

        if not st.session_state.messages:
            st.markdown("**Exemples de questions**")
            exemples = [
                "Faut-il déclarer un fichier clients à l'autorité de protection, au Maroc et en Côte d'Ivoire ?",
                "Quelles sont les conditions de constitution d'une SAS en droit OHADA ?",
                "Comment fonctionne la procédure d'injonction de payer dans l'espace OHADA ?",
                "Que prévoit la Convention de Malabo sur les transferts de données hors de l'Union africaine ?",
            ]
            c1, c2 = st.columns(2)
            for n, ex in enumerate(exemples):
                if (c1 if n % 2 == 0 else c2).button(ex, key=f"ex{n}", width="stretch"):
                    st.session_state.question_exemple = ex
                    st.rerun()

        for n, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"], avatar=str(LOGO_ICONE) if m["role"] == "assistant" and LOGO_ICONE.exists() else None):
                st.markdown(m["content"])
                if m.get("sources"):
                    st.markdown("**Articles consultés**")
                    afficher_sources(m["sources"], f"h{n}")

        with st.form("question", clear_on_submit=True):
            saisie = st.text_area("Votre question", height=110, label_visibility="collapsed",
                                  placeholder="Posez votre question juridique, ou citez un article : « Que prévoit l'article 309 de l'AUSCGIE ? »")
            envoye = st.form_submit_button("Rechercher et répondre", type="primary")
        question = saisie.strip() if envoye and saisie.strip() else st.session_state.pop("question_exemple", None)

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            if not choix_espaces:
                reponse, trouves = "Sélectionnez au moins un pays ou espace juridique dans le panneau de gauche.", []
                st.warning(reponse)
            else:
                trouves = rechercher(question, passages, index, articles, domaines, choix_espaces)
                contexte = "\n\n".join(
                    f"[{n + 1}] {p['ref']} — {p['titre']}\n{p['texte']}" for n, p in enumerate(trouves)
                ) or "(Aucun extrait pertinent trouvé dans le corpus.)"
                historique = [{"role": m["role"], "content": m["content"][:1500]}
                              for m in st.session_state.messages[-5:-1]]
                messages = [{"role": "system", "content": SYSTEME + "\n\n" + FORMATS[format_rep]},
                            *historique,
                            {"role": "user", "content": f"Extraits des textes :\n\n{contexte}\n\nQuestion : {question}"}]
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
                    if trouves:
                        st.markdown("**Articles consultés**")
                        afficher_sources(trouves, "dernier")
            st.session_state.messages.append({"role": "assistant", "content": reponse, "sources": [
                {"ref": s["ref"], "texte": s["texte"], "contexte": s["contexte"]} for s in trouves]})

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
    f"<div class='jck-pied'>Conçu et développé par <b>{AUTEUR}</b> · JCKouassi, Juriste 2.0 · "
    f"Information juridique, pas un conseil juridique.</div>", unsafe_allow_html=True)
