# -*- coding: utf-8 -*-
"""
Assistant de conformité (loi ivoirienne n° 2013-450) — version en ligne (Streamlit).
Le corpus juridique est intégré directement ci-dessous (dictionnaire CORPUS).
Réponses via l'API open source gratuite Groq ; la clé est lue dans les Secrets.
"""

import re
import streamlit as st
from rank_bm25 import BM25Okapi
from groq import Groq

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5

SYSTEME = (
    "Tu es un assistant d'information sur la conformité à la loi ivoirienne "
    "n° 2013-450 relative à la protection des données à caractère personnel. "
    "Règles impératives :\n"
    "1. Réponds UNIQUEMENT à partir des extraits de corpus fournis.\n"
    "2. Cite toujours l'article ou la décision sur lequel tu t'appuies "
    "(par exemple : art. 14, décision n° 2023-0937).\n"
    "3. Si la réponse ne figure pas dans les extraits, dis-le clairement "
    "et n'invente rien.\n"
    "4. Rappelle, quand c'est pertinent, que tu fournis une information et non "
    "un conseil juridique, et qu'un juriste doit valider les décisions importantes.\n"
    "Réponds en français, de façon claire et concise."
)

# --- Corpus juridique intégré (nom de source -> texte) ---
CORPUS = {
    'loi_2013-450.txt': "LOI N° 2013-450 DU 19 JUIN 2013 RELATIVE A LA PROTECTION DES DONNEES A CARACTERE PERSONNEL (CÔTE D'IVOIRE)\n\nREMARQUE : extrait de travail, transcrit à partir d'un scan OCR. Les numéros d'articles\net les montants doivent être vérifiés sur le texte officiel (Journal officiel) avant toute\ncitation définitive. Remplace idéalement ce fichier par le PDF officiel complet.\n\n--- DÉFINITIONS (article 1er) ---\n\nConsentement de la personne concernée : toute manifestation de volonté expresse, non\néquivoque, libre, spécifique et informée par laquelle la personne concernée ou son\nreprésentant accepte que ses données à caractère personnel fassent l'objet d'un traitement.\n\nDonnées sensibles : données à caractère personnel relatives aux opinions ou activités\nreligieuse, philosophique, politique, syndicale, à la vie sexuelle ou raciale, à la santé,\naux mesures d'ordre social, aux poursuites, aux sanctions pénales ou administratives.\n\nMineur : toute personne âgée de moins de dix-huit ans.\n\nResponsable du traitement : la personne physique ou morale, publique ou privée, qui, seule\nou conjointement, décide de collecter et de traiter des données à caractère personnel et en\ndétermine les finalités.\n\nAutorité de protection : l'autorité administrative indépendante chargée de veiller à ce que\nles traitements des données à caractère personnel soient mis en œuvre conformément à la loi.\n\nCorrespondant à la protection des données / code de conduite / agrément : notions définies\npar la loi (voir articles 12, et définitions du code de conduite homologué par l'Autorité et\nde l'agrément de produits ou systèmes).\n\n--- CHAMP ET FORMALITÉS ---\n\nArticle 3 : Sont soumis à la loi toute collecte, traitement, transmission, stockage et\nutilisation de données à caractère personnel par une personne physique, l'État, les\ncollectivités ou les personnes morales, et tout traitement mis en œuvre sur le territoire\nnational.\n\nArticle 5 : Le traitement des données à caractère personnel est soumis à une déclaration\npréalable auprès de l'Autorité de protection. L'Autorité délivre un récépissé ; le traitement\npeut être mis en œuvre dès réception, sans exonération de responsabilité.\n\nArticle 6 : Sont dispensés de déclaration préalable, notamment : les traitements à usage\nexclusivement personnel ou domestique ; ceux dont la publication est prescrite par la loi ;\nles registres à usage privé ; et les traitements pour lesquels le responsable a désigné un\ncorrespondant à la protection des données (sauf transfert vers un pays tiers).\n\nArticle 7 : Sont soumis à AUTORISATION PRÉALABLE de l'Autorité avant toute mise en œuvre :\nles traitements de données génétiques, médicales et de recherche ; les données relatives aux\ninfractions, condamnations ou mesures de sûreté ; le traitement portant sur un numéro\nnational d'identification ou les numéros de téléphone ; les données biométriques ; les\ntraitements d'intérêt public à fins historiques, statistiques ou scientifiques ; le transfert\nde données vers un pays tiers.\n\nArticle 12 : Le correspondant à la protection des données est une personne qualifiée. Il tient\nla liste des traitements, est accessible à toute personne qui en fait la demande, ne peut être\nsanctionné par l'employeur pour l'exercice de ses missions et peut saisir l'Autorité.\n\n--- PRINCIPES DIRECTEURS ---\n\nArticle 14 : Le traitement est légitime si la personne concernée donne expressément son\nconsentement préalable. Il peut être dérogé à cette exigence lorsque le responsable est dûment\nautorisé et que le traitement est nécessaire : au respect d'une obligation légale ; à une\nmission d'intérêt public ; à l'exécution d'un contrat ; à la sauvegarde des droits et libertés\nfondamentaux de la personne concernée.\n\nArticle 15 : La collecte, l'enregistrement, le traitement, le stockage, la transmission et\nl'interconnexion doivent se faire de manière licite et loyale.\n\nArticle 16 : Les données sont collectées pour des finalités déterminées, explicites et\nlégitimes, et ne peuvent être traitées ultérieurement de manière incompatible. Elles doivent\nêtre adéquates, pertinentes et non excessives, et conservées pendant une durée n'excédant pas\nla période nécessaire aux finalités pour lesquelles elles ont été collectées ou traitées.\n\nArticle 17 : Les données collectées doivent être exactes et, si nécessaire, mises à jour ;\nles données inexactes ou incomplètes doivent être effacées ou rectifiées.\n\nArticle 18 : Le principe de transparence implique une information obligatoire et claire du\nresponsable du traitement sur les données à caractère personnel.\n\nArticle 19 : Les données doivent être traitées de manière confidentielle et protégées,\nnotamment lorsque le traitement comporte des transmissions dans un réseau.\n\nArticle 20 : Le responsable du traitement doit choisir un sous-traitant offrant des garanties\nsuffisantes pour la protection et la confidentialité des données.\n\n--- DROITS DE LA PERSONNE CONCERNÉE ---\n\nArticle 28 : Le responsable du traitement informe la personne concernée, au plus tard lors de\nla collecte : de son identité, des finalités, des catégories de données et de destinataires,\nde la possibilité de refuser de figurer sur le fichier, du droit d'accès et de rectification,\nde la durée de conservation et de tout transfert vers un pays tiers.\n\nArticle 29 : Toute personne peut obtenir du responsable la confirmation et la communication des\ndonnées la concernant, ainsi que les informations permettant de connaître et de contester le\ntraitement. À défaut d'accès, l'Autorité dispose d'un pouvoir d'investigation et peut ordonner\nla rectification, l'effacement ou le verrouillage des données non conformes.\n\nArticle 30 : Toute personne a le droit de s'opposer, pour des motifs légitimes, au traitement\nde ses données, et de s'opposer gratuitement au traitement à des fins de prospection.\n\nArticle 31 : Toute personne peut exiger que soient rectifiées, complétées, mises à jour,\nverrouillées ou supprimées les données inexactes, incomplètes, périmées, ou dont la collecte,\nl'utilisation ou la conservation est interdite.\n\nArticle 33 : La personne concernée a droit à l'effacement (droit à l'oubli numérique),\nnotamment lorsque les données ne sont plus nécessaires, que le consentement est retiré, que le\ndélai de conservation a expiré ou que le traitement n'est pas conforme à la loi.\n\n--- TRANSFERT VERS UN PAYS TIERS ---\n\nArticle 26 : Le responsable ne peut transférer des données vers un pays tiers que si cet État\nassure un niveau de protection supérieur ou équivalent. Le transfert nécessite l'autorisation\npréalable de l'Autorité de protection et fait l'objet d'un contrôle régulier au regard de sa\nfinalité.\n\n--- SÉCURITÉ ---\n\nArticle 40 : Le responsable du traitement prend toute précaution, au regard de la nature des\ndonnées, pour empêcher qu'elles soient déformées, endommagées, ou que des tiers non autorisés\ny aient accès. Il choisit un sous-traitant offrant des garanties suffisantes de sécurité.\n\nArticle 41 : Le responsable est tenu, notamment : d'empêcher l'accès non autorisé aux\ninstallations et aux supports ; d'empêcher la lecture, la copie, la modification ou\nl'effacement non autorisés ; de garantir que seules les personnes autorisées accèdent aux\ndonnées relevant de leur autorisation ; de pouvoir vérifier a posteriori qui a accédé aux\ndonnées et ce qui a été introduit, modifié ou effacé ; de sauvegarder les données par des\ncopies de sécurité protégées.\n\nArticle 42 : Le responsable établit un rapport annuel pour l'Autorité de protection sur le\nrespect des mesures de sécurité de l'article 41.\n\nArticle 45 : Est puni d'un emprisonnement d'un mois à deux ans et d'une amende de 1 000 000 à\n10 000 000 FCFA quiconque entrave l'action de l'Autorité de protection (opposition aux\nmissions, refus de communiquer, dissimulation de documents).\n\n--- L'AUTORITÉ DE PROTECTION ET LES SANCTIONS ---\n\nArticle 46 : Les missions de l'autorité de protection des données à caractère personnel sont\nconfiées à l'autorité administrative indépendante en charge de la régulation des\ntélécommunications et des technologies de l'information et de la communication (ARTCI).\n\nArticle 47 : L'Autorité de protection informe, reçoit déclarations et plaintes, octroie et\nretire les autorisations, procède à des vérifications par des agents assermentés, prononce des\nsanctions administratives et pécuniaires, tient un répertoire public des traitements, conseille,\ndonne son avis sur les textes, élabore des règles de conduite et autorise les transferts\ntransfrontaliers.\n\nArticle 49 : L'Autorité de protection peut prononcer à l'égard du responsable : un avertissement\npour non-respect des obligations ; une mise en demeure de faire cesser les manquements dans le\ndélai qu'elle fixe.\n\nArticle 50 : En cas de violation des droits et libertés, l'Autorité peut, après procédure\ncontradictoire, décider de l'interruption du traitement, du verrouillage de certaines données,\nou de l'interdiction temporaire ou définitive d'un traitement contraire à la loi.\n\nArticle 51 : Après avoir entendu le responsable qui ne se conforme pas à la mise en demeure,\nl'Autorité peut prononcer : le retrait provisoire ou définitif de l'autorisation ; une sanction\npécuniaire proportionnée. Le montant ne peut excéder 10 000 000 FCFA ; en cas de manquement\nréitéré dans les cinq ans, il ne peut excéder 100 000 000 FCFA ou, pour une entreprise, 5 % du\nchiffre d'affaires hors taxes du dernier exercice, dans la limite de 500 000 000 FCFA. Ces\nsanctions s'appliquent sans préjudice des sanctions pénales.\n\nArticle 53 : Les responsables de traitement disposent d'un délai de six mois à compter de\nl'entrée en vigueur de la loi pour se mettre en conformité.\n",
    'decisions_artci.md': "# Décisions de l'ARTCI (Autorité de régulation des télécommunications de Côte d'Ivoire)\n\nCes trois décisions ont été rendues le 23 août 2023. Dans les trois cas, l'ARTCI\na prononcé un avertissement assorti d'une mise en demeure de se mettre en\nconformité (article 49 de la loi n° 2013-450), sans amende.\n\n## Décision n° 2023-0937 — Yango CIV LLC\nManquements : enregistrement des conversations des passagers de VTC et exposition\nde leurs données (noms, numéros de téléphone, courriels) sans consentement\n(art. 14) ; conservation au-delà de la durée nécessaire (art. 16) ; mesures de\nsécurité insuffisantes (art. 19 et 41).\nSanction : avertissement et mise en demeure (art. 49), désactivation de l'option\nd'enregistrement. Pas d'amende.\n\n## Décision n° 2023-0936 — Standard Chartered Bank Côte d'Ivoire S.A.\nManquements : collecte excessive de données sans consentement clair ; sécurité\ninsuffisante ; absence de notification des violations aux autorités et aux clients.\nSanction : avertissement et mise en demeure (art. 49). Pas d'amende.\n\n## Décision n° 2023-0938 — Société Quipux Afrique S.A.\nManquements : collecte sans consentement éclairé ; conservation au-delà de la\nfinalité ; failles de sécurité.\nSanction : avertissement et mise en demeure (art. 49). Pas d'amende.\n\n## À retenir sur le pouvoir de sanction de l'ARTCI\nL'ARTCI peut aller plus loin que l'avertissement : l'article 51 l'autorise à\nprononcer des sanctions pécuniaires (jusqu'à 10 000 000 FCFA ; jusqu'à\n100 000 000 FCFA ou 5 % du chiffre d'affaires, dans la limite de 500 000 000 FCFA,\nen cas de récidive), et à retirer l'autorisation (provisoirement ou définitivement),\nsans préjudice des sanctions pénales.\n",
}


def _tokeniser(texte):
    return re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+", texte.lower())


def _decouper(texte, taille, chevauchement):
    texte = texte.strip()
    morceaux, debut = [], 0
    while debut < len(texte):
        morceaux.append(texte[debut:debut + taille].strip())
        debut += taille - chevauchement
    return [m for m in morceaux if m]


@st.cache_resource(show_spinner="Préparation du corpus juridique…")
def charger_index():
    docs, sources = [], []
    for nom, texte in CORPUS.items():
        for morceau in _decouper(texte, CHUNK_SIZE, CHUNK_OVERLAP):
            docs.append(morceau)
            sources.append(nom)
    bm25 = BM25Okapi([_tokeniser(d) for d in docs]) if docs else None
    return docs, sources, bm25


def retrouver(question, docs, sources, bm25, k=TOP_K):
    if bm25 is None:
        return []
    scores = bm25.get_scores(_tokeniser(question))
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], sources[i]) for i in indices]


st.set_page_config(page_title="Assistant conformité — Côte d'Ivoire", page_icon="⚖️")
st.title("Assistant conformité — données personnelles (Côte d'Ivoire)")
st.caption(
    "Répond à partir de la loi n° 2013-450 et des décisions de l'ARTCI, en citant ses "
    "sources. ⚠️ Outil d'information, pas un conseil juridique : faites valider les "
    "décisions importantes par un juriste."
)

if "GROQ_API_KEY" not in st.secrets:
    st.error(
        "La clé GROQ_API_KEY n'est pas configurée. Ajoute-la dans les Secrets de "
        "l'application (menu ⋮ > Settings > Secrets)."
    )
    st.stop()

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
LLM_MODEL = st.secrets["GROQ_MODEL"] if "GROQ_MODEL" in st.secrets else "llama-3.3-70b-versatile"
docs, sources, bm25 = charger_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Pose ta question sur la loi 2013-450…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    passages = retrouver(question, docs, sources, bm25)
    contexte = "\n\n".join(f"[Source : {s}]\n{d}" for d, s in passages)
    invite = (
        f"Extraits du corpus juridique :\n{contexte}\n\n"
        f"Question de l'utilisateur : {question}"
    )

    with st.chat_message("assistant"):
        flux = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEME},
                {"role": "user", "content": invite},
            ],
            stream=True,
            temperature=0.2,
        )

        def generer():
            for bloc in flux:
                yield bloc.choices[0].delta.content or ""

        reponse = st.write_stream(generer())
        if passages:
            st.caption("Sources consultées : " + ", ".join(sorted({s for _, s in passages})))

    st.session_state.messages.append({"role": "assistant", "content": reponse})
