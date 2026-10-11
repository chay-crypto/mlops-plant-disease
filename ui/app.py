import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")


def lisible(label):
    plante, _, maladie = label.partition("___")
    plante = plante.replace("_", " ").strip()
    maladie = maladie.replace("_", " ").strip()
    return plante, maladie


st.set_page_config(page_title="Diagnostic des plantes")
st.title("Diagnostic des maladies des plantes")
st.write("Déposez une photo de feuille : le modèle reconnaît 38 couples plante / maladie.")

st.sidebar.header("Connexion")
api_key = st.sidebar.text_input("Clé d'API", type="password")

if st.sidebar.button("Voir mon usage"):
    r = requests.get(f"{API_URL}/usage", headers={"X-API-Key": api_key}, timeout=10)
    if r.ok:
        st.sidebar.success(f"{r.json()['user']} : {r.json()['requests']} requête(s)")
    else:
        st.sidebar.error("Clé d'API invalide.")

fichier = st.file_uploader("Photo de feuille", type=["jpg", "jpeg", "png"])

if fichier:
    st.image(fichier, width=300)

    if st.button("Analyser"):
        if not api_key:
            st.error("Entrez votre clé d'API dans le panneau de gauche.")
        else:
            with st.spinner("Analyse en cours..."):
                r = requests.post(
                    f"{API_URL}/predict",
                    headers={"X-API-Key": api_key},
                    files={"file": (fichier.name, fichier.getvalue(), fichier.type)},
                    timeout=30,
                )

            if r.status_code == 401:
                st.error("Clé d'API invalide.")
            elif not r.ok:
                st.error(f"Erreur de l'API ({r.status_code}).")
            else:
                preds = r.json()["predictions"]
                plante, maladie = lisible(preds[0]["label"])
                confiance = preds[0]["probability"] * 100

                if maladie.lower() == "healthy":
                    st.success(f"{plante} : plante saine ({confiance:.1f} %)")
                else:
                    st.warning(f"{plante} : {maladie} ({confiance:.1f} %)")

                st.subheader("Les 3 hypothèses du modèle")
                for p in preds:
                    plante, maladie = lisible(p["label"])
                    st.write(f"{plante} — {maladie}")
                    st.progress(p["probability"])