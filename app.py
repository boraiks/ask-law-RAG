import os
import json

import numpy as np
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------------- Sayfa ayarları -----------------------------
st.set_page_config(
    page_title="TCK Hukuk Asistanı",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- Model ve indeksi bir kez yükle -------------------
# @st.cache_resource: uygulama yeniden çalışsa da model/index bir kez yüklenir.
@st.cache_resource(show_spinner="Model ve TCK indeksi yükleniyor (ilk açılış biraz sürer)...")
def kaynaklari_yukle():
    index = faiss.read_index("tck.faiss")
    with open("tck_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)
    return index, meta["kayitlar"], meta["maddeler"], model


index, kayitlar, maddeler, embed_model = kaynaklari_yukle()
madde_dict = {m["madde_no"]: m for m in maddeler}

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
GEN_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SISTEM_CEVAP = """Sen bir Türk ceza hukuku bilgilendirme asistanısın. Sana bir vatandaşın
sorusu ve ilgili olabilecek TCK maddeleri verilecek.

KURALLAR:
- Getirilen maddelerden SADECE soruyla DOĞRUDAN ilgili olanları kullan. İlgisiz maddelere hiç değinme.
- Cevabı SADECE verilen maddelere dayandır. Maddelerde OLMAYAN bilgi UYDURMA.
- Hangi maddeye dayandığını mutlaka belirt (örn. "TCK m.86'ya göre...").
- Soru bu maddelerle cevaplanamıyorsa "Elimdeki maddelerde bu konuda bir hüküm bulamadım" de.
- Sade, anlaşılır Türkçe kullan; hukuki terimleri açıkla.
- Cevabın sonuna şunu ekle: "Bu bilgilendirme amaçlıdır, hukuki tavsiye değildir. Durumunuz için bir avukata danışın."
"""


# ------------------------------- RAG mantığı ------------------------------
def getir(soru, k_aday=20, k_madde=5):
    """Soruya en yakın maddeleri getirir (madde + soru vektörleri karışık gelir,
    aynı maddeye birden çok isabet teke indirilir)."""
    q = np.asarray(embed_model.encode([soru], normalize_embeddings=True), dtype="float32")
    _, I = index.search(q, k_aday)
    gorulen, secilen = set(), []
    for idx in I[0]:
        no = kayitlar[idx]["madde_no"]
        if no not in gorulen:
            gorulen.add(no)
            secilen.append(no)
        if len(secilen) >= k_madde:
            break
    return [madde_dict[no] for no in secilen if no in madde_dict]


def cevapla(soru):
    secilen = getir(soru)
    baglam = "\n\n".join(
        f"[TCK m.{m['madde_no']} - {m['baslik']}]\n{m['metin']}" for m in secilen
    )
    icerik = f"SORU: {soru}\n\nİLGİLİ MADDELER:\n{baglam}"
    r = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SISTEM_CEVAP},
            {"role": "user", "content": icerik},
        ],
        temperature=0.3,
    )
    return r.choices[0].message.content, secilen


# ------------------------------ Kenar çubuğu ------------------------------
with st.sidebar:
    st.title("⚖️ TCK Asistanı")
    st.caption("Türk Ceza Kanunu soru-cevap asistanı")
    st.divider()

    st.markdown("#### Nasıl kullanılır?")
    st.markdown(
        "Başınıza gelen durumu **günlük dille** yazmanız yeterli — "
        "hukuk terimi bilmenize gerek yok."
    )

    st.markdown("#### Örnek sorular")
    for ornek in [
        "Komşum bana tokat attı, ne yapabilirim?",
        "İnternetten dolandırıldım",
        "Biri beni öldürmekle tehdit ediyor",
        "İzinsiz fotoğrafımı paylaştılar",
    ]:
        st.markdown(f"- {ornek}")

    st.divider()
    aktif = sum(1 for m in maddeler if m.get("yururlukte", True))
    st.metric("Taranan TCK maddesi", aktif)
    st.warning(
        "Bu bir **bilgilendirme** aracıdır, hukuki tavsiye değildir. "
        "Durumunuz için bir avukata danışın."
    )


# -------------------------------- Sohbet ----------------------------------
st.title("TCK Hukuk Asistanı")

if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []


def maddeleri_goster(maddeler_listesi):
    """Cevabın altındaki açılır 'Kaynak maddeler' bölümü."""
    with st.expander(f"📋 Kaynak maddeler ({len(maddeler_listesi)})"):
        for m in maddeler_listesi:
            st.markdown(f"**TCK m.{m['madde_no']} — {m['baslik']}**")
            st.caption(m["metin"])


# Önceki mesajları çiz
for msg in st.session_state.mesajlar:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["icerik"])
        if msg.get("maddeler"):
            maddeleri_goster(msg["maddeler"])

# Yeni soru
if soru := st.chat_input("Sorunuzu yazın..."):
    st.session_state.mesajlar.append({"rol": "user", "icerik": soru})
    with st.chat_message("user"):
        st.markdown(soru)

    with st.chat_message("assistant"):
        with st.spinner("İlgili maddeler aranıyor ve cevap hazırlanıyor..."):
            try:
                cevap, kaynak = cevapla(soru)
            except Exception as e:
                cevap, kaynak = f"Bir hata oluştu: {e}", []
        st.markdown(cevap)
        if kaynak:
            maddeleri_goster(kaynak)

    st.session_state.mesajlar.append(
        {"rol": "assistant", "icerik": cevap, "maddeler": kaynak}
    )