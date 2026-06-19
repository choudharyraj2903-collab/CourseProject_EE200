"""
app.py
Q3B — Signals to Softwares: 'Zapptain America'
Streamlit app with two modes: single-clip identification
and batch identification producing results.csv.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile

from fingerprint import (
    load_database, match_query, TARGET_SR
)

st.set_page_config(page_title="Audio Fingerprint Identifier", layout="wide")

# ── Load database once, cache across reruns ────────────────
@st.cache_resource
def get_database():
    return load_database("database.pkl")

db = get_database()

st.title("🎵 Audio Fingerprint Identifier")
st.caption(f"Database loaded: {len(db)} unique hashes indexed")

mode = st.sidebar.radio("Mode", ["Single-Clip Identification", "Batch Identification"])


# ════════════════════════════════════════════════════════════
# SINGLE-CLIP MODE
# ════════════════════════════════════════════════════════════

if mode == "Single-Clip Identification":
    st.header("Single-Clip Mode")
    uploaded = st.file_uploader(
        "Upload a query audio clip (mp3/wav)", type=["mp3", "wav"]
    )

    if uploaded is not None:
        # Save to temp file — librosa needs a path or seekable buffer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Fingerprinting query..."):
            best_song, votes, histogram, (f, t, Sxx), peaks = match_query(tmp_path, db)

        os.unlink(tmp_path)

        # ── Result ──────────────────────────────────────────
        if best_song:
            st.success(f"### 🎯 Matched Song: **{best_song}**")
        else:
            st.error("### ❌ No match found in database")

        col1, col2 = st.columns(2)

        # ── Spectrogram ─────────────────────────────────────
        with col1:
            st.subheader("Spectrogram")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            Sdb = 10 * np.log10(Sxx + 1e-10)
            ax1.pcolormesh(t, f, Sdb, cmap='hot', shading='gouraud',
                           vmin=np.percentile(Sdb, 30), vmax=Sdb.max())
            ax1.set_ylim(0, 5000)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Frequency (Hz)")
            st.pyplot(fig1)

        # ── Constellation ────────────────────────────────────
        with col2:
            st.subheader("Constellation of Peaks")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            if len(peaks) > 0:
                ax2.scatter(peaks[:, 0], peaks[:, 1], c='steelblue', s=8)
            ax2.set_ylim(0, 5000)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Frequency (Hz)")
            st.pyplot(fig2)

        # ── Offset histogram ─────────────────────────────────
        st.subheader("Offset Histogram (decides the match)")
        if best_song and best_song in histogram:
            top_candidates = sorted(
                votes.keys(), key=lambda s: -max(votes[s].values())
            )[:4]

            cols = st.columns(len(top_candidates))
            for c, song in zip(cols, top_candidates):
                offsets = histogram[song]
                fig3, ax3 = plt.subplots(figsize=(4, 3))
                bins = np.arange(min(offsets)-0.5, max(offsets)+1.5, 0.5)
                ax3.hist(offsets, bins=bins,
                         color='steelblue' if song == best_song else 'gray')
                ax3.set_title(f"{'★ ' if song == best_song else ''}{song}", fontsize=9)
                ax3.set_xlabel("Offset (s)")
                c.pyplot(fig3)
        else:
            st.info("No matching hashes found to build a histogram.")


# ════════════════════════════════════════════════════════════
# BATCH MODE
# ════════════════════════════════════════════════════════════

else:
    st.header("Batch Mode")
    st.write("Upload multiple query clips. Output: `results.csv` with columns `filename, prediction`.")

    uploaded_files = st.file_uploader(
        "Upload multiple query clips", type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Run Batch Identification"):
        results = []
        progress = st.progress(0)

        for i, uploaded in enumerate(uploaded_files):
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            best_song, _, _, _, _ = match_query(tmp_path, db)
            os.unlink(tmp_path)

            results.append({
                "filename": uploaded.name,
                "prediction": best_song if best_song else "NO_MATCH"
            })
            progress.progress((i + 1) / len(uploaded_files))

        results_df = pd.DataFrame(results)
        st.success(f"Identified {len(results)} clips")
        st.dataframe(results_df)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results.csv",
            data=csv_bytes,
            file_name="results.csv",
            mime="text/csv"
        )