import streamlit as st
import librosa
import numpy as np
import scipy.signal
import scipy.fft
import soundfile as sf  # For Metadata Extraction
import tempfile
import os
import gc

# ==============================================================================
# MODULE 1: DSP KERNEL
# ==============================================================================

class MasteringDSP:
    @staticmethod
    def load_audio(file_path):
        try:
            # Librosa loads FLAC natively if soundfile is installed
            y, sr = librosa.load(file_path, sr=44100, mono=False, duration=180)
            if y.ndim == 1: y = np.stack((y, y))
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Critical Load Error: {e}")

    @staticmethod
    def get_file_metadata(file_path):
        """
        Extracts Sample Rate and Bit Depth/Format info directly from the file header.
        """
        try:
            with sf.SoundFile(file_path) as f:
                sr = f.samplerate
                subtype = f.subtype
                fmt = f.format
                
                # Bit Depth Analysis
                if 'PCM' in subtype:
                    quality = f"{subtype.split('_')[-1]}-bit INT"
                elif 'FLOAT' in subtype:
                    quality = "32-bit FLOAT"
                elif 'MPEG' in subtype:
                    quality = "Lossy (MP3)" # MP3 bitrate is hard to get without specific libs, keeps it safe
                else:
                    quality = subtype
                
                return sr, quality, fmt
        except Exception:
            return 0, "Unknown", "Unknown"

    @staticmethod
    def k_weighting_filter(y, sr):
        b1, a1 = scipy.signal.iirfilter(1, 1500/(sr/2), btype='high', ftype='butter') 
        y_h = scipy.signal.lfilter(b1, a1, y)
        b2, a2 = scipy.signal.butter(1, 38/(sr/2), btype='high')
        y_k = scipy.signal.lfilter(b2, a2, y_h)
        return y_k

    @staticmethod
    def analyze_loudness_dynamics(y_stereo, sr):
        y_resampled = scipy.signal.resample(y_stereo, int(y_stereo.shape[1] * 2), axis=1)
        true_peak = np.max(np.abs(y_resampled))
        true_peak_db = 20 * np.log10(true_peak + 1e-9)
        
        y_k_L = MasteringDSP.k_weighting_filter(y_stereo[0], sr)
        y_k_R = MasteringDSP.k_weighting_filter(y_stereo[1], sr)
        mean_power = (np.mean(y_k_L**2) + np.mean(y_k_R**2)) / 2.0
        lufs_approx = -0.691 + 10 * np.log10(mean_power + 1e-9)
        
        plr = true_peak_db - lufs_approx
        return round(lufs_approx, 1), round(true_peak_db, 1), round(plr, 1)

    @staticmethod
    def analyze_stereo_image(y_stereo):
        if y_stereo.shape[0] < 2: return 1.0, 0.0
        L, R = y_stereo[0], y_stereo[1]
        
        dot_prod = np.mean((L - np.mean(L)) * (R - np.mean(R)))
        std_prod = np.std(L) * np.std(R) + 1e-9
        correlation = dot_prod / std_prod
        
        side = (L - R) * 0.5
        mid = (L + R) * 0.5
        side_energy = np.sum(side**2)
        mid_energy = np.sum(mid**2) + 1e-9
        width_percent = (side_energy / (mid_energy + side_energy)) * 100
        
        return round(correlation, 2), round(width_percent, 1)

    @staticmethod
    def analyze_frequency_spectrum(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        spec = np.abs(librosa.stft(y_mono, n_fft=4096))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
        
        bands = {
            "SUB": (20, 60), "BASS": (60, 250),
            "LOW_MID": (250, 2000), "HIGH_MID": (2000, 6000), "AIR": (6000, 20000)
        }
        
        energy_dist = {}
        total_energy = np.sum(spec) + 1e-9
        
        for name, (f_min, f_max) in bands.items():
            mask = (freqs >= f_min) & (freqs < f_max)
            band_energy = np.sum(spec[mask, :])
            energy_dist[name] = (band_energy / total_energy) * 100
            
        return energy_dist

    @staticmethod
    def analyze_precision_rhythm(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        _, y_perc = librosa.effects.hpss(y_mono, margin=3.0)
        onset_env = librosa.util.normalize(librosa.onset.onset_strength(y=y_perc, sr=sr, aggregate=np.median))
        
        bpms = np.arange(60, 185, 0.5)
        fs_frame = sr / 512
        n = len(onset_env)
        f = scipy.fft.rfft(onset_env, n=2*n)
        autocorr = scipy.fft.irfft(f * np.conj(f))[:n]
        
        scores = []
        for bpm in bpms:
            tau = (60.0 / bpm) * fs_frame
            period = int(round(tau))
            score = 0
            count = 0
            for m in [1, 2, 4]:
                idx = period * m
                if idx < len(autocorr):
                    score += autocorr[idx]
                    count += 1
            scores.append(score / (count if count else 1))
            
        scores = np.array(scores)
        weighting = np.exp(-0.5 * ((bpms - 120) / 50) ** 2)
        best_bpm = bpms[np.argmax(scores * weighting)]
        
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        stability = 1.0 - np.mean(np.std(tempogram, axis=0))
        groove = int(stability * 100)
        
        return round(best_bpm), groove

    @staticmethod
    def analyze_musical_key(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        y_harm, _ = librosa.effects.hpss(y_mono, margin=3.0)
        chroma = librosa.feature.chroma_cens(y=y_harm, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_mean /= np.linalg.norm(chroma_mean)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        maj_p = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        min_p = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        maj_p /= np.linalg.norm(maj_p)
        min_p /= np.linalg.norm(min_p)
        
        best_score, best_key = -1, ""
        for i in range(12):
            if np.dot(chroma_mean, np.roll(maj_p, i)) > best_score:
                best_score = np.dot(chroma_mean, np.roll(maj_p, i))
                best_key = f"{keys[i]} Maj"
            if np.dot(chroma_mean, np.roll(min_p, i)) > best_score:
                best_score = np.dot(chroma_mean, np.roll(min_p, i))
                best_key = f"{keys[i]} Min"
                
        return best_key

# ==============================================================================
# MODULE 2: AI ADVISER & MATRIX ENGINE
# ==============================================================================

class AIAdviser:
    @staticmethod
    def generate_report(data):
        advice_list = []
        
        # 1. Loudness Check (Simple)
        if data['lufs'] > -7:
            advice_list.append(("WARN", "High Loudness Level", "음압이 매우 높습니다(-7 LUFS 초과). 클럽/CD 마스터링 의도가 아니라면 다이내믹 레인지 손실을 체크하세요."))
        else:
            advice_list.append(("PASS", "Safe Loudness Range", "다이내믹 레인지가 보존된 안전한 음압 범위 내에 있습니다."))

        # 3. Stereo Phase Check
        if data['corr'] < 0.2:
            advice_list.append(("CRIT", "Phase Cancellation Risk", "위상 상관도가 낮아 모노 호환성에 문제가 발생할 수 있습니다. 저음역대(Kick/Bass)의 모노 센터링을 확인하세요."))
        elif data['corr'] < 0.6:
            advice_list.append(("INFO", "Wide Stereo Image", "넓은 스테레오 이미지를 가지고 있습니다. 의도된 공간감인지, 위상 간섭인지 확인이 필요합니다."))
        else:
            advice_list.append(("PASS", "Solid Phase Coherence", "위상 호환성이 좋으며 단단한 믹스 밸런스를 유지하고 있습니다."))

        # 4. Frequency Balance Check
        sub_energy = data['freq']['SUB']
        air_energy = data['freq']['AIR']
        
        if sub_energy > 25:
            advice_list.append(("WARN", "Excessive Low-End", f"SUB 대역 에너지가 {int(sub_energy)}%로 높습니다. 불필요한 초저역 부밍(Booming)이 없는지 확인하세요."))
        elif sub_energy < 5:
            advice_list.append(("WARN", "Lack of Low-End", "저음역 에너지가 부족합니다. 믹스가 전체적으로 가볍게 들릴 수 있습니다."))
            
        if air_energy > 15:
            advice_list.append(("WARN", "Bright High-End", f"AIR 대역 에너지가 {int(air_energy)}%로 높습니다. 보컬의 치찰음이나 심벌의 자극적인 대역을 디에서로 제어하세요."))

        return advice_list

    @staticmethod
    def generate_matrix(data):
        # 1. DYNAMICS (PLR)
        if data['plr'] < 8: d_val = "COMPRESSED"
        elif data['plr'] > 14: d_val = "DYNAMIC"
        else: d_val = "BALANCED"

        # 2. TONE (Spectral)
        sub = data['freq']['SUB']
        air = data['freq']['AIR']
        if sub > 25: t_val = "BOOMY / DEEP"
        elif air > 15: t_val = "BRIGHT / AIRY"
        elif sub < 10 and air < 8: t_val = "MID-FOCUSED"
        else: t_val = "NEUTRAL"

        # 3. IMAGE
        corr = data['corr']
        if corr < 0.3: i_val = "PHASE ISSUE"
        elif corr < 0.6: i_val = "WIDE"
        else: i_val = "CENTERED"
        
        # 4. LOUDNESS
        lufs = data['lufs']
        if lufs > -9: l_val = "LOUD / CD"
        elif lufs < -16: l_val = "GENTLE"
        else: l_val = "STREAMING"
        
        return d_val, t_val, i_val, l_val

# ==============================================================================
# MODULE 3: PIPELINE
# ==============================================================================

def run_analysis_pipeline(file_path):
    # Extract Metadata First
    meta_sr, meta_quality, meta_fmt = MasteringDSP.get_file_metadata(file_path)
    
    # Load for DSP
    y, sr = MasteringDSP.load_audio(file_path)
    
    try:
        lufs, true_peak, plr = MasteringDSP.analyze_loudness_dynamics(y, sr)
        corr, width = MasteringDSP.analyze_stereo_image(y)
        freq_dist = MasteringDSP.analyze_frequency_spectrum(y, sr)
        bpm, groove = MasteringDSP.analyze_precision_rhythm(y, sr)
        key = MasteringDSP.analyze_musical_key(y, sr)
        
        data = {
            "lufs": lufs, "true_peak": true_peak, "plr": plr,
            "corr": corr, "width": width,
            "freq": freq_dist,
            "bpm": bpm, "groove": groove, "key": key,
            "meta": {"sr": meta_sr, "quality": meta_quality, "fmt": meta_fmt}
        }
        
        report = AIAdviser.generate_report(data)
        matrix = AIAdviser.generate_matrix(data)
        
        return data, report, matrix
        
    finally:
        del y
        gc.collect()

# ==============================================================================
# MODULE 4: DASHBOARD UI
# ==============================================================================

def configure_ui():
    st.set_page_config(page_title="Re:finder Pro", page_icon="🎚️", layout="wide")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;600;800&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top left, #1a1a2e 0%, #0a0a0a 100%);
            color: #E0E0E0;
        }
        .app-title { font-size: 3.5rem; font-weight: 900; letter-spacing: -2px; color: #FFF; margin:0; text-shadow: 0 0 20px rgba(0, 188, 212, 0.3); }
        .app-subtitle { font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #00bcd4; margin-bottom: 40px; letter-spacing: 2px; text-transform: uppercase; font-weight: 700; }
        
        .panel {
            background: rgba(25, 25, 35, 0.7); backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 25px; margin-bottom: 20px; height: 100%;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); transition: all 0.3s ease;
        }
        .panel:hover { border-color: rgba(0, 188, 212, 0.4); box-shadow: 0 15px 40px rgba(0, 188, 212, 0.1); transform: translateY(-3px); }
        .panel-header { font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #888; text-transform: uppercase; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-bottom: 20px; letter-spacing: 1px; font-weight: 600; }
        
        .big-val { font-size: 3.2rem; font-weight: 900; color: #FFF; line-height: 1; letter-spacing: -1.5px; text-shadow: 0 0 10px rgba(255, 255, 255, 0.2); }
        .unit { font-size: 1.1rem; color: #666; font-weight: 500; margin-left: 5px; }
        .sub-metric { font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #AAA; margin-top: 10px; display: flex; align-items: center; }
        
        /* Tech Specs */
        .specs-bar { display: flex; justify-content: space-between; align-items: center; background: rgba(0, 188, 212, 0.1); border: 1px solid rgba(0, 188, 212, 0.3); border-radius: 8px; padding: 12px 20px; margin-bottom: 20px; font-family: 'JetBrains Mono'; color: #00bcd4; }
        .specs-label { font-weight: 700; margin-right: 10px; }
        .specs-val { color: #FFF; margin-right: 20px; }
        
        /* Spectrum */
        .freq-row { display: flex; align-items: center; margin-bottom: 12px; font-family: 'JetBrains Mono'; font-size: 0.8rem; }
        .freq-label { width: 65px; color: #888; font-weight: 600; }
        .freq-bar-bg { flex-grow: 1; height: 10px; background: rgba(0, 0, 0, 0.3); border-radius: 5px; overflow: hidden; margin: 0 15px; box-shadow: inset 0 2px 5px rgba(0,0,0,0.5); }
        .freq-bar-fill { height: 100%; border-radius: 5px; background: linear-gradient(90deg, #00bcd4, #7b1fa2); box-shadow: 0 0 15px rgba(0, 188, 212, 0.6); transition: width 0.6s ease; }
        .freq-val { width: 45px; text-align: right; color: #FFF; font-weight: 700; }
        
        /* Report Cards */
        .report-card { background: rgba(30, 30, 40, 0.6); border-left: 4px solid #555; padding: 15px; margin-bottom: 10px; border-radius: 8px; backdrop-filter: blur(5px); }
        .report-PASS { border-left-color: #00e676; background: linear-gradient(90deg, rgba(0,230,118,0.1), transparent); }
        .report-WARN { border-left-color: #ffea00; background: linear-gradient(90deg, rgba(255,234,0,0.1), transparent); }
        .report-CRIT { border-left-color: #ff1744; background: linear-gradient(90deg, rgba(255,23,68,0.1), transparent); }
        .report-INFO { border-left-color: #2979ff; background: linear-gradient(90deg, rgba(41,121,255,0.1), transparent); }
        .report-title { font-weight: 800; color: #FFF; font-size: 0.95rem; margin-bottom: 6px; display: flex; align-items: center; }
        .report-msg { font-size: 0.9rem; color: #CCC; line-height: 1.5; }
        .status-icon { margin-right: 10px; font-size: 1.2rem; }
        
        /* Matrix Box */
        .matrix-box { text-align: center; padding: 15px 10px; border-radius: 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); }
        .matrix-label { font-family: 'JetBrains Mono'; font-size: 0.75rem; color: #00bcd4; margin-bottom: 8px; letter-spacing: 1px; font-weight: 700; }
        .matrix-val { font-size: 1.2rem; font-weight: 900; color: #FFF; margin-bottom: 6px; }
        .matrix-sub { font-size: 0.7rem; color: #666; font-family: 'JetBrains Mono'; font-style: italic; }
        
        /* Disclaimer */
        .disclaimer { font-family: 'Inter', sans-serif; font-size: 0.8rem; color: #666; text-align: center; margin-top: 20px; border-top: 1px solid #333; padding-top: 20px; }

        div.stButton > button { background: linear-gradient(45deg, #00bcd4, #0097a7); color: #FFF; width: 100%; border: none; padding: 18px; font-weight: 800; border-radius: 8px; font-family: 'JetBrains Mono'; letter-spacing: 1px; box-shadow: 0 5px 20px rgba(0, 188, 212, 0.3); transition: all 0.3s ease; }
        div.stButton > button:hover { box-shadow: 0 8px 25px rgba(0, 188, 212, 0.5); transform: translateY(-2px); }
    </style>
    """, unsafe_allow_html=True)

def main():
    configure_ui()
    
    st.markdown('<div class="app-title">Re:finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">/// MASTERING GRADE DSP & AI DIAGNOSTICS</div>', unsafe_allow_html=True)

    if 'last_file_id' not in st.session_state: st.session_state['last_file_id'] = None
    
    # Updated File Uploader with FLAC
    uploaded_file = st.file_uploader("DROP AUDIO MASTER (WAV/MP3/FLAC)", type=["mp3", "wav", "flac"])
    
    if uploaded_file:
        current_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state['last_file_id'] != current_id:
            st.session_state['last_file_id'] = current_id
            st.session_state['result'] = None
            st.rerun()
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.session_state.get('result') is None:
            st.write("")
            if st.button("ENGAGE ANALYSIS ENGINE"):
                with st.spinner("PROCESSING SIGNAL CHAIN..."):
                    try:
                        data, report, matrix = run_analysis_pipeline(tmp_path)
                        st.session_state['result'] = {"data": data, "report": report, "matrix": matrix, "filename": uploaded_file.name}
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
    else:
        if st.session_state['last_file_id'] is not None:
            st.session_state['last_file_id'] = None
            st.session_state['result'] = None
            st.rerun()

    # --- RESULTS DASHBOARD ---
    if st.session_state.get('result'):
        d = st.session_state['result']['data']
        report = st.session_state['result']['report']
        matrix = st.session_state['result']['matrix']
        fname = st.session_state['result'].get('filename', 'Unknown Track')
        
        st.write("")
        st.markdown("---")
        st.write("")
        
        # ROW 0: FILE TECH SPECS (NEW)
        st.markdown(f"""
        <div class="specs-bar">
            <div><span class="specs-label">FILE:</span> <span class="specs-val">{fname}</span></div>
            <div>
                <span class="specs-label">SAMPLE RATE:</span> <span class="specs-val">{d['meta']['sr']} Hz</span>
                <span class="specs-label">QUALITY:</span> <span class="specs-val">{d['meta']['quality']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ROW 1: CORE METRICS
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">LOUDNESS</div>
                <div class="big-val">{d['lufs']} <span class="unit">LUFS</span></div>
                <div class="sub-metric"><i>▶</i> TRUE PEAK: {d['true_peak']} dBTP</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">STEREO IMAGE</div>
                <div class="big-val">{d['corr']} <span class="unit">CORR</span></div>
                <div class="sub-metric"><i>▶</i> WIDTH: {d['width']}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">MUSICALITY</div>
                <div class="big-val">{d['bpm']} <span class="unit">BPM</span></div>
                <div class="sub-metric"><i>▶</i> KEY: {d['key']}</div>
            </div>""", unsafe_allow_html=True)
            
        # ROW 2: SPECTRUM & AI REPORT
        c_spec, c_report = st.columns([1, 1])
        
        with c_spec:
            st.markdown(f"""
            <div class="panel" style="height:100%">
                <div class="panel-header">SPECTRAL BALANCE</div>
                <div class="freq-row"><div class="freq-label">AIR</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['AIR']}%;"></div></div><div class="freq-val">{int(d['freq']['AIR'])}%</div></div>
                <div class="freq-row"><div class="freq-label">HI-MID</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['HIGH_MID']}%;"></div></div><div class="freq-val">{int(d['freq']['HIGH_MID'])}%</div></div>
                <div class="freq-row"><div class="freq-label">LO-MID</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['LOW_MID']}%;"></div></div><div class="freq-val">{int(d['freq']['LOW_MID'])}%</div></div>
                <div class="freq-row"><div class="freq-label">BASS</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['BASS']}%;"></div></div><div class="freq-val">{int(d['freq']['BASS'])}%</div></div>
                <div class="freq-row"><div class="freq-label">SUB</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['SUB']}%;"></div></div><div class="freq-val">{int(d['freq']['SUB'])}%</div></div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_report:
            st.markdown(f"""
            <div class="panel" style="height:100%; overflow-y: auto;">
                <div class="panel-header">🤖 AI DIAGNOSTIC REPORT</div>
            """, unsafe_allow_html=True)
            
            for status, title, msg in report:
                icon = "✅" if status == "PASS" else "⚠️" if status == "WARN" else "🛑" if status == "CRIT" else "ℹ️"
                st.markdown(f"""
                <div class="report-card report-{status}">
                    <div class="report-title"><span class="status-icon">{icon}</span> {title}</div>
                    <div class="report-msg">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ROW 3: SUMMARY MATRIX & DISCLAIMER
        st.write("")
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">📌 AI FEEDBACK SUMMARY (KEY POINTS)</div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="matrix-box">
                <div class="matrix-label">DYNAMICS</div>
                <div class="matrix-val">{matrix[0]}</div>
                <div class="matrix-sub">PLR: <8 (Dense) ~ >14 (Dynamic)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="matrix-box">
                <div class="matrix-label">TONAL BALANCE</div>
                <div class="matrix-val">{matrix[1]}</div>
                <div class="matrix-sub">Spectrum: Sub-Bass vs Air</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="matrix-box">
                <div class="matrix-label">STEREO FIELD</div>
                <div class="matrix-val">{matrix[2]}</div>
                <div class="matrix-sub">Corr: <0.3 (Phase) ~ >0.6 (Mono)</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="matrix-box">
                <div class="matrix-label">LOUDNESS TYPE</div>
                <div class="matrix-val">{matrix[3]}</div>
                <div class="matrix-sub">LUFS: >-9 (Loud) ~ <-16 (Gentle)</div>
            </div>""", unsafe_allow_html=True)
        
        # DISCLAIMER
        st.markdown("""
            <div class="disclaimer">
                Caution: AI로 측정한 수치이며, 오디오는 주관적인 영역임을 명시해주십시오.
            </div>
        """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
