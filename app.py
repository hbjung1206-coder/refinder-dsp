import streamlit as st
import librosa
import numpy as np
import scipy.signal
import scipy.fft
import tempfile
import os
import gc

# ==============================================================================
# MODULE 1: DSP KERNEL (PHYSICS & MATH)
# ==============================================================================

class MasteringDSP:
    @staticmethod
    def load_audio(file_path):
        try:
            y, sr = librosa.load(file_path, sr=44100, mono=False, duration=180)
            if y.ndim == 1: y = np.stack((y, y))
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Critical Load Error: {e}")

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
# MODULE 2: AI ADVISER ENGINE
# ==============================================================================

class AIAdviser:
    @staticmethod
    def generate_report(data):
        advice_list = []
        
        # 1. Loudness Check
        if data['lufs'] < -15:
            advice_list.append(("WARN", "Low Loudness", "스트리밍 표준(-14 LUFS)보다 소리가 작습니다. 리미터나 맥시마이저를 사용하여 전체 음압을 2~3dB 더 확보하세요."))
        elif data['lufs'] > -7:
            advice_list.append(("WARN", "Extreme Loudness", "음압이 매우 높습니다(-7 LUFS 초과). 다이내믹 레인지가 손실되었는지 확인하세요."))
        else:
            advice_list.append(("PASS", "Healthy Loudness", "스트리밍 서비스에 최적화된 적절한 음압입니다."))

        # 2. True Peak Check
        if data['true_peak'] > -0.5:
            advice_list.append(("CRIT", "True Peak Clipping Risk", f"True Peak가 {data['true_peak']}dBTP 입니다. 인코딩 시 클리핑 방지를 위해 리미터 Ceiling을 -1.0dBTP로 낮추세요."))
        else:
            advice_list.append(("PASS", "Safe Headroom", "디지털 클리핑으로부터 안전한 헤드룸을 확보했습니다."))

        # 3. Stereo Phase Check
        if data['corr'] < 0.2:
            advice_list.append(("CRIT", "Phase Cancellation", "위상 상관도(Correlation)가 위험 수준입니다. 모노 호환성을 점검하고 이미저 사용을 줄이세요."))
        elif data['corr'] < 0.6:
            advice_list.append(("INFO", "Wide Stereo Image", "스테레오 이미지가 넓은 편입니다. 베이스/킥이 모노 중앙에 잘 위치하는지 확인하세요."))
        else:
            advice_list.append(("PASS", "Solid Phase", "위상 호환성이 좋으며 단단한 믹스입니다."))

        # 4. Frequency Balance Check
        sub_energy = data['freq']['SUB']
        air_energy = data['freq']['AIR']
        
        if sub_energy > 25:
            advice_list.append(("WARN", "Excessive Sub-Bass", f"서브 베이스 에너지가 {int(sub_energy)}%로 높습니다. 부밍(Booming)을 주의하세요."))
        elif sub_energy < 5:
            advice_list.append(("WARN", "Lack of Low-End", "저음역 에너지가 부족하여 믹스가 가볍게 들릴 수 있습니다."))
            
        if air_energy > 15:
            advice_list.append(("WARN", "Harsh High-End", f"고음역 에너지가 {int(air_energy)}%로 높습니다. 치찰음(Sibilance)을 체크하세요."))

        return advice_list

# ==============================================================================
# MODULE 3: PIPELINE
# ==============================================================================

def run_analysis_pipeline(file_path):
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
            "bpm": bpm, "groove": groove, "key": key
        }
        
        report = AIAdviser.generate_report(data)
        return data, report
        
    finally:
        del y
        gc.collect()

# ==============================================================================
# MODULE 4: DASHBOARD UI (VISUAL UPGRADE)
# ==============================================================================

def configure_ui():
    st.set_page_config(page_title="Re:finder Pro", page_icon="🎚️", layout="wide")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;600;800&display=swap');
        
        /* ----- Global Theme ----- */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at top left, #1a1a2e 0%, #0a0a0a 100%); /* Deep Cyberpunk Background */
            color: #E0E0E0;
        }
        
        /* ----- Headers ----- */
        .app-title {
            font-size: 3.5rem; font-weight: 900; letter-spacing: -2px; color: #FFF; margin:0;
            text-shadow: 0 0 20px rgba(0, 188, 212, 0.3); /* Subtle Neon Glow */
        }
        .app-subtitle {
            font-family: 'JetBrains Mono'; font-size: 0.9rem; color: #00bcd4; /* Cyan accent */
            margin-bottom: 40px; letter-spacing: 2px; text-transform: uppercase; font-weight: 700;
        }
        
        /* ----- Pro Panels (Glassmorphism) ----- */
        .panel {
            background: rgba(25, 25, 35, 0.7); /* Semi-transparent */
            backdrop-filter: blur(12px); /* Frost Effect */
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            height: 100%;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3); /* Deep Shadow */
            transition: all 0.3s ease;
        }
        .panel:hover {
            border-color: rgba(0, 188, 212, 0.4); /* Cyan hover glow */
            box-shadow: 0 15px 40px rgba(0, 188, 212, 0.1);
            transform: translateY(-3px);
        }
        .panel-header {
            font-family: 'JetBrains Mono'; font-size: 0.8rem; color: #888;
            text-transform: uppercase; border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 10px; margin-bottom: 20px; letter-spacing: 1px; font-weight: 600;
        }
        
        /* ----- Metrics (Glowing Numbers) ----- */
        .big-val {
            font-size: 3.2rem; font-weight: 900; color: #FFF; line-height: 1; letter-spacing: -1.5px;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
        }
        .unit { font-size: 1.1rem; color: #666; font-weight: 500; margin-left: 5px; }
        .sub-metric {
            font-family: 'JetBrains Mono'; font-size: 0.85rem; color: #AAA; margin-top: 10px;
            display: flex; align-items: center;
        }
        .sub-metric i { margin-right: 8px; color: #00bcd4; } /* Icon accent */
        
        /* ----- Visualizers (Neon Bars) ----- */
        .freq-row { display: flex; align-items: center; margin-bottom: 12px; font-family: 'JetBrains Mono'; font-size: 0.8rem; }
        .freq-label { width: 65px; color: #888; font-weight: 600; }
        .freq-bar-bg {
            flex-grow: 1; height: 10px; background: rgba(0, 0, 0, 0.3);
            border-radius: 5px; overflow: hidden; margin: 0 15px;
            box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        }
        .freq-bar-fill {
            height: 100%; border-radius: 5px;
            /* Cyan to Purple Gradient */
            background: linear-gradient(90deg, #00bcd4, #7b1fa2);
            box-shadow: 0 0 15px rgba(0, 188, 212, 0.6); /* Neon Glow */
            transition: width 0.6s cubic-bezier(0.25, 0.8, 0.25, 1); /* Smooth spring animation */
        }
        .freq-val { width: 45px; text-align: right; color: #FFF; font-weight: 700; }
        
        /* ----- AI Report Cards (Sleek) ----- */
        .report-card {
            background: rgba(30, 30, 40, 0.6); border-left: 4px solid #555;
            padding: 15px; margin-bottom: 10px; border-radius: 8px;
            backdrop-filter: blur(5px);
        }
        .report-PASS { border-left-color: #00e676; background: linear-gradient(90deg, rgba(0,230,118,0.1), transparent); }
        .report-WARN { border-left-color: #ffea00; background: linear-gradient(90deg, rgba(255,234,0,0.1), transparent); }
        .report-CRIT { border-left-color: #ff1744; background: linear-gradient(90deg, rgba(255,23,68,0.1), transparent); }
        .report-INFO { border-left-color: #2979ff; background: linear-gradient(90deg, rgba(41,121,255,0.1), transparent); }
        
        .report-title { font-weight: 800; color: #FFF; font-size: 0.95rem; margin-bottom: 6px; display: flex; align-items: center; }
        .report-msg { font-size: 0.9rem; color: #CCC; line-height: 1.5; }
        .status-icon { margin-right: 10px; font-size: 1.2rem; }
        
        /* ----- Components ----- */
        div.stButton > button {
            background: linear-gradient(45deg, #00bcd4, #0097a7); /* Gradient Button */
            color: #FFF; width: 100%; border: none; padding: 18px;
            font-weight: 800; border-radius: 8px; font-family: 'JetBrains Mono'; letter-spacing: 1px;
            box-shadow: 0 5px 20px rgba(0, 188, 212, 0.3);
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            box-shadow: 0 8px 25px rgba(0, 188, 212, 0.5);
            transform: translateY(-2px);
        }
        .stFileUploader label { font-family: 'JetBrains Mono'; font-weight: 600; color: #00bcd4; }
    </style>
    """, unsafe_allow_html=True)

def main():
    configure_ui()
    
    st.markdown('<div class="app-title">Re:finder</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">/// MASTERING GRADE DSP & AI DIAGNOSTICS</div>', unsafe_allow_html=True)

    if 'last_file_id' not in st.session_state: st.session_state['last_file_id'] = None
    
    uploaded_file = st.file_uploader("DROP AUDIO MASTER (WAV/MP3)", type=["mp3", "wav"])
    
    if uploaded_file:
        current_id = f"{uploaded_file.name}-{uploaded_file.size}"
        if st.session_state['last_file_id'] != current_id:
            st.session_state['last_file_id'] = current_id
            st.session_state['result'] = None
            st.rerun()
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.audio(uploaded_file, format='audio/mp3')
        
        if st.session_state.get('result') is None:
            st.write("")
            if st.button("ENGAGE ANALYSIS ENGINE"):
                with st.spinner("PROCESSING SIGNAL CHAIN..."):
                    try:
                        data, report = run_analysis_pipeline(tmp_path)
                        st.session_state['result'] = {"data": data, "report": report}
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
        
        st.write("")
        st.markdown("---")
        st.write("")
        
        # ROW 1: CORE METRICS (3 Columns)
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
            
        # ROW 2: SPECTRUM & AI REPORT (2 Columns)
        c_spec, c_report = st.columns([1, 1])
        
        with c_spec:
            # Frequency Spectrum Panel with Neon Bars
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
            # Sleek AI Report Panel
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

if __name__ == "__main__":
    main()