import streamlit as st
import librosa
import librosa.display
import numpy as np
import scipy.signal
import scipy.fft
import soundfile as sf
import tempfile
import os
import gc
import matplotlib.pyplot as plt
import io

# ==============================================================================
# MODULE 1: DSP KERNEL
# ==============================================================================

class MasteringDSP:
    @staticmethod
    def load_audio(file_path):
        try:
            duration = librosa.get_duration(path=file_path)
            # Intro Skip Logic (Analyze main part)
            start_time = 30.0 if duration > 40.0 else 0.0
            
            y, sr = librosa.load(file_path, sr=44100, mono=False, offset=start_time, duration=180)
            if y.ndim == 1: y = np.stack((y, y))
            return y, sr
        except Exception as e:
            raise RuntimeError(f"Critical Load Error: {e}")

    @staticmethod
    def get_file_metadata(file_path):
        try:
            with sf.SoundFile(file_path) as f:
                sr = f.samplerate
                subtype = f.subtype
                fmt = f.format
                if 'PCM' in subtype: quality = f"{subtype.split('_')[-1]}-bit INT"
                elif 'FLOAT' in subtype: quality = "32-bit FLOAT"
                elif 'MPEG' in subtype: quality = "Lossy (MP3)" 
                else: quality = subtype
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
        
        return round(float(lufs_approx), 1), round(float(true_peak_db), 1), round(float(plr), 1)

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
        
        return round(float(correlation), 2), round(float(width_percent), 1)

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
            energy_dist[name] = float((band_energy / total_energy) * 100)
            
        return energy_dist

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

    @staticmethod
    def generate_spectrogram(y_stereo, sr):
        y_mono = librosa.to_mono(y_stereo)
        if len(y_mono) > sr * 30: 
            y_vis = y_mono[:sr*30] 
        else:
            y_vis = y_mono
            
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_vis)), ref=np.max)
        
        plt.figure(figsize=(10, 3), facecolor='none') 
        ax = plt.axes()
        ax.set_facecolor('none')
        
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', cmap='ocean', ax=ax)
        
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
        plt.close()
        buf.seek(0)
        return buf

# ==============================================================================
# MODULE 2: AI ADVISER (KOREAN V1.5)
# ==============================================================================

class AIAdviser:
    @staticmethod
    def generate_report(data):
        advice_list = []
        if data['lufs'] > -7:
            advice_list.append(("WARN", "High Loudness Level", "음압이 매우 높습니다(-7 LUFS 초과). 클럽/CD 마스터링 의도가 아니라면 다이내믹 레인지 손실을 체크하세요."))
        else:
            advice_list.append(("PASS", "Safe Loudness Range", "다이내믹 레인지가 보존된 안전한 음압 범위 내에 있습니다."))

        if data['corr'] < 0.2:
            advice_list.append(("CRIT", "Phase Cancellation Risk", "위상 상관도가 낮습니다. 모노 환경(스마트폰 등)에서 소리가 사라질 수 있습니다."))
        elif data['corr'] < 0.6:
            advice_list.append(("INFO", "Wide Stereo Image", "넓은 스테레오 이미지를 가지고 있습니다."))
        else:
            advice_list.append(("PASS", "Solid Phase Coherence", "위상 호환성이 좋으며 단단한 믹스 밸런스를 유지하고 있습니다."))

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
        if data['plr'] < 8: d_val = "COMPRESSED"
        elif data['plr'] > 14: d_val = "DYNAMIC"
        else: d_val = "BALANCED"

        sub = data['freq']['SUB']
        air = data['freq']['AIR']
        if sub > 25: t_val = "BOOMY / DEEP"
        elif air > 15: t_val = "BRIGHT / AIRY"
        elif sub < 10 and air < 8: t_val = "MID-FOCUSED"
        else: t_val = "NEUTRAL"

        corr = data['corr']
        if corr < 0.3: i_val = "PHASE ISSUE"
        elif corr < 0.6: i_val = "WIDE"
        else: i_val = "CENTERED"
        
        lufs = data['lufs']
        if lufs > -9: l_val = "LOUD / CD"
        elif lufs < -16: l_val = "GENTLE"
        else: l_val = "STREAMING"
        
        return d_val, t_val, i_val, l_val

# ==============================================================================
# MODULE 3: PIPELINE
# ==============================================================================

def run_analysis_pipeline(file_path):
    meta_sr, meta_quality, meta_fmt = MasteringDSP.get_file_metadata(file_path)
    y, sr = MasteringDSP.load_audio(file_path)
    
    try:
        lufs, true_peak, plr = MasteringDSP.analyze_loudness_dynamics(y, sr)
        corr, width = MasteringDSP.analyze_stereo_image(y)
        freq_dist = MasteringDSP.analyze_frequency_spectrum(y, sr)
        key = MasteringDSP.analyze_musical_key(y, sr)
        spec_img = MasteringDSP.generate_spectrogram(y, sr)
        
        data = {
            "lufs": lufs, "true_peak": true_peak, "plr": plr,
            "corr": corr, "width": width,
            "freq": freq_dist,
            "key": key,
            "meta": {"sr": meta_sr, "quality": meta_quality, "fmt": meta_fmt},
            "spec_img": spec_img
        }
        
        report = AIAdviser.generate_report(data)
        matrix = AIAdviser.generate_matrix(data)
        return data, report, matrix
    finally:
        del y
        gc.collect()

# ==============================================================================
# MODULE 4: DASHBOARD UI (iOS Pastel Mint + KOREAN TEXT)
# ==============================================================================

def configure_ui():
    st.set_page_config(page_title="Re:finder Pro", page_icon="🎚️", layout="wide")
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Inter:wght@200;300;400;500;600;700;800&display=swap');
        
        :root {
            --mint-primary: #4FD1C5;
            --mint-secondary: #81F7E5;
            --mint-glow: rgba(79, 209, 197, 0.5);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(79, 209, 197, 0.15);
            --text-main: #F5F5F7;
            --text-sub: #86868b;
            --bg-deep: #0a0f12;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: radial-gradient(circle at 20% 20%, rgba(26, 188, 156, 0.15) 0%, rgba(10, 15, 20, 0) 50%),
                        radial-gradient(circle at 80% 80%, rgba(79, 209, 197, 0.1) 0%, rgba(10, 15, 20, 0) 50%),
                        #0a0f12;
            color: var(--text-main);
            scroll-behavior: smooth;
        }
        
        /* TYPOGRAPHY */
        .app-title { font-size: 3.5rem; font-weight: 800; letter-spacing: -1.5px; color: var(--text-main); margin:0; }
        .app-subtitle { font-family: 'JetBrains Mono'; font-size: 0.85rem; color: var(--mint-primary); letter-spacing: 1.5px; text-transform: uppercase; font-weight: 600; display: flex; align-items: center; }
        .app-subtitle::before { content: ''; display: inline-block; width: 8px; height: 8px; background: var(--mint-primary); border-radius: 50%; margin-right: 10px; box-shadow: 0 0 10px var(--mint-primary); }

        /* HEADER LAYOUT */
        .header-container { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 40px; padding-top: 20px; }
        .guide-text { font-family: 'Inter'; font-size: 0.85rem; color: var(--text-main); text-align: right; line-height: 1.5; }
        .guide-sub { font-family: 'Inter'; font-size: 0.7rem; color: var(--mint-primary); text-align: right; opacity: 0.8; margin-top: 4px; }

        /* iOS GLASSMORPHISM PANELS */
        .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(30px) saturate(150%);
            -webkit-backdrop-filter: blur(30px) saturate(150%);
            border: 1px solid var(--glass-border);
            border-radius: 28px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        .panel:hover {
            border-color: rgba(79, 209, 197, 0.3);
            box-shadow: 0 12px 40px rgba(79, 209, 197, 0.1);
            transform: translateY(-2px);
        }
        
        .panel-header {
            font-family: 'Inter'; font-size: 0.8rem; color: var(--text-sub); text-transform: uppercase;
            letter-spacing: 1px; font-weight: 600; display: flex; align-items: center; margin-bottom: 15px;
        }
        
        /* TOOLTIPS */
        .tooltip { position: relative; display: inline-block; cursor: help; color: var(--mint-primary); border-bottom: 1px dotted var(--mint-primary); transition: all 0.3s; }
        .tooltip:hover { color: var(--mint-secondary); border-bottom-style: solid; }
        .tooltip .tooltiptext {
            visibility: hidden; width: 260px; background-color: rgba(20, 25, 30, 0.95); color: var(--text-main); text-align: left;
            border-radius: 16px; padding: 15px; position: absolute; z-index: 100;
            bottom: 135%; left: 50%; margin-left: -130px; opacity: 0; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            font-family: 'Inter', sans-serif; font-size: 0.8rem; font-weight: 400; line-height: 1.5;
            border: 1px solid var(--glass-border); box-shadow: 0 10px 30px rgba(0,0,0,0.5); text-transform: none;
            backdrop-filter: blur(10px);
        }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; transform: translateY(0); }
        
        /* METRICS */
        .big-val { font-size: 3.2rem; font-weight: 200; color: var(--text-main); line-height: 1; letter-spacing: -1px; }
        .unit { font-size: 1.0rem; color: var(--text-sub); font-weight: 400; margin-left: 6px; }
        .sub-metric { font-family: 'Inter'; font-size: 0.85rem; color: var(--text-sub); margin-top: 12px; display: flex; align-items: center; font-weight: 500; }
        .sub-metric i { color: var(--mint-primary); margin-right: 8px; font-size: 0.7rem; }
        
        /* FREQUENCY BARS */
        .freq-row { display: flex; align-items: center; margin-bottom: 10px; font-family: 'JetBrains Mono'; font-size: 0.75rem; }
        .freq-label { width: 55px; color: var(--text-sub); font-weight: 600; }
        .freq-bar-bg { flex-grow: 1; height: 8px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; margin: 0 12px; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3); }
        .freq-bar-fill {
            height: 100%; border-radius: 4px;
            background: linear-gradient(90deg, var(--mint-primary), var(--mint-secondary));
            box-shadow: 0 0 10px var(--mint-glow);
            transition: width 0.5s ease-out;
        }
        .freq-val { width: 40px; text-align: right; color: var(--text-main); font-weight: 700; }
        
        /* REPORTS */
        .report-card { background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-left-width: 4px; padding: 12px 15px; margin-bottom: 8px; border-radius: 12px; }
        .report-title { font-weight: 600; color: var(--text-main); font-size: 0.9rem; margin-bottom: 4px; }
        .report-msg { font-size: 0.85rem; color: var(--text-sub); line-height: 1.4; }
        .report-PASS { border-left-color: var(--mint-secondary); background: linear-gradient(90deg, rgba(79, 209, 197, 0.05), transparent); }
        .report-WARN { border-left-color: #FFD54F; }
        .report-CRIT { border-left-color: #FF5252; }
        .report-INFO { border-left-color: #448AFF; }
        
        .matrix-box { text-align: center; padding: 18px 10px; border-radius: 20px; background: rgba(255,255,255,0.02); border: 1px solid var(--glass-border); transition: all 0.3s; }
        .matrix-box:hover { background: rgba(79, 209, 197, 0.05); border-color: var(--mint-primary); }
        .matrix-label { font-family: 'Inter'; font-size: 0.7rem; color: var(--mint-primary); margin-bottom: 8px; letter-spacing: 1px; font-weight: 700; text-transform: uppercase; }
        .matrix-val { font-size: 1.1rem; font-weight: 700; color: var(--text-main); margin-bottom: 4px; }
        .matrix-sub { font-family: 'Inter'; font-size: 0.7rem; color: var(--text-sub); }
        
        /* HISTORY */
        .history-box {
            background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 15px 20px; border-radius: 16px; margin-bottom: 10px; display: flex; align-items: center; justify-content: space-between; transition: all 0.2s;
        }
        .history-box:hover { border-color: var(--mint-primary); background: rgba(79, 209, 197, 0.03); }
        .history-idx { font-family: 'JetBrains Mono'; color: var(--text-sub); font-size: 1.0rem; font-weight: 700; margin-right: 15px; opacity: 0.5; }
        .history-name { font-weight: 600; color: var(--text-main); font-size: 0.95rem; margin-bottom: 4px; }
        .history-meta { font-family: 'Inter'; font-size: 0.75rem; color: var(--text-sub); font-weight: 500; }
        .history-val { color: var(--mint-primary); font-weight: 600; margin-left: 6px; }

        .disclaimer { font-family: 'Inter', sans-serif; font-size: 0.75rem; color: var(--text-sub); text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.05); opacity: 0.7; }

        /* BUTTONS */
        div.stButton > button {
            background: rgba(79, 209, 197, 0.15);
            color: var(--mint-primary);
            width: 100%; border: 1px solid rgba(79, 209, 197, 0.3);
            padding: 18px; font-weight: 700; border-radius: 16px;
            font-family: 'Inter'; letter-spacing: 0.5px;
            backdrop-filter: blur(10px);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        }
        div.stButton > button:hover {
            background: var(--mint-primary);
            color: #0a0f12;
            border-color: var(--mint-secondary);
            box-shadow: 0 0 20px var(--mint-glow);
            transform: scale(1.02);
        }
        .stFileUploader label { font-family: 'Inter'; font-weight: 500; color: var(--mint-primary); }
        
        .small-btn button {
            padding: 8px 16px !important; font-size: 0.75rem !important; width: auto !important;
            background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(255,255,255,0.1) !important; color: var(--text-sub) !important;
            border-radius: 12px !important;
        }
        .small-btn button:hover {
            background: var(--mint-primary) !important; color: #0a0f12 !important; border-color: var(--mint-primary) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def main():
    configure_ui()
    
    # CUSTOM HEADER WITH FLEXBOX FOR GUIDE
    st.markdown("""
    <div class="header-container">
        <div>
            <div class="app-title">Re:finder</div>
            <div class="app-subtitle">AI & DSP Diagnostics for Everyone who makes Music</div>
        </div>
        <div>
            <div class="guide-text">완성된 곡을 여기에 올려주세요.<br>나머지는 AI가 분석합니다.</div>
            <div class="guide-sub">* 지속적으로 업데이트 예정입니다.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if 'history' not in st.session_state: st.session_state['history'] = []
    if 'last_file_id' not in st.session_state: st.session_state['last_file_id'] = None
    
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
                        full_snapshot = {"data": data, "report": report, "matrix": matrix, "filename": uploaded_file.name}
                        st.session_state['result'] = full_snapshot
                        st.session_state['history'].append(full_snapshot)
                        if len(st.session_state['history']) > 5: st.session_state['history'].pop(0)
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

    # --- MAIN DASHBOARD ---
    if st.session_state.get('result'):
        d = st.session_state['result']['data']
        report = st.session_state['result']['report']
        matrix = st.session_state['result']['matrix']
        fname = st.session_state['result'].get('filename', 'Unknown Track')
        
        st.write("")
        st.write("")
        
        # FILE INFO PANEL
        st.markdown(f"""
        <div class="panel" style="padding: 18px 24px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display:flex; align-items:center;">
                    <div style="width:40px; height:40px; background:rgba(79, 209, 197, 0.1); border-radius:12px; display:flex; align-items:center; justify-content:center; margin-right:15px;">
                        <span style="font-size:1.2rem;">🎵</span>
                    </div>
                    <div>
                        <div style="font-family:'Inter'; font-size:0.75rem; color:var(--text-sub); font-weight:600; margin-bottom:2px;">FILE NAME</div>
                        <div style="color:var(--text-main); font-weight:700; font-size:1.0rem; word-break: break-all;">{fname}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-family:'Inter'; font-size:0.75rem; color:var(--text-sub); font-weight:600; margin-bottom:2px;">FORMAT</div>
                    <div style="font-family:'Inter'; color:var(--mint-primary); font-weight:700; font-size:0.9rem;">{d['meta']['sr']}Hz / {d['meta']['quality'].split('-')[0]}bit</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <div class="tooltip">LOUDNESS<span class="tooltiptext"><b>음압(Loudness)</b><br>곡의 평균적인 볼륨입니다. 상업 음원은 보통 -14 ~ -6 LUFS 사이입니다. 너무 낮으면 작게 들리고, 너무 높으면 소리가 찌그러집니다.</span></div>
                </div>
                <div class="big-val">{d['lufs']} <span class="unit">LUFS</span></div>
                <div class="sub-metric"><i>●</i> TRUE PEAK: {d['true_peak']} dBTP</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <div class="tooltip">STEREO IMAGE<span class="tooltiptext"><b>스테레오 이미지(Stereo)</b><br>소리의 좌우 넓이와 위상입니다. Correlation이 +1이면 정중앙(모노), 0이면 넓음, -1이면 소리가 사라질 위험(역위상)이 있습니다.</span></div>
                </div>
                <div class="big-val">{d['corr']} <span class="unit">CORR</span></div>
                <div class="sub-metric"><i>●</i> WIDTH: {d['width']}%</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="panel">
                <div class="panel-header">
                    <div class="tooltip">MUSICALITY<span class="tooltiptext"><b>조성(Key)</b><br>이 곡의 중심이 되는 음계입니다. 오토튠이나 배음 제어 플러그인을 사용할 때 이 키를 설정해야 불협화음을 막을 수 있습니다.</span></div>
                </div>
                <div class="big-val" style="font-weight:300;">{d['key']}</div>
                <div class="sub-metric"><i>●</i> KEY DETECTION</div>
            </div>""", unsafe_allow_html=True)
            
        c_spec, c_report = st.columns([1, 1])
        with c_spec:
            st.markdown("""
            <div class="panel">
                <div class="panel-header">
                    <div class="tooltip">SPECTRAL BALANCE<span class="tooltiptext"><b>주파수 밸런스</b><br>저음(BASS)부터 고음(AIR)까지 소리의 에너지가 어떻게 분포되어 있는지 보여줍니다.</span></div>
                </div>
            """, unsafe_allow_html=True)
            
            st.image(d['spec_img'], width="stretch")
            
            st.markdown(f"""
                <div style="margin-top:20px;">
                    <div class="freq-row"><div class="freq-label">AIR</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['AIR']}%;"></div></div><div class="freq-val">{int(d['freq']['AIR'])}%</div></div>
                    <div class="freq-row"><div class="freq-label">HI-MID</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['HIGH_MID']}%;"></div></div><div class="freq-val">{int(d['freq']['HIGH_MID'])}%</div></div>
                    <div class="freq-row"><div class="freq-label">LO-MID</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['LOW_MID']}%;"></div></div><div class="freq-val">{int(d['freq']['LOW_MID'])}%</div></div>
                    <div class="freq-row"><div class="freq-label">BASS</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['BASS']}%;"></div></div><div class="freq-val">{int(d['freq']['BASS'])}%</div></div>
                    <div class="freq-row"><div class="freq-label">SUB</div><div class="freq-bar-bg"><div class="freq-bar-fill" style="width:{d['freq']['SUB']}%;"></div></div><div class="freq-val">{int(d['freq']['SUB'])}%</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with c_report:
            st.markdown(f"""
            <div class="panel" style="height:100%; overflow-y: auto;">
                <div class="panel-header">🤖 AI DIAGNOSTIC REPORT</div>
            """, unsafe_allow_html=True)
            for status, title, msg in report:
                icon_color = "var(--mint-secondary)" if status == "PASS" else "#FFD54F" if status == "WARN" else "#FF5252" if status == "CRIT" else "#448AFF"
                st.markdown(f"""
                <div class="report-card report-{status}">
                    <div class="report-title">
                        <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{icon_color}; margin-right:8px;"></span>
                        {title}
                    </div>
                    <div class="report-msg">{msg}</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown(f"""
        <div class="panel">
            <div class="panel-header">
                <div class="tooltip">📌 AI FEEDBACK SUMMARY<span class="tooltiptext"><b>AI 요약</b><br>복잡한 수치들을 한눈에 보기 쉽게 4가지 성향으로 요약했습니다.</span></div>
            </div>
        """, unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="matrix-box"><div class="matrix-label">DYNAMICS</div><div class="matrix-val">{matrix[0]}</div><div class="matrix-sub">PLR: <8 (Dense) ~ >14 (Dynamic)</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="matrix-box"><div class="matrix-label">TONAL BALANCE</div><div class="matrix-val">{matrix[1]}</div><div class="matrix-sub">Spectrum: Sub-Bass vs Air</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="matrix-box"><div class="matrix-label">STEREO FIELD</div><div class="matrix-val">{matrix[2]}</div><div class="matrix-sub">Corr: <0.3 (Phase) ~ >0.6 (Mono)</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="matrix-box"><div class="matrix-label">LOUDNESS TYPE</div><div class="matrix-val">{matrix[3]}</div><div class="matrix-sub">LUFS: >-9 (Loud) ~ <-16 (Gentle)</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # --- HISTORY SECTION ---
        if len(st.session_state['history']) > 0:
            st.write("")
            st.write("")
            st.markdown(f"""
            <div class="panel">
                 <div class="panel-header">📋 RECENT ANALYSIS HISTORY (LOAD SNAPSHOT)</div>
            """, unsafe_allow_html=True)
            
            for i, snapshot in enumerate(reversed(st.session_state['history'])):
                s_name = snapshot['filename']
                s_lufs = snapshot['data']['lufs']
                s_plr = snapshot['data']['plr']
                s_key = snapshot['data']['key']
                
                is_active = (s_name == fname) and (d['lufs'] == s_lufs)
                
                c_info, c_btn = st.columns([5, 1])
                with c_info:
                    active_style = "border-color: var(--mint-primary); background: rgba(79, 209, 197, 0.05);" if is_active else ""
                    st.markdown(f"""
                    <div class="history-box" style="{active_style} margin-bottom:0;">
                        <div style="display:flex; align-items:center;">
                            <span class="history-idx">#{len(st.session_state['history']) - i}</span>
                            <div>
                                <div class="history-name" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 300px;">{s_name}</div>
                                <div>
                                    <span class="history-meta">LUFS <span class="history-val">{s_lufs}</span></span>
                                    <span class="history-meta" style="margin-left:12px">PLR <span class="history-val">{s_plr}</span></span>
                                    <span class="history-meta" style="margin-left:12px">KEY <span class="history-val">{s_key}</span></span>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_btn:
                    st.write("")
                    st.markdown('<div class="small-btn">', unsafe_allow_html=True)
                    if st.button("LOAD", key=f"load_hist_{i}"):
                        st.session_state['result'] = snapshot
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            
        st.markdown("""<div class="disclaimer">Caution: AI로 측정한 수치이며, 오디오는 주관적인 영역임을 명시해주십시오.</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
