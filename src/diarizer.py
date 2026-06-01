# Token-Free Speaker Diarization Module
import logging
from math import gcd
from collections import defaultdict
import numpy as np
import soundfile as sf
import onnxruntime as ort
import librosa
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from faster_whisper.vad import SileroVAD

# ロガーの初期化（メインプロセスと連動）
logger = logging.getLogger(__name__)

class TokenFreeDiarizer:
    def __init__(self, onnx_model_path: str):
        """
        HFトークン不要・完全ポータブル仕様の話者分離モジュール（堅牢化版）
        :param onnx_model_path: WeSpeaker等の話者埋め込みONNXモデルへのパス
        """
        # 1. ONNX Runtimeの初期化と形状検証 (問題Dへの対応)
        try:
            self.ort_session = ort.InferenceSession(
                onnx_model_path, 
                providers=['CPUExecutionProvider']
            )
            input_info = self.ort_session.get_inputs()[0]
            logger.info(
                f"話者識別ONNXモデル読込完了: name={input_info.name}, "
                f"shape={input_info.shape}, type={input_info.type}"
            )
        except Exception as e:
            raise RuntimeError(f"話者識別ONNXモデルの読み込みに失敗しました: {e}")

        # 2. SileroVADの初期化とシグネチャ・動作確認テスト (問題Aへの対応)
        try:
            self.vad_model = SileroVAD()
            # ダミー音声（16kHz、1秒間の無音）を流してAPIが正常に叩けるかテスト
            dummy_audio = np.zeros(16000, dtype=np.float32)
            _ = self.vad_model.get_speech_timestamps(
                dummy_audio,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=500,
                sampling_rate=16000
            )
            logger.info("SileroVADの正常動作を確認しました。")
        except Exception as e:
            raise RuntimeError(
                f"SileroVADの初期化または動作テストに失敗しました。 "
                f"faster-whisperのバージョンに互換性がない可能性があります: {e}"
            )

    def _load_and_resample_audio(self, audio_path: str, target_sr: int = 16000) -> np.ndarray:
        """動画・Demucs由来の音声（44.1kHz等）を自動で16kHzモノラル/float32にクレンジングする"""
        audio_data, sample_rate = sf.read(audio_path, dtype="float32")
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)
            
        if sample_rate != target_sr:
            g = gcd(target_sr, sample_rate)
            audio_data = resample_poly(audio_data, target_sr // g, sample_rate // g)
            
        return audio_data

    def _compute_fbank(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        WeSpeaker ONNXが要求するFBANK（メルスペクトログラム）特徴量を計算する
        出力形状: (1, T, 80)
        """
        min_samples = 400  # win_length (25ms)
        
        # 境界値で稀に発生する短すぎるチャンクへのパディング対策 (問題Bへの対応)
        if len(audio_chunk) < min_samples:
            audio_chunk = np.pad(audio_chunk, (0, min_samples - len(audio_chunk)), mode='constant')
        
        mel = librosa.feature.melspectrogram(
            y=audio_chunk, sr=sr, n_mels=80,
            n_fft=512, hop_length=160, win_length=400,
            center=False
        )
        
        log_mel = librosa.power_to_db(mel).T  # 形状: (T, 80)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
        
        return log_mel[np.newaxis].astype(np.float32)

    def _extract_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """音声チャンクからFBANKを経由して声紋特徴量を抽出する"""
        feats = self._compute_fbank(audio_chunk)
        
        inputs = {self.ort_session.get_inputs()[0].name: feats}
        outputs = self.ort_session.run(None, inputs)
        
        embedding = outputs[0][0]
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def _estimate_n_speakers(self, embeddings: np.ndarray, min_speakers: int, max_speakers: int) -> int:
        """シルエットスコアを用いて、データから数学的に最適な話者数を動的に割り出す"""
        if len(embeddings) <= 2:
            return 1
            
        best_score = -1
        best_n = min_speakers
        upper = min(max_speakers + 1, len(embeddings))
        
        for n in range(max(2, min_speakers), upper):
            cl = AgglomerativeClustering(n_clusters=n, metric='cosine', linkage='average')
            labels = cl.fit_predict(embeddings)
            
            if len(set(labels)) < 2:
                continue
                
            score = silhouette_score(embeddings, labels, metric='cosine')
            if score > best_score:
                best_score = score
                best_n = n
                
        return best_n

    def process(self, audio_path: str, min_speakers: int = 1, max_speakers: int = 6) -> list[dict]:
        """
        完全ローカル・トークン不要の話者分離メインパイプライン（型ヒント・ロギング強化版）(問題Cへの対応)
        """
        # 1. 音声のロードと16kHzモノラル化
        audio_data = self._load_and_resample_audio(audio_path, target_sr=16000)
        logger.info(f"音声読込＆リサンプリング完了: {len(audio_data)/16000:.1f}秒")
        
        # 2. Silero VADによる発話区間抽出
        speech_chunks = self.vad_model.get_speech_timestamps(
            audio_data,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            sampling_rate=16000
        )
        logger.info(f"VAD検出区間数: {len(speech_chunks)}区間")
        
        if not speech_chunks:
            logger.warning("発話区間が1つも検出されませんでした。")
            return []
            
        embeddings = []
        valid_segments = []
        
        # 3. 各発話区間から声紋特徴量を抽出
        for chunk in speech_chunks:
            start_idx = chunk['start']
            end_idx = chunk['end']
            
            # 0.5秒未満の極端に短いセグメントはクラスタリングを乱すためスキップ
            if (end_idx - start_idx) / 16000 < 0.5:
                continue
                
            audio_segment = audio_data[start_idx:end_idx]
            embedding = self._extract_embedding(audio_segment)
            
            embeddings.append(embedding)
            valid_segments.append({
                'start': start_idx / 16000,
                'end': end_idx / 16000
            })
            
        if not embeddings:
            logger.warning("有効な長さ（0.5秒以上）を持つ発話セグメントがありませんでした。")
            return []
            
        # 4. 最適な話者数の動的推定
        estimated_n = self._estimate_n_speakers(np.array(embeddings), min_speakers, max_speakers)
        logger.info(f"シルエットスコアによる自動推定話者数: {estimated_n}名")
        
        # 5. 最終クラスタリングの執行
        clustering = AgglomerativeClustering(n_clusters=estimated_n, metric='cosine', linkage='average')
        labels = clustering.fit_predict(np.array(embeddings))
        
        # 6. 結果のパッケージング
        diarized_segments = []
        for seg, label in zip(valid_segments, labels):
            seg['speaker'] = f"SPEAKER_{label:02d}"
            diarized_segments.append(seg)
            
        # 7. 話者別累積時間のログ出力（NoiseFilter / ガヤ排除へのバトンタッチ確認用）
        duration_map = defaultdict(float)
        for seg in diarized_segments:
            duration_map[seg['speaker']] += seg['end'] - seg['start']
            
        logger.info("--- 話者別 累積発話時間インデックス ---")
        for spk, dur in sorted(duration_map.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {spk}: {dur:.2f}秒")
        logger.info("---------------------------------------")
            
        return diarized_segments
