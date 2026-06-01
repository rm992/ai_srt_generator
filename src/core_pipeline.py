# src/core_pipeline.py
import os
import logging
from faster_whisper import WhisperModel

from src.audio_processor import AudioProcessor
from src.diarizer import TokenFreeDiarizer
from src.noise_filter import SubtitleNoiseFilter
from src.txt_validator import SubtitleTextValidator

logger = logging.getLogger(__name__)

class FullAutomaticSubtitlePipeline:
    def __init__(self, whisper_model_size: str = "large-v3", onnx_speaker_path: str = "models/speaker/wespeaker_resnet34.onnx"):
        """
        すべてのAIモデルと例外管理ロジックを統括する中央管制パイプラインクラス
        """
        logger.info("🤖 AIハイブリッド・字幕生成パイプラインの初期化を開始します...")
        
        # 1. 各特化型モジュールのマウント
        self.audio_processor = AudioProcessor(ffmpeg_path="ffmpeg")
        self.diarizer = TokenFreeDiarizer(onnx_model_path=onnx_speaker_path)
        self.noise_filter = SubtitleNoiseFilter(max_main_speakers=2, min_duration_threshold=1.5)
        self.txt_validator = SubtitleTextValidator()
        
        # 2. Faster-Whisperのロード（ポータブル環境向けにCPU/int8で安全かつ超高速化）
        logger.info(f"🧠 Faster-Whisper ({whisper_model_size}) をロード中...")
        self.whisper_model = WhisperModel(
            whisper_model_size, 
            device="cpu", 
            compute_type="int8",
            download_root="models/whisper" # モデルの重みをアプリ内に強制隔離
        )
        logger.info("🎉 すべてのAIコンポーネントが正常に起動しました。準備完了。")

    def _format_time_srt(self, seconds: float) -> str:
        """秒数をSRT標準のタイムコード形式 (HH:MM:SS,mmm) に変換する数理ヘルパー"""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        mils = int(round((seconds % 1) * 1000))
        if mils == 1000:
            secs += 1
            mils = 0
        return f"{hrs:02d}:{mins:02d}:{secs:02d},{mils:03d}"

    def _generate_srt_string(self, final_segments: list[dict]) -> str:
        """確定した字幕セグメントリストから、物理SRTテキストを構築する"""
        srt_lines = []
        for idx, seg in enumerate(final_segments, 1):
            start_str = self._format_time_srt(seg['start'])
            end_str = self._format_time_srt(seg['end'])
            
            # 視認性向上のため、話者ラベルを隠蔽、またはデバッグ用に残すことも可能
            # ここではプロ仕様の純粋なテキストのみを出力
            text = seg['text']
            
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start_str} --> {end_str}")
            srt_lines.append(f"{text}\n")
            
        return "\n".join(srt_lines)

    def run(self, video_path: str, output_srt_path: str):
        """
        動画を1本投入するだけで、全自動でSRTを吐き出すメインランナー
        """
        logger.info(f"⚡ パイプライン実行開始対象: {video_path}")
        temp_wav = "temp_vocal_clean.wav"
        
        try:
            # Step 1: 音声抽出 ＆ 強力FFmpeg信号処理
            cleaned_audio = self.audio_processor.extract_and_clean_vocal(video_path, temp_wav)
            
            # Step 2: トークンフリー話者分離（生データの生成）
            raw_speaker_segments = self.diarizer.process(cleaned_audio)
            
            # Step 3: あなたの数理ロジックによるガヤ・笑い声・重複の一括自動棄却
            clean_time_slots = self.noise_filter.execute(raw_speaker_segments)
            
            if not clean_time_slots:
                logger.warning("ガヤ排除の結果、有効な発話区間が残りませんでした。処理を停止します。")
                return

            # Step 4: 研ぎ澄まされた確定枠だけを Faster-Whisper の `clip_timestamps` で狙い撃ち
            logger.info("🎙️ メインキャストの確定タイムライン枠へ、ピンポイント音声認識を執行します...")
            
            # clip_timestamps用に [(start1, end1), ...] の形に変形
            clips = [(seg['start'], seg['end']) for seg in clean_time_slots]
            
            # condition_on_previous_text=False でハルシネーションの連鎖（ループ）の芽を完全に摘む
            whisper_segments, _ = self.whisper_model.transcribe(
                cleaned_audio,
                beam_size=5,
                language="ja",
                condition_on_previous_text=False,
                clip_timestamps=clips
            )
            
            # 生成ジェネレータから実リストへマウント
            transcribed_segments = []
            for seg, slot in zip(whisper_segments, clean_time_slots):
                transcribed_segments.append({
                    'start': seg.start,
                    'end': seg.end,
                    'text': seg.text.strip()
                })
            
            # Step 5: 幻聴（ハルシネーション）自動検閲防壁の執行
            final_subtitle_segments = self.txt_validator.filter_segments(transcribed_segments)
            
            # Step 6: SRTファイルへの物理書き出し
            if final_subtitle_segments:
                srt_string = self._generate_srt_string(final_subtitle_segments)
                with open(output_srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_string)
                logger.info(f"🎉 完全にシンクロしたプロ品質の字幕ファイルを自動生成しました: {output_srt_path}")
            else:
                logger.warning("検閲の結果、書き出す字幕テキストがありませんでした。")

        finally:
            # ユーザーのディスクを圧迫しないよう、巨大な一時WAVファイルは自律的に消去（美学）
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                logger.info("🧹 一時音声ファイルを安全にクリーンアップしました。")
