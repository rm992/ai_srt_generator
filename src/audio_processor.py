# src/audio_processor.py
import os
import subprocess
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, ffmpeg_path: str):
        """
        同梱FFmpegを利用して動画から音声の分離・クレンジングを行うモジュール
        """
        self.ffmpeg_cmd = "ffmpeg"  # EnvManagerがPATHをジャックしているため、これで同梱版が動く

    def extract_and_clean_vocal(self, video_path: str, output_wav_path: str) -> str:
        """
        動画から音声を抽出し、BGM・低周波ノイズをカットするFFmpegバンドパスフィルターを執行
        """
        logger.info(f"🎬 FFmpegによる音声抽出＆バンドパスフィルターを開始: {video_path}")
        
        if os.path.exists(output_wav_path):
            os.remove(output_wav_path)

        # 80Hz〜8000Hz以外のノイズ（重低音BGMや高周波の金属音）を極限までカット
        # さらに afftdn (インテリジェントノイズ除去) を組み合わせて声を浮き立たせる
        # 16kHz, モノラル(1ch) で出力
        command = [
            self.ffmpeg_cmd, "-y",
            "-i", video_path,
            "-vn",  # 映像を無視
            "-af", "highpass=f=80, lowpass=f=8000, afftdn", # 信号処理防壁
            "-ar", "16000",
            "-ac", "1",
            output_wav_path
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            logger.info(f"✨ 音声クレンジング完了。一時ファイルを生成しました: {output_wav_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpegの執行に失敗しました。詳細: {e.stderr}")
            raise RuntimeError(f"音声前処理に失敗しました: {e.stderr}")

        return output_wav_path
