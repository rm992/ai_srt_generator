# main.py
import sys
import os
import logging

# ログを画面に見やすく、プロっぽく出力する共通フォーマット
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from src.env_manager import PortableEnvManager
from src.core_pipeline.py import FullAutomaticSubtitlePipeline  # パスに応じて適宜修正

def main():
    print("==========================================================")
    print("  AI駆動型・全自動高精度SRT字幕ジェネレーター CLIコア v1.0.0 ")
    print("==========================================================")

    # 1. 環境パスの最優先ジャック（FFmpegのポータブル強制バインド）
    try:
        PortableEnvManager.initialize_ffmpeg()
    except RuntimeError as e:
        logger.error(f"環境バインドエラー:\n{e}")
        sys.exit(1)

    # 2. サンプル動画でのテスト駆動（引数があればそれを採用、なければ sample.mp4）
    video_input = sys.argv[1] if len(sys.argv) > 1 else "sample.mp4"
    output_srt = os.path.splitext(video_input)[0] + ".srt"

    if not os.path.exists(video_input):
        logger.error(f"対象の動画ファイルが見つかりません。ルートに配置するかパスを指定してください: {video_input}")
        print("\n💡 使い方: python main.py [動画ファイルのパス]")
        sys.exit(1)

    # 3. パイプラインの一斉執行
    try:
        # 初回起動時は whisper モデルの自動DLが走ります（models/ フォルダ内に隔離保存）
        pipeline = FullAutomaticSubtitlePipeline(whisper_model_size="large-v3")
        pipeline.run(video_input, output_srt)
        print("\n✨ 全自動字幕生成プロセスが正常に完走しました！ ✨\n")
    except Exception as e:
        logger.exception(f"致命的なパイプラインエラーが発生しました: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
