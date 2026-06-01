import os
import sys
import shutil
import logging
import subprocess

logger = logging.getLogger(__name__)

class PortableEnvManager:
    @staticmethod
    def initialize_ffmpeg() -> str:
        """
        アプリケーション内のポータブルFFmpegを検出し、
        プロセスの環境変数（PATH）の最優先位置に強制バインドする。
        :return: 有効化されたFFmpegの絶対パス
        """
        # src/ フォルダの1階層上（プロジェクトルート）を基準にする
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        bin_dir = os.path.join(base_dir, "bin")
        
        # OSごとの実行ファイル名（Windowsは.exe、Mac/Linuxは拡張子なし）
        ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        bundled_ffmpeg_path = os.path.join(bin_dir, ffmpeg_name)
        
        # 1. 同梱バイナリの存在チェック
        if os.path.exists(bundled_ffmpeg_path):
            logger.info(f"同梱のポータブルFFmpegを検出しました: {bundled_ffmpeg_path}")
            
            # 現在のプロセス環境変数 PATH の「先頭（最優先）」に bin ディレクトリを挿入
            current_path = os.environ.get("PATH", "")
            if bin_dir not in current_path:
                os.environ["PATH"] = bin_dir + os.path.pathsep + current_path
                logger.info("環境変数 PATH の最優先位置にポータブル bin ディレクトリをマウントしました。")
                
            final_path = bundled_ffmpeg_path
        else:
            # 2. 同梱がない場合のセーフティガード（開発環境やフォールバック用）
            system_ffmpeg = shutil.which("ffmpeg")
            if system_ffmpeg:
                logger.warning(
                    f"bin フォルダに同梱バイナリが見つかりません。 "
                    f"システム環境上のFFmpegで代替します: {system_ffmpeg}"
                )
                final_path = system_ffmpeg
            else:
                # ユーザーがバイナリを入れ忘れており、PCにもない場合は即座に安全停止
                raise RuntimeError(
                    "致命的なエラー: FFmpeg が見つかりません。\n"
                    f"アプリケーションの「bin」フォルダ配下に {ffmpeg_name} を配置するか、"
                    "システムにFFmpegをインストールして環境パスを通してください。"
                )
                
        # 3. 最終的な動作・バージョン検証
        try:
            # 引数なし、または -version で正常に終了するかチェック
            result = subprocess.run(
                [ffmpeg_name, "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True, 
                check=True
            )
            version_line = result.stdout.splitlines()[0]
            logger.info(f"FFmpeg バインド成功確認: {version_line}")
        except Exception as e:
            raise RuntimeError(f"マウントされたFFmpegの起動テストに失敗しました: {e}")
            
        return final_path
