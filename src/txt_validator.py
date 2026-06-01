import re
import logging

logger = logging.getLogger(__name__)

class SubtitleTextValidator:
    def __init__(self, custom_blacklist: list[str] = None):
        """
        AIが生成したハルシネーションやパニックテキストを正規表現とロジックで抹殺する検閲クラス
        :param custom_blacklist: 追加で排除したい特定の固有名詞やフレーズのリスト
        """
        # 1. 定番のハルシネーション・ブラックリスト（部分一致で即座に弾く）
        self.blacklist = {
            "ご視聴ありがとうございました",
            "ご視聴ありがとうございましたー",
            "チャンネル登録",
            "高評価よろしくお願いします",
            "ベルマークの通知",
            "動画をご視聴",
            "字幕のご視聴",
            "視聴いただきありがとうございました",
            "お楽しみに",
            "シェアしてください",
            "Subtitles by",
            "subtitles by",
        }
        if custom_blacklist:
            self.blacklist.update(custom_blacklist)

        # 2. パニックテキスト・無限ループ検知用の強力な正規表現パターン群
        self.panic_patterns = [
            # 同一文字の4回以上の連続（例：「ああああ」「ええええ」）
            re.compile(r"(.)\1{3,}"),
            # 同一音節（2文字以上の文字列）の3回以上の連続ループ（例：「お笑いお笑いお笑い」「あ、あ、あ、」）
            re.compile(r"(.+?)\1{2,}"),
            # 記号（、。,.? !）のみ、または記号が異常に連続しているケース
            re.compile(r"^[、。，．,\.\?\! \s]+$"),
            # カンマやスペース区切りで同じ単語が並ぶプロンプトオウム返し（例：「お笑い, お笑い, お笑い」）
            re.compile(r"([^,，、\s]+)[,，、\s]+\1"),
        ]

    def is_hallucination(self, text: str) -> tuple[bool, str]:
        """
        単一のテキストが幻聴・不要テキストであるかを精密検査する
        :param text: 検査対象の文字列
        :return: (True/False, 棄却理由の理由ラベル)
        """
        # 前後の空白を除去
        cleaned_text = text.strip()

        # 防壁 0: 空っぽのテキストは無条件で弾く
        if not cleaned_text:
            return True, "EMPTY_TEXT"

        # 防壁 1: 短すぎる無意味文字の検閲（1文字かつ記号や平仮名の特定文字）
        if len(cleaned_text) <= 1 and cleaned_text in "っんあいうえおつ。、.":
            return True, "TOO_SHORT_MEANINGLESS"

        # 防壁 2: ブラックリスト（定番フレーズ）の部分一致判定
        for bad_word in self.blacklist:
            if bad_word in cleaned_text:
                return True, f"BLACKLIST_MATCH ({bad_word})"

        # 防壁 3: 正規表現によるパニック・ループ検知
        for pattern in self.panic_patterns:
            if pattern.search(cleaned_text):
                return True, f"REGEXP_PANIC_PATTERN ({pattern.pattern})"

        return False, "VALID"

    def filter_segments(self, segments: list[dict]) -> list[dict]:
        """
        文字起こしが完了したタイムライン（セグメントリスト）から、
        ハルシネーションを含むセグメントをタイムラインごと一括抹殺する
        :param segments: [{'start': 1.2, 'end': 3.4, 'text': 'こんにちは', ...}, ...]
        :return: 厳選クレンジングされたセグメントリスト
        """
        logger.info("=== 幻聴（ハルシネーション）自動検閲防壁 起動 ===")
        valid_segments = []

        for seg in segments:
            # セグメントからテキストを取得（キーがない場合は空文字）
            text = seg.get("text", "")
            
            is_bad, reason = self.is_hallucination(text)
            
            if is_bad:
                logger.warning(
                    f"💀 幻聴検閲によりセグメントを抹殺スキップ! "
                    f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] "
                    f"理由: {reason} | テキスト: '{text}'"
                )
            else:
                valid_segments.append(seg)

        logger.info(
            f"=== 幻聴自動検閲防壁 正常終了 (総枠: {len(segments)} -> 生存: {len(valid_segments)}) ==="
        )
        return valid_segments
