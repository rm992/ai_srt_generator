import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class SubtitleNoiseFilter:
    def __init__(self, max_main_speakers: int = 2, min_duration_threshold: float = 1.5):
        """
        AI話者分離の出力からノイズ・ガヤ・重複を力づくで排除する例外管理クラス
        :param max_main_speakers: 残すメインキャストの最大人数（例: ツートップなら2）
        :param min_duration_threshold: 累積発話時間がこの秒数未満の話者は無条件でゴミとして排除
        """
        self.max_main_speakers = max_main_speakers
        self.min_duration_threshold = min_duration_threshold

    def filter_by_cumulative_duration(self, segments: list[dict]) -> list[dict]:
        """【第1防壁】累積発話時間ベースでガヤ話者ラベルを一括自動棄却する"""
        if not segments:
            return []

        # 1. 話者ごとの総発話時間を集計
        duration_map = defaultdict(float)
        for seg in segments:
            duration_map[seg['speaker']] += seg['end'] - seg['start']

        # 2. 発話時間が長い順にソート
        sorted_speakers = sorted(duration_map.items(), key=lambda x: x[1], reverse=True)
        
        # 3. メインキャスト（生存承認）とガヤ（棄却）の選別
        approved_speakers = set()
        for idx, (spk, total_dur) in enumerate(sorted_speakers):
            if idx < self.max_main_speakers and total_dur >= self.min_duration_threshold:
                approved_speakers.add(spk)
                logger.info(f"🟢 話者承認: {spk} (累積発話時間: {total_dur:.2f}秒) -> メインキャストとして保持")
            else:
                logger.warning(f"🔴 話者棄却: {spk} (累積発話時間: {total_dur:.2f}秒) -> ガヤ・ノイズとして抹殺")

        # 4. 承認された話者のセグメントのみを抽出
        filtered_segments = [seg for seg in segments if seg['speaker'] in approved_speakers]
        logger.info(f"累積時間フィルタリング完了: {len(segments)} -> {len(filtered_segments)} セグメント")
        return filtered_segments

    def merge_adjacent_segments(self, segments: list[dict], max_gap: float = 1.5) -> list[dict]:
        """【第2防壁】同一話者の近接発話（ギャップがmax_gap秒以内）を自動結合する"""
        if not segments:
            return []

        # タイムライン順（開始時間順）にソート
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        merged_segs = []

        for current in sorted_segs:
            if not merged_segs:
                merged_segs.append(current)
                continue

            last = merged_segs[-1]

            # 同一話者、かつ発話の隙間が指定秒数（1.5秒）以内か判定
            if current['speaker'] == last['speaker'] and (current['start'] - last['end']) <= max_gap:
                gap = current['start'] - last['end']
                # 前のセグメントの終了時間を後ろのセグメントの終了時間へ拡張して結合
                last['end'] = max(last['end'], current['end'])
                logger.debug(f"🔗 近接発話結合 [{current['speaker']}]: ギャップ {gap:.2f}秒 を埋めました")
            else:
                merged_segs.append(current)

        logger.info(f"インテリジェント・マージ完了: {len(segments)} -> {len(merged_segs)} セグメント")
        return merged_segs

    def remove_overlapping_segments(self, segments: list[dict], iou_threshold: float = 0.4) -> list[dict]:
        """【第3防壁】双方向IoU（時間重複比）が40%以上のセグメントは、尺の短い方を重複ノイズとして自動除外する"""
        if not segments:
            return []

        # 開始時間順にソート
        sorted_segs = sorted(segments, key=lambda x: x['start'])
        keep_flags = [True] * len(sorted_segs)

        for i in range(len(sorted_segs)):
            if not keep_flags[i]:
                continue
            for j in range(i + 1, len(sorted_segs)):
                if not keep_flags[j]:
                    continue

                seg1 = sorted_segs[i]
                seg2 = sorted_segs[j]

                # 重複区間（交差部分）の計算
                overlap_start = max(seg1['start'], seg2['start'])
                overlap_end = min(seg1['end'], seg2['end'])
                overlap_dur = overlap_end - overlap_start

                if overlap_dur > 0:
                    # 両者のうち「短い方の尺」に対する重複割合（双方向IoU/占有率）を計算
                    dur1 = seg1['end'] - seg1['start']
                    dur2 = seg2['end'] - seg2['start']
                    min_dur = min(dur1, dur2)
                    
                    iou = overlap_dur / min_dur

                    if iou >= iou_threshold:
                        # 尺の短い方のインデックスを特定してフラグを折る（抹殺）
                        if dur1 < dur2:
                            keep_flags[i] = False
                            logger.warning(f"❌ 重複排除: セグメントi({seg1['speaker']})の尺が短いため除外 (IoU: {iou:.2f})")
                            break # seg1が消えたので内側ループを抜ける
                        else:
                            keep_flags[j] = False
                            logger.warning(f"❌ 重複排除: セグメントj({seg2['speaker']})の尺が短いため除外 (IoU: {iou:.2f})")

        final_segs = [sorted_segs[idx] for idx, keep in enumerate(keep_flags) if keep]
        logger.info(f"重複排除（IoUフィルタ）完了: {len(segments)} -> {len(final_segs)} セグメント")
        return final_segs

    def execute(self, segments: list[dict]) -> list[dict]:
        """すべてのノイズフィルタリングを完全自律執行するメインエントリーポイント"""
        logger.info("=== 例外管理ノイズフィルター 執行開始 ===")
        
        # Step 1: 累積時間でゴミ話者を丸ごと消去
        x = self.filter_by_cumulative_duration(segments)
        
        # Step 2: 近接発話を滑らかに結合
        x = self.merge_adjacent_segments(x)
        
        # Step 3: タイムラインの重なりを排除して確定
        final_segments = self.remove_overlapping_segments(x)
        
        logger.info(f"=== 例外管理ノイズフィルター 正常完走 (最終: {len(final_segments)}区間) ===")
        return final_segments
