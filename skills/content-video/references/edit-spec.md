# 構成表（edit.json）の書き方

1つの edit.json に、素材・字幕・話者を共通で持ち、`versions` に書き出す動画を並べる（本編＋ショート3本など）。
パスは edit.json からの相対パスか絶対パス。時刻はすべて秒。

```jsonc
{
 "project": "社長チャンネル_第2回",                 // 出力ファイル名の先頭
 "tone": "trust",                             // 既定のトンマナ: trust | friendly | variety | wa | minimal
 "brand": {"accent": "#FEDB2E", "logo": "brand/logo.png", "channel": "山田社長｜AIと経営のリアル", "name": "山田社長"},
 "sources": {"talk": "~/Movies/talk_0001.mov",           // id: パス（文字列）か {"path", "rot": 180}
             "ep": {"path": "footage/episode.mp4"}},
 "cues": "cues.json",                         // 字幕（transcribe.py が作り、直してから --align）
 "speakers": {"山田": {"color": "#FEDB2E", "refs": [["ep", 612.0, 616.5], ["ep", 640.2, 644.0]]}},
 "look": {},                                  // 参考動画などから全versionに効かせる調整（下記）
 "out_dir": "納品", "work_dir": "work",
 "versions": {
  "main": {
   "format": "youtube", "aspect": "16:9", "tone": "trust",
   "parts": [
    {"kind": "clip", "src": "talk", "from": 312.4, "to": 318.0},          // 冒頭ダイジェスト（同じ言葉を後でもう一度使ってよい）
    {"kind": "card", "card": "title", "dur": 2.2, "text": "AIで、仕事と人生をおもしろく。"},
    {"kind": "talk", "src": "talk", "from": 3.0, "to": 650.0, "remove": [[120.5, 131.0]]},
    {"kind": "card", "card": "end", "dur": 6.0}
   ],
   "overlays": [
    {"type": "name", "src": "talk", "at": 4.0, "text": "山田 太郎", "sub": "株式会社サンプル 代表"},
    {"type": "chapter", "src": "talk", "at": 45.0, "text": "AIで変わったこと"},
    {"type": "card", "src": "talk", "at": 92.0, "dur": 6, "title": "3つのポイント", "items": ["…", "…", "…"]},
    {"type": "emphasis", "src": "talk", "at": 130.2, "text": "ついに来た"}
   ]
  },
  "short1": {"format": "short", "title": "【緊急】{AIの動画編集}が｜ここまで来た",
             "parts": [{"kind": "talk", "src": "talk", "from": 40.0, "to": 98.0}]}
 }
}
```

## version の項目
| キー | 意味 |
|---|---|
| `format` | youtube / short / clip / vlog（必須） |
| `aspect` | 9:16 / 16:9 / 1:1 / 4:5（既定：youtube は16:9、それ以外は9:16） |
| `tone` | このversionだけ別のトンマナにする |
| `title` | ショート・切り抜きの上のタイトル帯（`{強調}`、`｜`で改行）。ファイル名にも使う |
| `credit` | 切り抜きの出典表記（例：`出典：〇〇チャンネル 第17回`） |
| `parts` | 並べる部品（下記）。上から順に再生される |
| `overlays` | 見出し・カード・強調ワードなど（下記） |
| `cut` | ジェットカットの上書き：`{"pause": 0.32, "tail": 0.11, "lead": 0.08, "speed": 1.1}` |
| `camera` | 画面の動きの上書き：`{"hold": 10, "med": 1.14, "punch_every": 40, "punch_zoom": 1.08}` |
| `bgm` | BGMの上書き：`{"style": "pop", "music_db": -12, "bpm": 110}` |
| `sfx` | false で効果音なし |
| `snap` | `"beat"` でVlogのクリップの長さを曲の拍に合わせる |
| `subs` | false で字幕を出さない（テロップなし。タイトル・名札・カードは出る） |
| `look` | このversionだけの見た目の調整（下記） |

## parts（部品）
| kind | 項目 | 説明 |
|---|---|---|
| `talk` | `src`, `from`, `to`, `remove`, `pause`, `speed`, `fit`, `center`, `zoom`, `band_y`, `static` | 話している素材。無音の間を詰める（ジェットカット）。`remove` は削る範囲（言い直し・NG） |
| `clip` | `src`, `from`, `to` または `dur`, `speed`, `fit`, `audio` | 素材をそのまま使う（Bロール・Vlog・冒頭ダイジェスト）。`audio`: keep / duck（小さく残す・既定）/ mute |
| `card` | `card`, `dur`, `text`, `bg` | 全面のカード。`card`: title（ロゴ・チャンネル名）/ end（終了画面）/ eyecatch（区切り）/ text（全面テキスト）。`bg`: {"src", "at"} で背景の静止画、false で無地 |

画面の切り取り（`fit`）：
- `cover`：画面いっぱいに切り取る（`center: [x, y]` は素材上の位置 0〜1、`zoom` は1以上）
- `band`：横長の素材を縦画面に帯で置く（切り抜きの既定）。`band_y` で帯の中心の高さ（画面の割合）
- 比率ごとの調整：`"aspect": {"9:16": {"fit": "cover", "center": [0.4, 0.4]}}`
- 何も書かなければ、顔の位置（faces.py）を中心にし、トンマナの CAMERA に従って「寄り／引き」を10秒以上あけて切り替える。`"static": true` で切り替えない。

## overlays（重ねる要素）
時刻は `"src"` + `"at"`（素材の秒。カットされても自動で出力の秒に変換）か `"out"`（出力の秒）。長さは `"dur"`（出力の秒）。

| type | 項目 | 既定の長さ |
|---|---|---|
| `chapter` | `text` 見出し（概要欄のチャプターにもなる） | 2.4秒 |
| `card` | `title`, `items`（2〜4個）, `big`（大きな数字など）| 4秒（`dur` で指定推奨） |
| `emphasis` | `text` 強調ワード | 1.4秒 |
| `name` | `text`, `sub` 名札 | 3.2秒 |
| `caption` | `text`, `sub` 場所・日付など（Vlog） | 2.8秒 |
| `note` | `text` ※補足 | 3秒 |
| `title_band` | `text`（既定は version の `title`） | 本編中ずっと |
| `credit` | `text`（既定は version の `credit`） | 本編中ずっと |

カード・名札・強調ワードは、顔に重ならない位置に自動で置かれる（faces.py の結果を使う）。縦画面のカードは「背景ぼかし＋顔の丸窓」で表示する。

## cues.json（字幕）
```json
[{"src": "talk", "t0": 12.3, "t1": 14.8, "text": "AIで{動画編集}が変わる", "speaker": null}]
```
- `{…}` は強調（トンマナの強調色・マーカー）、`｜` は改行位置の指定。
- `"cut": true` はその行を字幕に出さない（声は残る。声も消すなら part の `remove` に入れる）。
- 字幕の表示時間はエンジンが1.2秒以上に整える（前後の無音に伸ばす／同じ話者の隣の行とつなぐ）。

`"fonts": {"thumb": "midashi_go"}` でサムネの文字だけフォントを変えられる。

## speakers（話者）
```json
"speakers": {"山田": {"color": "#FEDB2E", "refs": [["ep", 234.6, 249.9], ["ep", 283.6, 292.2]]}}
```
- `refs`：本人だけがはっきり話している区間（素材のid・開始・終了）を2〜4か所、各5〜15秒。名札ではなく、顔のアップと文字起こしから選ぶ。
- その範囲で話す人は全員登録する（2位との差で誤判定を防ぐため）。似た声（双子など）は別の話者にする。
- speakers.py が付けた行には `"voice": [類似度, 差]` が入る。手で付けた話者は `"locked": true` で固定でき、声と食い違うと表で指摘される。

## look（見た目の調整）
トンマナの標準からの差分だけを書く。プロジェクト全体（トップの `look`）と version ごとに書ける。
```json
"look": {
 "cut": {"pause": 0.28},                       // ジェットカット
 "camera": {"punch_every": 60},                // 寄りの頻度
 "audio": {"music_db": -12, "bgm": "pop"},     // BGM
 "subs": {"min_show": 1.3},
 "grade": "eq=contrast=1.06:saturation=1.1",   // 色（ffmpegのフィルター）
 "fonts": {"sub": "round", "title": "midashi_go"},   // 要素ごとのフォント（キーは scripts/cv/gfx.py の FAMILIES）
 "style": {"sub": {"emph": "#FF4D8D", "size": {"tall": 80}}, "accent": [254, 219, 46]}
}
```
