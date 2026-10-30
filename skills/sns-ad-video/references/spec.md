# project.json（構成表）の書き方

1案件につき1ファイル。パスは project.json からの相対パスか絶対パス（`~` も使える）。
完全な例：[../examples/salon/project.json](../examples/salon/project.json)

```json
{
 "project": "SALON_SAMPLE",                       // 納品ファイル名の先頭
 "footage_dir": "~/Movies/salon_footage",    // 素材フォルダ
 "out_dir": "納品", "work_dir": "work",        // 既定値
 "platform": "meta",                          // 9:16 のセーフゾーン: meta | tiktok
 "cta": "ご予約はリンクから",                   // エンドカードの導線（version 側でも上書きできる）
 "brand": {
  "name": "SALON SAMPLE",
  "logo": "brand/logo.png",                    // logo_prep.py を先に実行しておく
  "label": "BEAUTY ACUPUNCTURE",               // ロゴに添える小さな英字（空でもよい）
  "tagline": "がんばる自分に、ごほうびを。",       // エンドカードのタグライン（空でもよい）
  "colors": {"accent": "#D2BF5B", "ink": "#221E1B", "paper": "#F7F3EA", "night": "#1E1915"},
  "fonts": {"title": "midashi", "body": "maru"} // 役割 → フォントキー（省略時は既定）
 },
 "clips": { "IMG_2739.MOV": {"rot": 180, "patches": [[0.83, 0.45, 0.21, 0.19]]} },
 "versions": { "standard": {"taste": "standard", ...}, "vlog": {"taste": "vlog", ...} }
}
```

## brand
- colors：使う色は4つ。`accent`（ロゴの差し色）、`ink`（文字）、`paper`（明るい背景）、`night`（暗い背景）。
  `accent_deep` と `accent_light` は自動で作られる。logo_prep.py が出す `accent` の値をそのまま使える。
  `accent` が null（ロゴが黒・白だけ）のときは、看板・公式サイト・内装から1色決める。`accent_muted: true` はグレージュなど落ち着いた色という意味で、そのまま使ってよい。
- fonts の役割と既定値：

  | 役割 | 既定のフォント | 主な用途 |
  |---|---|---|
  | title | midashi | 見出し明朝 |
  | body | maru | 丸ゴシック。字幕・ボタン |
  | latin | optima_b | 英字ラベル・時刻 |
  | hand | klee | 手書き風 |
  | vertical | hmin6 | 縦書き |
  | ui | goth4 | 画面UI |
  | bold | goth8 | |
  | serif | mincho | タグライン |

  使えるキーは `core/text.py` の `_FONT_SPECS` にある。主なもの：
  - 明朝：mincho, midashi, hmin3, hmin6, yumin
  - ゴシック：goth3, goth4, goth6, goth8, tgoth
  - 丸ゴシック：maru, tmaru
  - 手書き風：klee, hand
  - 英字：optima, didot, avenir, futura, helv

## clips（素材ごとの共通設定。その素材を使うすべてのショットに効く）
| キー | 意味 |
|---|---|
| `rot` | 追加の回転（0/90/180/270、時計回り）。上下逆さの撮影は 180 |
| `patches` | ぼかす楕円のリスト。他社ロゴ、POP、PC画面、名札など |
| `focus` | 人物を残して背景をぼかす楕円（ポートレート風）。背景のPOP対策にも |
| `soft_below` | この高さ（0〜1）より下を段階的にぼかす。服のプリントなど下端の映り込み向け |
| `hdr` | HLG素材（probe.py が自動で検出するので、通常は書かなくてよい） |
| `flash_guard` | レーザー・光脱毛・フラッシュなどの1コマだけの閃光を、前のコマで置き換える（静止画の確認でも効く）。`probe.py --flash` で見つける |
| `note` | 自由メモ（描画には使わない） |

### 座標の決まり
素材の座標はすべて**正規化座標**で書く。基準は、回転を適用したあと正しい向きになった素材のフレーム。
- `x`：左0 → 右1
- `y`：上0 → 下1
- 楕円の半径 `rx`、`ry`：どちらも**素材の横幅**を1とした単位（`rx == ry` なら真円）

`grid.py <素材> <秒> --rot 180` を使うと、0.05刻みの目盛り入り画像で座標を読み取れる。

### patches の書き方
- 固定：`[cx, cy, rx, ry]` または `[cx, cy, rx, ry, 強さ]`（強さの既定は16。横幅1080pxの素材を等倍で見たときのぼかし半径）
- 追跡（動く対象）：`{"track": [[素材の秒, cx, cy], ...], "r": [rx, ry], "strength": 16}`。数点の位置を線形補間する。

## ショット（全テイスト共通の書き方）
```json
{"clip": "IMG_2722.MOV", "in": 0.85, "dur": 2.0, "speed": 0.5,
 "zoom": [1.15, 1.22], "center": [[0.5, 0.42], [0.5, 0.40]],
 "grade": "beauty", "fit": "auto", "shake": 0, "exposure": 1.0, "tint": 0.0,
 "patches": [[0.3, 0.6, 0.05, 0.05]], "focus": [0.4, 0.5, 0.15, 0.17],
 "aspect": {"1:1": {"center": [0.5, 0.35]}, "16:9": {"fit": "cover", "zoom": 1.6}}}
```
| キー | 意味 |
|---|---|
| `in` | 素材の開始秒 |
| `dur` | 出力上の長さ（秒）。Vlog・トークではテンプレートが自動で決めるので不要 |
| `speed` | 再生速度（素材の秒 ÷ 出力の秒）。`0.5` は半分の速さのスロー。60fps素材だけで使う |
| `zoom` | `[開始, 終了]`。1.0 で画面いっぱい（cover の場合）。1.0 より小さい値は不可 |
| `center` | 画面の中心にする素材上の点。`[x, y]` か、開始と終了の `[[x0,y0],[x1,y1]]` |
| `cam` | 細かくカメラを動かすとき：`[[ショット内の秒, zoom, cx, cy], ...]`（イーズで補間） |
| `fit` | `auto`（既定）／`cover`／`blur`。auto は、cover にすると素材の見える面積が45%未満になる場合（縦素材を16:9にするなど）に blur を選ぶ。blur は素材全体を中央に置き、左右か上下をぼかした背景で埋める |
| `grade` | 色調整：natural / beauty / soft / bright / pop / cine / none |
| `exposure` / `tint` | 明るさの倍率／色温度（マイナスで青み） |
| `shake` | 手持ち風の揺れ（出力px）。三脚の素材に 3〜7 を入れると生っぽくなる |
| `patches` | clips の patches に追加される |
| `focus` | clips の focus を上書きする |
| `aspect` | 比率ごとの上書き。キーは `"9:16" "4:5" "1:1" "16:9"`。中身はショットと同じキー |

**設定が優先される順**：テンプレートの既定 ＜ clips ＜ ショット ＜ `aspect` の上書き。
- `patches` だけは上書きでなく**足し算**になる（clips ＋ ショット ＋ aspect）。
- `focus` と `soft_below` は上書きになる。
- clips に書いた `soft_below` や `focus` は、その素材を使う**すべてのショット**に効く。別の部分を映すショットで不要なら、そのショットに `"soft_below": 0` や `"focus": null` を書いて打ち消す（そうしないと画面の半分がぼけることがある）。

**自動チェック（render.py --info でも実行される）**
- 素材の長さが足りないと、どこが何秒足りないかをエラーで知らせる。
- 30fps素材を `speed` < 1 で使うと、カクつくので警告を出す。

## versions
キーはバージョン名（ファイル名やフォルダ名になる）。`taste` には standard / motion / cinematic / vlog / chat のどれかを入れる。
共通のキー：
| キー | 意味 |
|---|---|
| `cta` | 導線の文言 |
| `label` | 納品ファイル名の「テイスト」部分。既定はテイストの日本語名 |
| `bgm` | `{"style": "piano｜pop｜cinematic｜lofi", "bpm": 100, "seed": 7, "gain_db": 0}` |

テイストごとの中身は [tastes.md](tastes.md) を見る。

## 出力の場所
| ファイル | 場所 |
|---|---|
| 静止画 | `work/review/<version>_<9x16>/tXX.XX.png` |
| 一覧 | `work/review/<version>_<9x16>_sheet.jpg` |
| 無音マスター | `work/render/<version>_<9x16>.mp4` |
| タイミング | `work/<version>/cues.json`（BGM・効果音・ナレーションの位置） |
| BGM | `work/<version>/bgm.wav`（バージョン単位で比率共通） |
| ナレーション | `work/<version>/voice/` |
| 納品物 | `納品/<Project>_<テイスト>_<9x16>_<秒>秒.mp4`、`_軽量版.mp4`、`_カバー.png` |
