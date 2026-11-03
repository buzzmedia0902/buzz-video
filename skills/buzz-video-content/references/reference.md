# 参考動画に寄せる（ref_probe.py）

「この動画みたいに」「この雰囲気で」と参考動画（URLかファイル）をもらったときに、その動画のトンマナを**数字と画像で測り**、一番近いトンマナを選んで、差分だけを `look` に書く。測るだけで、参考動画の映像・音・文字は作品に使わない。

## 使い方
```bash
"$PY" "$S/ref_probe.py" "<URL かファイル>" --minutes 3 --out work/ref     # $S と $PY は SKILL.md の冒頭のもの
```
- `--minutes` 最初の何分を測るか（既定3）。`--lang` 話している言葉（既定 ja。英語は en）。
- URL は yt-dlp で720p以下・その区間だけ取る（測るためだけに使う）。取れないときは yt-dlp を更新するか、ユーザーにファイルをもらう。
- 3分の動画で40秒〜1分ほど（Whisper と文字認識を含む）。同じ `--out` に上書きされる。
- 出力：`ref.json`（数字・近いトンマナ・`look` の提案）、`contact.jpg`（5秒ごとの画面）、`telops.jpg`（字幕・テロップの拡大）、`frames/`。

## 手順
1. ref_probe を動かし、**`contact.jpg` と `telops.jpg` を Read で見る**。フォント（丸ゴ／明朝／筆文字）、図解の作り、色の世界観、アニメは数字に出ない。ここで判断する。
2. 画面に出た「いちばん近いトンマナ」を、画像と合わせて確かめる。`tone_scores` の1位と2位が0.1以内なら決めつけず、ユーザーに選んでもらう（AskUserQuestion）。
3. `ref.json` の `suggest.look` を edit.json の `look`（全体、または version 側）に入れる。フォントは画像を見て自分で選ぶ（下の表）。
4. BGM が `maybe`（あるかも）のときは、ユーザーに「BGM入っていますか？」と聞く。
5. `render.py --stills` で参考動画と並べて見比べ、ずれている所だけ `look` を直す。

## 数字の読み方
| `ref.json` | 意味 | 目安・使い道 |
|---|---|---|
| `shots.cuts_per_min` | 1分あたりの画面の切り替え（ジャンプカット・ズームの切り替え・スライドの切り替え） | 信頼 8〜26、親しみ 5〜16、バラエティ 18〜70、和 3〜12、ミニマル 0〜10 |
| `shots.median_shot_s` | ショットの長さの中央値（秒） | 短いほど勢いがある |
| `camera.zoom_switch_per_min` | カットで顔の大きさが変わる回数。寄り引き＋話し手が体を動かした分も入る | 0なら固定。10以上で `camera.hold` を短く |
| `camera.push_per_min` | 1ショットの中でゆっくり寄る回数 | 0なら `punch_every` を 999（寄らない） |
| `audio.pauses.p97_s` | 残っている間の長さの上限。ジェットカットの `cut.pause` に近い（±0.05秒） | 信頼 0.32、親しみ 0.45、バラエティ 0.25、和・ミニマル 0.5 |
| `audio.pauses.peak_s` | いちばん多い間の長さ。`cut.tail`＋`cut.lead` に近い | 信頼は 0.19 |
| `audio.pauses.jet_cut` | `false` なら間を詰めていない（自然な間のまま） | そのときは `cut.pause` を 0 にできる |
| `audio.speech_share` | 話している時間の割合 | 0.9以上はぎっしり詰めた編集 |
| `speech.chars_per_sec` | 1秒あたりの文字数（日本語）。Whisper で一番話している60秒を数える。詰めた編集は時計の1秒あたり、間のある動画は話している時間の1秒あたり | 5.5 ゆっくり／6.5〜7.5 ふつう／8.5以上 速い。8.6以上は `cut.speed` 1.1 を提案 |
| `audio.lufs` | 全体の音の大きさ | 書き出しは -14 |
| `audio.music.present` | BGM。`yes`：間の音量とステレオの広がりがBGMらしい／`maybe`：何か鳴っている（モノラルのBGMか部屋の音）→**ユーザーに聞く**／`no`：ほぼ無音 | `no` なら `audio.bgm: "none"` |
| `audio.music.under_voice_db` | 声に対するBGMの大きさ（声より何dB下か。±3dB） | `audio.music_db` にそのまま使える |
| `audio.sfx.level` | 間にある効果音（ヒュッ・ポン）の多さ。話の下に重なった音は数えない | `heavy` なら `audio.sfx_db` を -1 に |
| `telops.subtitle.y` | 字幕の縦の位置（0=上、1=下） | `style.sub.y` に |
| `telops.subtitle.line_height_pct` | 字幕1行の高さ（画面の高さに対する%）| 字の大きさ ≒ 高さ ÷ 1.12。`style.sub.size` に |
| `telops.subtitle.lines_per_frame` / `one_line_pct` | 字幕が1行か2行か | 2行までなら標準のまま |
| `telops.title_band` | ずっと出ている文字（番組名・タイトル帯）の位置 | あるなら `title_band` を使う |
| `telops.colors` | 文字の塗り色（見つけた文字から。名前・色コード・割合） | 1位が字幕の色、2位以降の鮮やかな色が強調色 |
| `telops.outline_pct` | 黒フチのある文字の割合（暗い映像の上は判定できず、空欄になる） | 高い→信頼・バラエティ・和、低い→親しみ・ミニマル |
| `telops.plate_pct` | 文字の下に明るい色板（吹き出し・帯）がある割合 | 高い→親しみ |
| `look.saturation` / `contrast` / `brightness` | 画面の彩度・コントラスト・明るさ | 彩度0.46以上なら `grade` で彩度を上げる提案 |

「いちばん近いトンマナ」は、上の数字を各トンマナの標準（tones.md）に当てはめた点数。カット数・間・BGM・効果音・字幕の黒フチ・色板・色の数・字幕の大きさ・明るさで決める。音がない参考動画（無音の書き出しなど）は、音の項目を除いて計算する。

## look に書く
`suggest.look` は標準のトンマナとの差が大きいものだけが入っている。ほとんど同じなら `{}` になる（それは「そのトンマナのまま」でよいという意味）。

| 測った値 | edit.json `look` のキー | 例 |
|---|---|---|
| 間の詰め方 | `cut`: `pause` / `tail` / `lead` / `speed` | `"cut": {"pause": 0.25, "tail": 0.08, "lead": 0.06}` |
| 寄りの動き | `camera`: `hold` / `med` / `punch_every` / `punch_zoom` | `"camera": {"punch_every": 999}` |
| BGM・効果音 | `audio`: `music_db` / `bgm`（`"none"` で無し）/ `sfx_db` | `"audio": {"music_db": -18}` |
| 字幕の位置・大きさ・色 | `style.sub`: `y` / `size`（`{"tall": …, "wide": …}` で比率別）/ `fill` / `stroke` / `emph` | `"style": {"sub": {"y": {"tall": 0.77}, "fill": "#FFFFFF"}}` |
| 強調の色 | `style.accent`（`#RRGGBB`） | `"style": {"accent": "#FEDB2E"}` |
| 色味 | `grade`（ffmpegの `eq=`） | `"grade": "eq=contrast=1.06:saturation=1.15"` |
| フォント | `fonts`: 要素名 → `gfx.py` の `FAMILIES` のキー | `"fonts": {"sub": "round", "title": "midashi_go"}` |

フォントは画像を見て選ぶ。

| 見た目 | キー |
|---|---|
| 標準の太ゴシック | `gothic` |
| やわらかい丸ゴシック | `round` / `round_b` / `maru` / `mplus_round` |
| テレビ番組の見出しゴシック | `midashi_go` |
| 明朝 | `mincho` / `midashi_min` |
| 筆（楷書・行書） | `kaiti` / `xingkai` |

サムネだけ別のフォントにするときは `"fonts": {"thumb": "round_b"}`（thumb.py が読む）。

## 例（社長のSNS向けトーク動画、89秒）
測った値：1分に14.1回カット（ショット中央値2.9秒）／間は0.31秒まで（ジェットカット）／1秒に7.0文字／-14.1 LUFS／BGMほぼなし／字幕は下（y=0.77）で文字高5.0%、白＋黄色の強調。
→ 近いトンマナ：信頼・解説（0.71、2位ミニマル0.63）。
```json
"look": {
  "audio": {"bgm": "none"},
  "camera": {"punch_every": 999},
  "style": {"sub": {"y": {"tall": 0.77}, "size": {"tall": 86}}}
}
```
開発時の確認として、エンジン自身の書き出し（信頼・ショート）を同じ方法で測ると、カット1分17回（実際は19回、検出は約9割）、BGM声の16dB下（実際は14dB）、字幕y=0.69（実際は0.70）、`look` の提案は `{}`（標準のままでよい）で、実際の設定とほぼ一致した。

## 測れないもの・間違えやすいもの
- フォント、図解の作り、アニメ、効果音の種類、BGMの曲調は測れない。`contact.jpg` と `telops.jpg` を見て決める。
- カット検出は、画面の差の急な山を数える。ゆっくりしたつなぎ（ディゾルブ）や、動きの小さいジャンプカットは数え漏れる（既知の短い動画で32か所中28か所、誤検出なし）。
- 寄り引きの回数には、話し手が前に出た・のけぞった分も入る。「画面が動く忙しさ」の目安で、設定値は大きくずれたときだけ提案する。
- BGM は間の音で測る。部屋の音とBGMはモノラルだと区別できず `maybe` になる。効果音は間にあるものだけ。
- 黒フチは暗い映像の上では見分けられない（空欄になり、点数からは外す）。
- 無音の動画、顔が映らない動画は、その項目が `null` になる。
- 英語など日本語以外は `--lang en` を付ける。文字数ではなく `speech.words_per_sec`（語/秒）を見る。
- 参考動画が他人の作品のときは、雰囲気を測るだけにする。映像・音・文字をそのまま取り込まない。
