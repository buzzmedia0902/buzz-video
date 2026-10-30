# 音（BGM・効果音・ナレーション・ミックス）

## BGM：毎回オリジナルを作曲する（bgm.py）
`finish.py` が `cues.json` をもとに自動で呼ぶ。単独でも試せる。
```
"$PY" "$S/bgm.py" --style piano --dur 24 --card 20.3 --out work/test_bgm.wav
"$PY" "$S/bgm.py" --cues work/<version>/cues.json --style lofi --out work/<version>/bgm.wav
```
| style | 雰囲気 | 既定のテイスト |
|---|---|---|
| `piano` | フェルトピアノ＋パッド＋ベル。清潔感・やさしさ | スタンダード |
| `pop` | プラック＋軽いハウスのリズム。明るく速い | モーション |
| `cinematic` | ストリングス＋ピアノ＋低音の持続音。壮大・エモい | シネマティック |
| `lofi` | エレピ＋ゆるいドラム＋レコードノイズ。日常・会話 | Vlog・トーク |

- エンドカードが出る瞬間（`card_at`）に、曲が解決してチャイムが鳴るように作る。テンポは小節線が `card_at` に乗るよう自動で微調整される。
- 雰囲気を変えたいときは `"bgm": {"style": ..., "bpm": ..., "seed": 3}` を変える。seed を変えるとメロディが変わる。
- 作り直すときは `finish.py ... --rebuild-bgm`。style・長さ・card が変わったときは自動で作り直す。
- 市販の曲・フリー音源・店内BGMは使わない（権利確認の手間と審査のリスクを避けるため）。

## 効果音（sfx.py）
テンプレートが `cues.json` の `sfx` にタイミングを書き、finish.py が合成する。
名前：`send` `receive` `sticker` `link` `pop` `swish` `whoosh` `tick` `chime` `impact` `type`

## ナレーション（tts.py。Vlog のとき）
| engine | 声 | 料金・表記 |
|---|---|---|
| `aivis`（既定） | AivisSpeech。`497929760` = morioki（落ち着いた自然な大人の女性。入っていなければ `--list-speakers` で選び直すか、AivisHub などから追加する） | 無料。モデルごとの規約（商用利用・クレジット・禁止用途）を確認する。morioki は ACML 1.0 |
| `voicevox` | VOICEVOX のキャラクター（`--list-speakers` で一覧） | 無料。動画か投稿文に「VOICEVOX:キャラ名」の表記が必要 |
| `higgsfield` | ElevenLabs v4（Higgsfield CLI）。一番自然 | クレジットを消費。`higgsfield auth login` が必要 |
| `files` | クライアントの録音（`voice.dir` に line_01.m4a …） | なし |

- ローカルのエンジンは止まっていれば自動で起動し、使い終わったら止める（もともと動いていたものは止めない）。
- `"tempo": 1.15` で、声の高さを変えずに15%速くする（rubberband）。テンポよく聞かせたいときに使う。
- 行と行の間は `gaps` で0.25〜0.4秒にする。0.5秒を超えると間延びして聞こえる。

### 読み間違いの直し方
- 難しい漢字・専門用語は `speech` をかなで書く（例：美容鍼 → びようばり、施術 → せじゅつ）。
- 店名は、カタカナで書くと平板に読まれやすい。サロンでの実際の発音と違うときは、ユーザーに確認して直す。
- Whisper が店名を漢字で書いた（例：ミホン → 見本）だけなら、読みは合っている。`voice.aliases` に `{"見本": "ミホン"}` を足して判定を通す。
- ElevenLabs は英語由来の言葉をネイティブ英語の発音で読む。ひらがなで書く（でしゃっぷ、まにゅある）。数字もかなで書く。
- 短い前置き（「手順二。」など）は ElevenLabs が飛ばしやすい。言い回しを変える。

## ミックスと音量（finish.py）
- 素材の音は使わない。音はBGM＋効果音＋ナレーションだけで作る。
- BGMの音量（`bgm.gain_db`）：トークは -5 dB、Vlogは -3 dB、ほかは 0 dB。
- ナレーション中はBGMを約15 dB下げる。20msごとのRMSで声を検出し、60ms早めに下げ始め、声が途切れても400msは下げたままにする（短い間でBGMが上下してうねるのを防ぐ）。
- 全体を -14 LUFS（SNSの推奨）に合わせ、リミッターをかける。AACに変換するとピークが上がるので、変換後の True Peak を測り、-1 dBTP を超えたらリミッターを下げて作り直す（自動）。
- 確認：`report_<比率>.json` の `lufs` が -15〜-13、`true_peak` が -1.0 以下であること。
