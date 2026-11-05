---
name: buzz-video-content
description: 社長・個人・企業の「発信用」動画を編集するスキル。YouTube本編（横・トーク）、ショート／リール／TikTok（縦）、長い動画（番組・対談・配信など）からの切り抜き、日常Vlogに対応。最初にトンマナ（信頼・解説／親しみ・やわらか／バラエティ・勢い／和・重厚／洗練・ミニマル、または参考動画に寄せる）と用途・比率・長さを選択式で質問し、文字起こし・ジェットカット・テロップ・図解カード・顔を避けた配置・話者の色分け・オリジナルBGMと効果音・音量調整（-14 LUFS）・サムネ・概要欄／投稿文まで仕上げる。「YouTube用に編集して」「ショートにして」「この動画を切り抜いて」「テロップを入れて」「社長の発信用に」「Vlogにして」「トンマナを合わせて」など、広告以外のトーク・発信系の動画編集では必ずこのスキルを使う（店舗の広告動画は buzz-video-ad）。「buzz-video のセットアップをして」「動画編集の環境を準備して」と頼まれたときも、このスキルの手順でセットアップする。
---

# 発信動画（buzz-video-content）

素材 → 文字起こし → 構成表（edit.json）→ 静止画で確認 → 書き出し → BGM・効果音・音量 → サムネ・投稿文、までを一式で行う。
**最初にトンマナを決めると、テロップの見た目・カットの速さ・画面の動き・BGMと効果音がまとめて切り替わる**。毎回コードは書かず、構成表を書いて静止画で詰める。

```
S="${CLAUDE_SKILL_DIR}/scripts"   # このスキルの scripts フォルダ（展開されていなければ、この SKILL.md の隣の scripts）
PY=$(bash "$S/env.sh")          # 専用のPython環境（初回は自動作成）。--check で外部ツールも確認
```
- 初めて使うマシンでは、先に `bash "$S/setup.sh"` を実行する（ffmpeg・Whisper とそのモデル・フォント・HDR変換表をそろえる。足りないものは Homebrew で入れる。`--check` なら確認だけ）。
- Windows でも同じ手順で動く（Claude Code が Git Bash で実行する。Git for Windows が必要）。`"$PY"` は UTF-8 で動く起動用スクリプトなので、そのまま使う。顔検出は Apple Vision の代わりに MediaPipe（初回に自動で環境を作る）、書き出しは Mac より遅い。
- 「buzz-video のセットアップ」を頼まれたら、広告スキルの分もまとめて `bash "$S/../../../setup.sh"` を実行する（このファイルがない場合は `bash "$S/setup.sh"`）。結果の NG・-- の行を短く伝え、Font Book でのダウンロードが必要なフォントがあれば名前を伝える。

## 進め方

### 1. 素材を見る
- 素材とロゴの場所を確認する。作業フォルダを決め（例：`<案件名>_動画/`）、`"$PY" "$S/probe.py" <素材フォルダ> --out work/footage` を実行する。
  - probe.py はフォルダ内の動画・画像をすべて読む。関係ないファイルが多い場所なら、作業フォルダの `footage/` に使う素材だけをシンボリックリンクで入れてから渡す。
  - `speech` の割合でトーク（話が中心）かBロール（映像が中心）かのおおよそが分かる（音楽や環境音も高く出るので、最後は耳と映像で判断する）。
  - 長いクリップは `--per 40` などでコマを増やし、写ってはいけない人（子ども・一般の人）や場面（喫煙・他社の宣伝）を探す。
- `overview.jpg` とコンタクトシートを**実際に見て**、話している人・映ってはいけないもの（一般の人の顔、他社ロゴ、画面、個人情報）・HDR（→ `prep_hdr.py` をマシンで一度だけ）を書き出す。
- 長い動画からの切り抜きは、URLなら `find.py --download URL` で取得する（[references/formats.md](references/formats.md)「切り抜き」）。

### 2. 選択式で質問する（ここは省略しない）
ユーザーは選択式で答えるのを好む。AskUserQuestion で、**1回目は4問**（用途・トンマナ・比率・長さ）、**2回目は該当するものだけ**（参考動画・テロップの量・テンポ・サムネ／BGM）を聞く。
文面と「おすすめ」の付け方は [references/questions.md](references/questions.md)。トンマナは5つのうち素材に合う4つを選んで出す（残りは「その他」で選べる）。

### 3. 参考動画があれば測る
`"$PY" "$S/ref_probe.py" <URL かファイル>` で、カットの間隔・話す速さ・テロップの位置と色などを測り、近いトンマナと調整値を出す。コンタクトシートを見て、`edit.json` の `"look"` に反映する（[references/reference.md](references/reference.md)）。

### 4. 文字起こし → 直す → 時刻合わせ
```
"$PY" "$S/transcribe.py" edit.json --max-chars 16     # 縦は14〜16字、横は18〜20字で区切る
# cues.json を直す：固有名詞・聞き間違い・数字、強調は {AIの動画編集}、言い直しは "cut": true
"$PY" "$S/transcribe.py" edit.json --align           # 直した文に合わせて時刻を付け直す
"$PY" "$S/faces.py" edit.json                        # 顔の位置（寄りの構図・カードを顔から避ける・サムネ）
```
- 複数人が話す素材は `speakers.py` で声から話者を判定する。**名札や映っている人から推測しない**。判定できない行は話者なしのまま残す（[references/transcript.md](references/transcript.md)）。
- 文字起こしの対象は、話を聞かせる部分（talk と `"audio": "keep"` の clip）だけ。Vlog で声を残さないなら文字起こしは不要。字幕も声が聞こえる部分にしか出ない。
- 下書きには、名前（確認できない人名）や幻聴が混ざる。使わない行は消し、確認できない名前は字幕に出さない。

### 5. 構成表（edit.json）を書く
- 書式は [references/edit-spec.md](references/edit-spec.md)、用途ごとの組み立ては [references/formats.md](references/formats.md)、トンマナごとの違いは [references/tones.md](references/tones.md)。作例は [examples/talk_short_edit.json](examples/talk_short_edit.json)（同じ素材を2つのトンマナで作り比べた構成表）。
- 1つの edit.json に複数の `versions`（YouTube本編＋ショート3本など）を書ける。素材・字幕・話者は共通。
- 見出し（chapter）・図解カード（card）・強調ワード（emphasis）・名札（name）は、話の内容に合わせて素材の秒数で置く。

### 6. 静止画で確認しながら詰める
```
"$PY" "$S/render.py" edit.json --version <名前> --sheet 3      # 3秒ごとの一覧
"$PY" "$S/render.py" edit.json --version <名前> --stills 1.0 42.5
```
- 見るポイント：顔にテロップやカードがかかっていないか／字幕が2行以内で読めるか／タイトル帯・見出しが切れていないか／縦動画のカード表示（背景ぼかし＋顔の丸窓）が自然か。

### 7. 書き出し → 仕上げ
```
"$PY" "$S/render.py" edit.json --version <名前> --video        # 並列で書き出し（無音のマスター）
"$PY" "$S/finish.py" edit.json --version <名前>                # 声のカット・BGM・効果音・-14 LUFS・軽量版・字幕・概要欄
"$PY" "$S/thumb.py" edit.json --version <名前> --copy "…"      # YouTubeのサムネ3案（テストと比較用）
```
- 長い動画は書き出しに数分かかる。最初の1本は軽量版を送り（SendUserFile があればそれで。なければファイルの場所を伝える）、残りの見込み時間を添える。

### 8. 仕上げのチェック
- `report_*.json`（`qa.py`）で、音量 -14±1 LUFS、True Peak -1 dBTP以下、字幕の表示が1.2秒未満のものがないか、1分あたりの画面切り替え・寄りの回数を確認する。
- 別のエージェントに静止画と構成表を渡してレビューさせる（依頼文は [references/qa.md](references/qa.md)）。

### 9. 納品
- `納品/` に動画・軽量版・字幕（.srt）・サムネ／カバー・概要欄または投稿文を揃える（[references/delivery.md](references/delivery.md)）。投稿文・概要欄の説明文は型に沿って Claude が書く（finish.py が作るのはチャプター一覧まで）。アップロードは頼まれたときだけ。
- 報告は短く：何を作ったか、どこにあるか、公開前にユーザーが確認すべきこと（出演者の同意、BGM以外の音の権利、話者名）。

## 判断の目安
- **トンマナ**：ビジネスの解説・ノウハウは「信頼・解説」、人柄・日常・対談の後日談は「親しみ・やわらか」、エンタメ・企画ものは「バラエティ」、投資番組や対談の山場など勝負の場面は「和・重厚」、ブランド・高価格帯は「洗練・ミニマル」。
- **テンポは画面の動きではなく、カット・テロップ・音で作る**。画面の寄り引きは10秒以上あけ、ズームは40秒に1回程度（視聴者から「画面が動きすぎて見づらい」と言われた）。
- **字幕は1.2秒以上**表示する（「短くて読めない」と言われた）。言いよどみは字幕からだけ外し、声は残す。
- **BGMは声の約14dB下**（21dB下では「BGMがない」と感じられた）。効果音は見出し・カード・強調に合わせる。
- **YouTube本編の最後**は、山場で終えてから、アニメーションの高評価・チャンネル登録ボタンで締める。出演者クレジットは頼まれない限り入れない。

## つまずきやすい点
- **Whisperの時刻ずれ・幻聴**：まとめて文字起こしすると最大1秒ずれ、無音で「ご視聴ありがとうございました」が出る。transcribe.py は話している区切りごとに聞き取るので、そのまま使う。
- **話者の取り違え**：公開動画で他人の発言を本人のものにすると大問題になる。声で照合し、怪しい行は話者なしにする。
- **字幕の改行**：単語の途中や「ん」の前、英語名の途中で切らない（エンジンは形態素で判定済み）。
- **パネルが顔にかかる**：faces.py を実行していないと、カードの置き場所を顔から避けられない。
- **素材の音楽**：背景で流れている曲はContent IDに引っかかる可能性がある。報告に書く。
- **/tmp は再起動で消える**：作業ファイル・スクリプトは作業フォルダの中に置く。
- **24コマ・可変コマの素材**：30コマの動画に混ぜると、数コマごとに同じコマが重なってカクつく（report の `judder_frames`、finish が警告）。その part に `"smooth": true` を付けると、動きから間のコマを作る（書き出しは遅くなる）。iPhone の「120fps」表示でも、実際は24〜30コマのことが多い（probe.py は実際のコマ数を出す）。

## 参照ファイル
| ファイル | いつ読むか |
|---|---|
| [references/questions.md](references/questions.md) | 手順2の質問を出す前 |
| [references/tones.md](references/tones.md) | トンマナを選ぶ・説明するとき |
| [references/formats.md](references/formats.md) | YouTube本編／ショート／切り抜き／Vlogの組み立て |
| [references/edit-spec.md](references/edit-spec.md) | edit.json を書くとき |
| [references/transcript.md](references/transcript.md) | 文字起こしの直し方・話者判定・字幕のルール |
| [references/reference.md](references/reference.md) | 参考動画に寄せるとき |
| [references/audio.md](references/audio.md) | BGM・効果音・音量の決め方 |
| [references/thumbnail.md](references/thumbnail.md) | サムネ・カバー |
| [references/qa.md](references/qa.md) | チェックリスト、レビュー依頼文 |
| [references/delivery.md](references/delivery.md) | 納品物の名前、概要欄・投稿文の型、アップロード |
