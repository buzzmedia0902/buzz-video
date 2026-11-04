# buzz-video

[English](README.en.md)

**Claude Code に動画編集を任せるためのスキル集**です。スマホやカメラで撮った素材から、SNS で発信・集客するための動画を Claude が編集し、書き出しまで仕上げます。

| スキル | 作れるもの |
|---|---|
| **buzz-video-content**（発信動画） | YouTube本編（横）、ショート・リール・TikTok（縦）、長い動画からの切り抜き、日常Vlog |
| **buzz-video-ad**（広告動画） | 店舗・サロン・クリニック・飲食店などの SNS 広告・リール（9:16 / 4:5 / 1:1 / 16:9） |

## 特徴
- **最初に選択式で質問**します。用途・トンマナ・比率・長さを選ぶだけで、編集の方針が決まります。参考動画を渡すと、カットの間隔・話す速さ・テロップの位置を測って寄せます。
- **発信動画のトンマナは5種類**：信頼・解説／親しみ・やわらか／バラエティ・勢い／和・重厚／洗練・ミニマル。切り替えると、テロップの見た目・間の詰め方・画面の動き・BGM・効果音がまとめて変わります。
- **広告動画のテイストは5種類**：スタンダード／モーション／シネマティック／Vlogナレーション／LINE風トーク（チャット画面風。LINE のロゴや名前は使いません）。
- 文字起こし（Whisper）→ 間を詰めるカット（ジェットカット）→ テロップ・図解カード・見出し → 終了画面まで、Claude が構成表を書いてエンジンが描画します。
- テロップやカードは顔を避けて置きます（Apple Vision）。複数人の話者は声で判定して色分けし、判定できない行には名前を付けません。
- **BGM と効果音はその場で作曲・合成**します（既存の曲やサンプル音源は使いません）。
- 音量 -14 LUFS・True Peak -1 dBTP 以下、スマホ確認用の軽量版、字幕（.srt）、サムネ3案、概要欄・投稿文まで作ります。
- 広告では、業種ごとの規制（医療・鍼灸・化粧品など）を確認しながら文言を決めます。

## 動作環境
- **macOS**（Apple Silicon 推奨）：顔の検出に Apple Vision、動画の読み込みに VideoToolbox を使います。[Homebrew](https://brew.sh) が必要です（ffmpeg・whisper-cpp・uv などを入れます）。
- **Windows 10 / 11（64ビット）**：[Git for Windows](https://gitforwindows.org/) が必要です（Claude Code が Git Bash でスキルの手順を実行します）。顔の検出は MediaPipe、ツールはセットアップが `~/.local/opt` に入れます。Mac より書き出しに時間がかかります。
- [Claude Code](https://claude.com/claude-code)

## インストール

### 方法1：Claude Code のプラグインとして（おすすめ）
Claude Code で次の2行を実行します。
```
/plugin marketplace add buzzmedia0902/buzz-video
/plugin install buzz-video@buzz-video
```

### 方法2：スキルのフォルダに置く
```bash
git clone https://github.com/buzzmedia0902/buzz-video.git ~/buzz-video
mkdir -p ~/.claude/skills
ln -sfn ~/buzz-video/skills/buzz-video-content ~/.claude/skills/buzz-video-content
ln -sfn ~/buzz-video/skills/buzz-video-ad ~/.claude/skills/buzz-video-ad
```
Windows では、PowerShell で次のように実行します（管理者権限は不要です）。
```powershell
git clone https://github.com/buzzmedia0902/buzz-video.git $HOME\buzz-video
New-Item -ItemType Directory -Force $HOME\.claude\skills | Out-Null
New-Item -ItemType Junction -Path $HOME\.claude\skills\buzz-video-content -Target $HOME\buzz-video\skills\buzz-video-content
New-Item -ItemType Junction -Path $HOME\.claude\skills\buzz-video-ad -Target $HOME\buzz-video\skills\buzz-video-ad
```

### 初回のセットアップ
Claude に「buzz-video のセットアップをして」と頼むと、足りないものを入れます（方法2なら `bash ~/buzz-video/setup.sh` でも同じです）。何度実行しても、入っていないものだけを入れます。
- ツール：ffmpeg・whisper-cpp・uv（必須）、rubberband・yt-dlp（任意）。Mac は Homebrew で、Windows は配布元から `~/.local/opt` に入れます
- Whisper の音声認識モデル（約550MB）
- 無料フォント：M PLUS Rounded 1c
- Python の環境（専用の仮想環境を `~/.local/share/` に作ります。Windows は顔検出用の環境も作ります）

macOS に付いているフォントのうち、筑紫丸ゴシック・凸版文久見出し・楷体・行楷などは、Font Book（アプリ）で検索して「ダウンロード」してください。なくても動きますが、トンマナの見た目が少し簡素になります。Windows では游ゴシック・游明朝・メイリオで代用します。`bash setup.sh --check` を実行すると、足りないものを確認だけできます（何も入れません）。

## 使い方
素材の場所を伝えて頼むだけです。
- 「~/Movies/talk.mov を YouTube 用に編集して」
- 「この動画からショートを3本作って」
- 「この対談動画から切り抜きを作って」
- 「この素材とロゴでリール広告を作って」

Claude は素材を見てから選択式で質問し、静止画で確認しながら構成を詰めます。最初の1本ができたら、スマホで見られる軽量版を先に共有します。

プラグインとして入れた場合は、`/buzz-video:buzz-video-content`・`/buzz-video:buzz-video-ad` で直接呼び出すこともできます。

## ナレーション（任意。広告の Vlog テイスト）
日本語の読み上げには、手元で動く音声エンジンを使います。
- [AivisSpeech Engine](https://github.com/Aivis-Project/AivisSpeech-Engine) か [VOICEVOX Engine](https://github.com/VOICEVOX/voicevox_engine) の macOS 版を、`~/.local/opt/aivisspeech_engine-<版>/` または `~/.local/opt/voicevox_engine-<版>/` に置きます（中に `run` があるフォルダ）。
- 既定の声は AivisSpeech の「morioki」です。入っていない場合は、入っている声の一覧から選び直します（声のモデルは AivisSpeech の「音声合成モデルの管理」や AivisHub から追加できます）。
- 声のモデルごとに、利用規約（商用利用・クレジット表記・禁止されている使い方）を確認してください。VOICEVOX は「VOICEVOX:キャラクター名」の表記が必要です。

## 気をつけること
- 切り抜きや URL からの取得は、権利者の許可がある動画（自分の出演回、許諾を得た番組など）だけにしてください。元の番組・チャンネルのガイドラインにも従ってください。
- 公開前に、出演者の同意と、素材に入っている音楽・ロゴの権利を確認してください。
- 広告の規制の説明は一般的な注意点です。法的な助言ではありません。
- 書き出し・文字起こし・作曲は手元の Mac で行います。ただし、Claude が確認のために見る静止画や文字起こしの文章は、会話の一部として Anthropic に送られます。ほかにネットに接続するのは、ツールやモデルの取得、URL からの動画の取得、クラウドの音声を選んだときです。
- LINE は第三者の商標です。このプロジェクトは LINE とは関係ありません。

## 仕組み（開発する人へ）
Claude が構成表（`edit.json` / `project.json`）を書き、Python のエンジン（skia・Pillow・numpy）と ffmpeg が1コマずつ描画します。BGM は numpy で作曲します。各スキルの `SKILL.md` が手順、`references/` が詳しい決まり、`scripts/` がエンジンです。buzz-video-content の内部構成は [scripts/DEV.md](skills/buzz-video-content/scripts/DEV.md) にあります。

## ライセンス
MIT License（[LICENSE](LICENSE)）© 2026 BUZZ MEDIA
