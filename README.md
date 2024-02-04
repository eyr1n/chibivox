# chibivox

私が [VOICEVOX CORE](https://github.com/VOICEVOX/voicevox_core) を理解するためのリポジトリです．

上から順に読んでいけばわかるようにするために，必要最低限のコードのみに絞りました(AquesTalk 風記法のパーサなどはオミットしました)．

また，ビルドの簡略化のために [VOICEVOX CORE 0.14.5](https://github.com/VOICEVOX/voicevox_core/releases/tag/0.14.5) に幾つかの変更を加えています．

- onnxruntime -> ort
- open_jtalk -> jpreprocess(辞書もクレート付属のものを使用)

本家リポジトリに用意されているサンプルモデルを `model/` 以下に配置することで手っ取り早く音声合成できます．

```sh
cargo run -- こんにちは
```
