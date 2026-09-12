# YouTube同期メトロノーム

YouTubeの演奏動画に合わせてメトロノームを鳴らし、同じ小節を何回も練習するためのページです。
**https://mathman1111.github.io/youtube-practice-metronome/**

- URLを貼るだけ。一時停止・巻き戻し・速度変更をしてもずれない
- 小節単位のループ(`[` `]` `L`)、N回できたら自動で速度アップ
- 「音を録って自動で合わせる」で、どの曲でも拍を自動検出(タブの音声をブラウザ内で解析。Chrome / Edge)
- 手動でも合わせられる: 曲の頭で `D`、後半で `E`

## 解析済みの曲
- https://www.youtube.com/watch?v=224w0wYJqbg (165.99 BPM)
- https://www.youtube.com/watch?v=a5uCd7St2ys (102.73 BPM)
- https://www.youtube.com/watch?v=xzoShzMIlIM (175 BPM)
- https://www.youtube.com/watch?v=_hDWqju9SoE (175 BPM)

解析はローカル版(Python)で行っています。曲の追加希望は Issue へ。