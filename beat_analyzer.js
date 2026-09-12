// 拍検出(server.py の analyze と同じ手順の JavaScript 版)。ブラウザと node の両方で動く。
//   analyzeBeats(samples: Float32Array, sr: number, opts?) → { tempo, beats[], strength[], mode, duration, phaseBiasMs, candidates[] }
// 手順:
//   1. 対数メルスペクトルの正の差分(スペクトラルフラックス)で「音の立ち上がり」の包絡を作る
//   2. 包絡の自己相関でテンポ候補を出し、候補ごとに BPM×位相の等間隔グリッドを総当たりで詰める
//   3. 20秒窓で位相のずれを測り、一定ならグリッド(ずれは補間)、揺れる曲は窓ごとの位相をつないで追う
//   4. 最後に時間領域のエネルギー増加(2.9ms刻み)で位相を詰め直す(立ち上がり基準)
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.BeatAnalyzer = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------- FFT(実数入力、radix-2) ----------
  function makeFFT(n) {
    const levels = Math.log2(n) | 0;
    if (1 << levels !== n) throw new Error('FFT size must be power of 2');
    const cosT = new Float64Array(n / 2), sinT = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / n); sinT[i] = Math.sin(2 * Math.PI * i / n); }
    const rev = new Uint32Array(n);
    for (let i = 0; i < n; i++) { let x = i, y = 0; for (let j = 0; j < levels; j++) { y = (y << 1) | (x & 1); x >>= 1; } rev[i] = y; }
    return function (re, im) {
      for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
      for (let size = 2; size <= n; size <<= 1) {
        const half = size >> 1, step = n / size;
        for (let i = 0; i < n; i += size) {
          for (let j = i, k = 0; j < i + half; j++, k += step) {
            const tr = re[j + half] * cosT[k] + im[j + half] * sinT[k];
            const ti = -re[j + half] * sinT[k] + im[j + half] * cosT[k];
            re[j + half] = re[j] - tr; im[j + half] = im[j] - ti; re[j] += tr; im[j] += ti;
          }
        }
      }
    };
  }

  // ---------- メルフィルタ(Slaney 方式、librosa 既定) ----------
  function hzToMel(f) { const fSp = 200 / 3, minLog = 1000, minLogMel = minLog / fSp, logstep = Math.log(6.4) / 27; return f < minLog ? f / fSp : minLogMel + Math.log(f / minLog) / logstep; }
  function melToHz(m) { const fSp = 200 / 3, minLog = 1000, minLogMel = minLog / fSp, logstep = Math.log(6.4) / 27; return m < minLogMel ? fSp * m : minLog * Math.exp(logstep * (m - minLogMel)); }
  function melFilterbank(sr, nFft, nMels) {
    const nBins = nFft / 2 + 1, fmax = sr / 2;
    const mels = [], mMin = hzToMel(0), mMax = hzToMel(fmax);
    for (let i = 0; i < nMels + 2; i++) mels.push(melToHz(mMin + (mMax - mMin) * i / (nMels + 1)));
    const fb = [];
    for (let m = 0; m < nMels; m++) {
      const lo = mels[m], c = mels[m + 1], hi = mels[m + 2];
      const w = new Float32Array(nBins);
      const enorm = 2 / (hi - lo);
      for (let k = 0; k < nBins; k++) {
        const f = k * sr / nFft;
        const up = (f - lo) / (c - lo), down = (hi - f) / (hi - c);
        w[k] = Math.max(0, Math.min(up, down)) * enorm;
      }
      fb.push(w);
    }
    return fb;
  }

  // ---------- 立ち上がり包絡(スペクトラルフラックス) ----------
  function onsetEnvelope(y, sr, nFft, hop, wantChroma) {
    const fft = makeFFT(nFft);
    // ビン → 音階クラス(60Hz〜5kHz)。小節頭の和音の変化を見るため
    const binClass = new Int8Array(nFft / 2 + 1).fill(-1);
    for (let k = 1; k <= nFft / 2; k++) { const f = k * sr / nFft; if (f >= 60 && f <= 5000) binClass[k] = ((Math.round(12 * Math.log2(f / 440)) % 12) + 12) % 12; }
    const win = new Float64Array(nFft);
    for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft);
    const fb = melFilterbank(sr, nFft, 128);
    const nFrames = 1 + Math.floor(y.length / hop);
    const half = nFft >> 1;
    const re = new Float64Array(nFft), im = new Float64Array(nFft);
    let prev = null;
    const env = new Float32Array(nFrames);
    const chroma = wantChroma ? new Float32Array(nFrames * 12) : null;
    // 低域(〜250Hz、キック・ベース)だけの立ち上がり。裏拍にギターが乗る区間でも拍の位置を保つため
    const melCenters = []; { const mMin = hzToMel(0), mMax = hzToMel(sr / 2); for (let i = 1; i <= 128; i++) melCenters.push(melToHz(mMin + (mMax - mMin) * i / 129)); }
    const nLow = Math.max(3, melCenters.filter(f => f < 250).length);
    const envLow = wantChroma ? new Float32Array(nFrames) : null;
    const mel = new Float64Array(fb.length);
    for (let t = 0; t < nFrames; t++) {
      const start = t * hop - half;
      for (let i = 0; i < nFft; i++) {
        let idx = start + i;
        if (idx < 0) idx = -idx; else if (idx >= y.length) idx = 2 * (y.length - 1) - idx;   // 反射パディング
        re[i] = (idx >= 0 && idx < y.length ? y[idx] : 0) * win[i]; im[i] = 0;
      }
      fft(re, im);
      if (chroma) { const o = t * 12; for (let k = 1; k <= half; k++) { const c = binClass[k]; if (c >= 0) chroma[o + c] += re[k] * re[k] + im[k] * im[k]; } for (let c = 0; c < 12; c++) chroma[o + c] = Math.log10(1 + 1e3 * chroma[o + c]); }
      for (let m = 0; m < fb.length; m++) {
        const w = fb[m]; let s = 0;
        for (let k = 0; k <= half; k++) if (w[k]) s += w[k] * (re[k] * re[k] + im[k] * im[k]);
        mel[m] = 10 * Math.log10(Math.max(s, 1e-10));
      }
      if (prev) {
        let s = 0, sl = 0;
        for (let m = 0; m < fb.length; m++) { const d = Math.max(0, mel[m] - prev[m]); s += d; if (m < nLow) sl += d; }
        env[t] = s / fb.length;
        if (envLow) envLow[t] = sl / nLow;
      }
      prev = prev || new Float64Array(fb.length);
      prev.set(mel);
    }
    return wantChroma ? { env, chroma, envLow } : env;
  }

  // ---------- RMS(時間領域のエネルギー) ----------
  function rmsEnvelope(y, frame, hop) {
    const n = 1 + Math.floor(y.length / hop), half = frame >> 1, out = new Float32Array(n);
    for (let t = 0; t < n; t++) {
      let s = 0, c = 0;
      for (let i = t * hop - half; i < t * hop + half; i++) if (i >= 0 && i < y.length) { s += y[i] * y[i]; c++; }
      out[t] = c ? Math.sqrt(s / c) : 0;
    }
    return out;
  }
  // 2次バターワースのローパスを前後から掛けて位相ずれ無しに(4次相当)
  function lowpass(y, sr, fc) {
    const w0 = 2 * Math.PI * fc / sr, q = Math.SQRT1_2, alpha = Math.sin(w0) / (2 * q), cw = Math.cos(w0);
    const b0 = (1 - cw) / 2, b1 = 1 - cw, b2 = (1 - cw) / 2, a0 = 1 + alpha, a1 = -2 * cw, a2 = 1 - alpha;
    const run = (src) => {
      const out = new Float32Array(src.length); let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
      for (let i = 0; i < src.length; i++) {
        const x0 = src[i], y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        out[i] = y0; x2 = x1; x1 = x0; y2 = y1; y1 = y0;
      }
      return out;
    };
    const f = run(y); f.reverse(); const g = run(f); g.reverse(); return g;
  }

  function analyzeBeats(y, sr, opts) {
    opts = opts || {};
    const log = opts.log || (() => {});
    const hop = 512, nFft = 2048, fps = sr / hop, T = y.length / sr;
    const { env: onset, chroma, envLow: onsetLow } = onsetEnvelope(y, sr, nFft, hop, true);

    const envAt = (env, t) => { const f = Math.min(Math.max(Math.round(t * fps), 1), env.length - 2); return Math.max(env[f - 1], env[f], env[f + 1]); };
    // 追従用: 前後1フレームの max を取らず、線形補間で山の位置を鋭く見る
    const envSharp = (t) => { const x = Math.min(Math.max(t * fps, 0), onset.length - 1.001), f = Math.floor(x), w = x - f; return onset[f] * (1 - w) + onset[f + 1] * w + onsetLowW * (onsetLow[f] * (1 - w) + onsetLow[f + 1] * w); };
    // 低域の重み: 全帯域と同じくらいの平均になるよう正規化
    let onsetLowW = 0; { let a = 0, b = 0; for (let i = 0; i < onset.length; i++) { a += onset[i]; b += onsetLow[i]; } onsetLowW = b > 0 ? a / b : 0; }
    function gridScore(bpm, off, a, b) {
      a = a || 0; b = b === undefined ? T : b;
      const per = 60 / bpm; let s = 0, n = 0;
      let t = off; if (t < a) t += Math.ceil((a - t) / per) * per;
      for (; t < b; t += per) { s += envAt(onset, t); n++; }
      return n ? s / n : 0;
    }
    // bpm0 の±pct% で BPM と位相を総当たりで詰める(粗→細)。範囲 [a,b) に限定できる
    function refine(bpm0, a, b, pct) {
      a = a || 0; b = b === undefined ? T : b; pct = pct || 0.03;
      let best = [-1, bpm0, a];
      const stepC = Math.max(0.02, bpm0 * pct / 30);
      for (let bpm = bpm0 * (1 - pct); bpm < bpm0 * (1 + pct); bpm += stepC) {
        const per = 60 / bpm;
        for (let off = a; off < a + per; off += per / 40) { const sc = gridScore(bpm, off, a, b); if (sc > best[0]) best = [sc, bpm, off]; }
      }
      const b1 = best[1], o1 = best[2];
      const stepF = Math.max(0.005, stepC / 10);
      for (let bpm = b1 - stepC * 1.5; bpm < b1 + stepC * 1.5; bpm += stepF) {
        const per = 60 / bpm;
        for (let off = o1 - per / 40; off < o1 + per / 40; off += 0.003) { const sc = gridScore(bpm, off, a, b); if (sc > best[0]) best = [sc, bpm, off]; }
      }
      return best;
    }

    // 局所平均を引いた包絡と、その自己相関(範囲指定可)
    const sm = new Float32Array(onset.length);
    { const w = Math.round(fps * 2); let s = 0; const q = [];
      for (let i = 0; i < onset.length; i++) { q.push(onset[i]); s += onset[i]; if (q.length > w) s -= q.shift(); sm[i] = onset[i] - s / q.length; } }
    const lagMin = Math.floor(60 / 220 * fps), lagMax = Math.ceil(60 / 60 * fps);
    function autocorr(f0, f1) {
      f0 = f0 || 0; f1 = f1 === undefined ? sm.length : Math.min(f1, sm.length);
      const ac = new Float64Array(lagMax + 1);
      let e = 0; for (let i = f0; i < f1; i++) e += sm[i] * sm[i]; e /= Math.max(1, f1 - f0);
      for (let lag = lagMin; lag <= lagMax; lag++) { let s = 0, n = 0; for (let i = f0 + lag; i < f1; i++) { s += sm[i] * sm[i - lag]; n++; } ac[lag] = n ? s / n / (e || 1) : 0; }
      return ac;
    }
    function acPeaks(ac, max) {
      const peaks = [];
      for (let lag = lagMin + 1; lag < lagMax; lag++) if (ac[lag] > ac[lag - 1] && ac[lag] >= ac[lag + 1]) peaks.push([ac[lag], 60 * fps / lag]);
      peaks.sort((a, b) => b[0] - a[0]);
      const out = [];
      const add = (bpm) => { if (bpm >= 60 && bpm <= 220 && out.every(c => Math.abs(bpm - c) / c > 0.08)) out.push(bpm); };
      for (const [, bpm] of peaks) { add(bpm); if (out.length >= max) break; }
      return { out, add };
    }
    const acAt = (ac, bpm) => { const l = 60 / bpm * fps; let m = -1; for (const lag of [Math.floor(l), Math.ceil(l)]) if (lag >= lagMin && lag <= lagMax) m = Math.max(m, ac[lag]); return Math.max(m, 0.02); };

    // 候補の中から、コントラスト(拍上 − 半拍)× 自己相関 × 事前分布 で選ぶ。倍/半分が並んだらコントラストで決める
    function chooseTempo(cands, a, b, ac, priorCenter, priorWidth, pct) {
      const results = [];
      for (const c of cands) {
        const [sc, bpm, off] = refine(c, a, b, pct);
        const half = gridScore(bpm, off + 30 / bpm, a, b);
        const contrast = sc - half;
        const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / priorCenter) / priorWidth, 2));
        results.push({ score: contrast * prior * acAt(ac, bpm), contrast, bpm, off });
      }
      // 付点(1.5倍・2.5倍…)の周期のピークが他にあるなら、その候補が本当の拍で、他は拍の上の強弱パターン。
      // 空奏列車(175BPM)で 2/3 の 116.6 が「コントラスト」で勝っていたので、半整数比で説明できる候補を強く優先する
      for (const r of results) {
        let ev = 0;
        for (const o of results) { const ratio = (60 / o.bpm) / (60 / r.bpm); if (ratio > 1.2 && Math.abs(ratio - Math.round(ratio - 0.5) - 0.5) < 0.04) ev++; }
        r.evidence = ev; r.score *= 1 + 2 * ev;
      }
      results.sort((x, y) => y.score - x.score);
      let top = results[0];
      for (const r of results.slice(1)) {
        const ratio = r.bpm / top.bpm;
        if (Math.abs(ratio - 2) < 0.06 && (r.contrast >= top.contrast * 0.8 || r.evidence > top.evidence)) top = r;
        else if (Math.abs(ratio - 0.5) < 0.015 && top.contrast < r.contrast * 0.8 && !(top.evidence > r.evidence)) top = r;
      }
      return { top, results };
    }

    // 1. 曲全体のテンポ
    const acG = autocorr();
    const { out: cands, add } = acPeaks(acG, 6);
    for (const c of cands.slice(0, 4)) { add(c * 2); add(c / 2); }
    if (!cands.length) cands.push(120);
    const g = chooseTempo(cands, 0, T, acG, 120, 0.8, 0.03);
    for (const r of g.results) log(`候補 ${r.bpm.toFixed(2)} BPM コントラスト ${r.contrast.toFixed(2)} 付点の根拠 ${r.evidence} 評価 ${r.score.toFixed(3)}`);
    const gBpm = g.top.bpm;

    // 2. 途中でテンポが変わる曲: 12秒窓(4秒刻み)ごとの局所テンポを測り、2%以上違う区間に分ける
    const segments = [];   // { a, b, bpm0 }
    const W = 12, STEP = 4;
    if (T >= 30) {
      const wins = [];
      for (let a = 0; a + W <= T + 1e-9; a += STEP) {
        const b = a + W;
        const acL = autocorr(Math.round(a * fps), Math.round(b * fps));
        const { out: lc, add: addL } = acPeaks(acL, 3);
        addL(gBpm);
        const r = chooseTempo(lc, a, b, acL, gBpm, 0.6, 0.06);   // 全体テンポ中心の事前分布で倍/半分の取り違えを防ぐ。±6% は自己相関の整数ラグの粗さ(140BPM付近で5%刻み)を吸収するため
        // 全体テンポのグリッド(位相はこの窓で合わせ直す)のコントラストと比べ、局所テンポが明らかに勝つ窓だけ「変化」とみなす。
        // 半分・2/3・3/4 など単純な比の候補は拍の階層の取り違えなので変化と数えない(一定テンポの曲の静かな所で 83 や 110 が出た)
        const gr = refine(gBpm, a, b, 0.015);
        const gContrast = gr[0] - gridScore(gr[1], gr[2] + 30 / gr[1], a, b);
        const ratio = r.top.bpm / gBpm;
        const simple = [0.5, 2, 2 / 3, 1.5, 0.75, 4 / 3, 1 / 3, 3].some(q => Math.abs(ratio / q - 1) < 0.04);   // 局所推定は3%程度ずれる(付点8分の区間で 137 が 141 と出た)ので 4% まで同一視
        const changed = !simple && Math.abs(ratio - 1) > 0.06 && r.top.contrast > gContrast * 1.6 && r.top.contrast > 0.05;   // 6%未満の変化・ゆるやかな変化は追従(トラッカー)が吸収する
        wins.push({ a, b, bpm: changed ? r.top.bpm : gr[1], contrast: r.top.contrast, changed });
        if (opts.debugWins) { const perG = 60 / gBpm; const ph = (((gr[2] - g.top.off) % perG) + perG) % perG; log(`窓 ${a}s: 局所 ${r.top.bpm.toFixed(1)} c=${r.top.contrast.toFixed(2)} 全体グリッド ${gr[1].toFixed(2)} c=${gContrast.toFixed(2)} 位相 ${(ph * 1000).toFixed(0)}ms ${changed ? '変化' : ''}`); }
      }
      // 「変化」と判定された窓が、テンポの近い(2%以内)まま 3窓以上(12秒以上)続いたときだけ別区間にする。
      // 1〜2窓だけの変化はフィルや静かな所での取り違え(一定テンポの曲で 99〜108秒に 127BPM が出た)
      const groups = [];
      for (let i = 0; i < wins.length; i++) {
        const w = wins[i];
        const g0 = groups[groups.length - 1];
        if (w.changed && g0 && g0.open && Math.abs(w.bpm / g0.bpm - 1) < 0.02) { g0.j = i; g0.sum += w.bpm; g0.n++; g0.bpm = g0.sum / g0.n; }
        else if (w.changed) { if (g0) g0.open = false; groups.push({ i, j: i, bpm: w.bpm, sum: w.bpm, n: 1, open: true }); }
        else if (g0) g0.open = false;
      }
      let cursor = 0;
      for (const gr of groups.filter(x => x.n >= 3)) {
        const a = wins[gr.i].a + (W - STEP) / 2, b = wins[gr.j].a + W - (W - STEP) / 2;
        if (a - cursor >= 8) segments.push({ a: cursor, b: a, bpm0: gBpm });
        else if (segments.length) segments[segments.length - 1].b = a; else { /* 先頭が変化区間 */ }
        segments.push({ a: segments.length ? a : 0, b, bpm0: gr.bpm });
        cursor = b;
      }
      if (segments.length) { if (T - cursor >= 8) segments.push({ a: cursor, b: T, bpm0: gBpm }); else segments[segments.length - 1].b = T; }
    }
    if (!segments.length) segments.push({ a: 0, b: T, bpm0: gBpm });
    if (segments.length > 1) log(`テンポ区間 ${segments.map(sg => `${sg.a.toFixed(0)}-${sg.b.toFixed(0)}s ${sg.bpm0.toFixed(1)}`).join(', ')}`);

    // 3. 立ち上がり基準の位相補正に使う、時間領域のエネルギー増加
    const hopE = 64, fpsE = sr / hopE;
    const rms = rmsEnvelope(y, 256, hopE);
    const denv = new Float32Array(rms.length);
    for (let i = 1; i < rms.length; i++) denv[i] = Math.max(0, rms[i] - rms[i - 1]);

    // 4. 区間ごとに BPM×位相を詰めてグリッドを置く。1区間なら 20秒窓の位相ずれも補間で吸収
    let beats = [], segStarts = [], tempi = [], biasSum = 0;
    const fitSegments = () => { biasSum = 0; for (const sg of segments) {
      const single = segments.length === 1;
      const [, bpm, off0] = single ? [0, g.top.bpm, g.top.off] : refine(sg.bpm || sg.bpm0, sg.a, sg.b, 0.03);
      const per = 60 / bpm;
      // 追従: 8秒窓を4秒ずつ進め、直前の拍から外挿した位置の±半拍・テンポ±3%の範囲で最も音に乗る拍列を選ぶ。
      // 固定グリッドだと、生バンドの曲(90'S TOKYO BOYS で 102.6→100 BPM に2.5%遅くなる)で後半の拍が1拍ずれた
      const W2 = 8, ST = 4;
      const grid = [];
      let curBpm = bpm;
      let t0 = off0; while (t0 - per >= Math.max(0, sg.a - 6)) t0 -= per;
      let t = t0; while (t < sg.a) { grid.push(t); t += per; }
      let last = grid.length ? grid[grid.length - 1] : t0 - per;
      for (let a = sg.a; a < sg.b; a += ST) {
        const b = Math.min(a + W2, sg.b + 6);
        let best = null;
        for (let bpmC = curBpm * 0.97; bpmC <= curBpm * 1.03; bpmC += curBpm * 0.0025) {
          const perC = 60 / bpmC;
          for (let d = -perC / 4; d <= perC / 4; d += 0.004) {   // 1窓で動かせるのは±1/4拍まで(裏拍に乗ったギターに引かれて半拍飛ぶのを防ぐ)
            let sc = 0, n = 0;
            for (let tt = last + perC + d; tt < b; tt += perC) { sc += envSharp(tt); n++; }
            if (!n) continue;
            // 連続性: 外挿からのずれとテンポの変化に軽いペナルティ(音が無い所では外挿を保つ)
            sc = (sc / n) * (1 - 1.0 * Math.abs(bpmC / curBpm - 1) - 0.3 * Math.pow(d / (perC / 4), 2));   // ずれは2乗で軽く抑える(小さな補正は自由、大きな飛びは高い)
            if (!best || sc > best.sc) best = { sc, bpm: bpmC, d, per: perC };
          }
        }
        const end = Math.min(a + ST, sg.b);
        let tt = last + best.per + best.d;
        while (tt < end) { grid.push(tt); last = tt; tt += best.per; }
        curBpm = best.bpm;
      }
      { let tt = last + 60 / curBpm; while (tt < Math.min(T, sg.b + 6)) { grid.push(tt); tt += 60 / curBpm; } }
      // 立ち上がり基準の位相補正(追従後の拍列全体を、時間領域のエネルギー増加が最大になる位置へ)
      let eb = [-1, 0];
      for (let d = -per / 3; d < per / 3; d += 0.001) {
        let sc = 0, n = 0;
        for (const v of grid) { if (v < sg.a || v >= sg.b) continue; const f = Math.min(Math.max(Math.round((v + d) * fpsE), 0), denv.length - 1); sc += denv[f]; n++; }
        sc = n ? sc / n : 0;
        if (sc > eb[0]) eb = [sc, d];
      }
      const bias = eb[1]; biasSum += bias;
      for (let i = 0; i < grid.length; i++) grid[i] += bias;
      sg.grid = grid; sg.bpm = bpm;
    } };
    // 4b. 隣り合う区間の境界を、両側のグリッドが最も音に乗る位置に動かす(±6秒、0.25秒刻み)。
    //     境界が動くと区間のフィットも変わるので、フィット→境界→フィット→境界 と2回繰り返す
    const refineBoundaries = () => { for (let i = 1; i < segments.length; i++) {
      const p = segments[i - 1], q = segments[i];
      let best = [-Infinity, q.a];
      for (let tau = q.a - 6; tau <= q.a + 6; tau += 0.25) {
        if (tau <= p.a + 4 || tau >= q.b - 4) continue;
        // 拍上の強さから半拍ずれた位置の強さを引く(コントラスト)。拍の数の差で評価が偏らないように
        let sc = 0;
        const hp = 30 / p.bpm, hq = 30 / q.bpm;
        for (const v of p.grid) if (v >= tau - 6 && v < tau) sc += envAt(onset, v) - envAt(onset, v + hp);
        for (const v of q.grid) if (v >= tau && v < tau + 6) sc += envAt(onset, v) - envAt(onset, v + hq);
        if (sc > best[0]) best = [sc, tau];
        if (opts.debugWins) log(`境界候補 ${tau.toFixed(2)} 評価 ${sc.toFixed(2)}`);
      }
      p.b = q.a = best[1];
    } };
    fitSegments();
    if (segments.length > 1) { refineBoundaries(); fitSegments(); refineBoundaries(); }
    for (const sg of segments) {
      segStarts.push(beats.length);
      for (const v of sg.grid) if (v >= sg.a && v < sg.b) beats.push(v);
      tempi.push({ start: Math.round(sg.a * 100) / 100, bpm: Math.round(sg.bpm * 100) / 100 });
    }
    // 4c. 曲の前後の無音(RMS が曲の中央値の 8% 未満)に乗った拍は落とす
    {
      const med = Array.from(rms).filter(v => v > 0).sort((x, y) => x - y); const thr = (med[med.length >> 1] || 0) * 0.08;
      const loud = (t) => { const f0 = Math.max(0, Math.round(t * fpsE)), f1 = Math.min(rms.length, Math.round((t + 0.5) * fpsE)); let m = 0; for (let f = f0; f < f1; f++) m = Math.max(m, rms[f]); return m >= thr; };
      let lo = 0; while (lo < beats.length && !loud(beats[lo])) lo++;
      let hi = beats.length; while (hi > lo && !loud(beats[hi - 1] - 0.5)) hi--;
      if (lo > 0 || hi < beats.length) { beats = beats.slice(lo, hi); segStarts = segStarts.map(v => Math.min(Math.max(v - lo, 0), beats.length)); }
    }

    // 5. 拍の強さ(1拍目推定用): 拍直後 50ms のエネルギー増加(全帯域 + 200Hz以下)と、拍ごとの和音の変化
    const rmsLow = rmsEnvelope(lowpass(y, sr, 110), 256, hopE);   // 110Hz 以下 = ほぼキック(200Hz だとベースが混ざって拍の区別が付かなかった)
    const denvLow = new Float32Array(rmsLow.length);
    for (let i = 1; i < rmsLow.length; i++) denvLow[i] = Math.max(0, rmsLow[i] - rmsLow[i - 1]);
    const riseAt = (env, t) => { const f0 = Math.max(0, Math.round((t - 0.01) * fpsE)), f1 = Math.min(env.length - 1, Math.round((t + 0.05) * fpsE)); let m = 0; for (let f = f0; f <= f1; f++) m = Math.max(m, env[f]); return m; };
    const oAll = beats.map(t => riseAt(denv, t)), oLow = beats.map(t => riseAt(denvLow, t));
    // 拍 i の区間のクロマ(平均・正規化)と、直前の拍とのコサイン距離
    const beatChroma = beats.map((t, i) => {
      const t1 = i + 1 < beats.length ? beats[i + 1] : Math.min(T, t + 0.5);
      const f0 = Math.min(Math.max(Math.round(t * fps), 0), onset.length - 1), f1 = Math.min(Math.max(Math.round(t1 * fps), f0 + 1), onset.length);
      const v = new Float32Array(12);
      for (let f = f0; f < f1; f++) for (let c = 0; c < 12; c++) v[c] += chroma[f * 12 + c];
      let n = 0; for (let c = 0; c < 12; c++) n += v[c] * v[c]; n = Math.sqrt(n) || 1;
      for (let c = 0; c < 12; c++) v[c] /= n;
      return v;
    });
    const oChg = beats.map((_, i) => { if (!i) return 0; let d = 0; for (let c = 0; c < 12; c++) d += beatChroma[i][c] * beatChroma[i - 1][c]; return Math.max(0, 1 - d); });
    const mA = Math.max(...oAll, 1e-9), mL = Math.max(...oLow, 1e-9), mC = Math.max(...oChg, 1e-9);
    // 小節幅のクロマ新規性: 拍 i の前4拍と後4拍の和音の違い(コードは小節頭で変わることが多い)
    const winChroma = (i0, i1) => { const v = new Float32Array(12); for (let i = Math.max(0, i0); i < Math.min(beats.length, i1); i++) for (let c = 0; c < 12; c++) v[c] += beatChroma[i][c]; let n = 0; for (let c = 0; c < 12; c++) n += v[c] * v[c]; n = Math.sqrt(n) || 1; for (let c = 0; c < 12; c++) v[c] /= n; return v; };
    const oNov = beats.map((_, i) => { if (i < 4 || i + 4 > beats.length) return 0; const a = winChroma(i - 4, i), b = winChroma(i, i + 4); let d = 0; for (let c = 0; c < 12; c++) d += a[c] * b[c]; return Math.max(0, 1 - d); });
    const mN = Math.max(...oNov, 1e-9);
    const strength = beats.map((_, i) => Math.round((0.3 * oAll[i] / mA + 0.2 * oLow[i] / mL + 0.2 * oChg[i] / mC + 0.3 * oNov[i] / mN) * 1e4) / 1e4);
    const features = opts.features ? { rise: oAll.map(v => v / mA), low: oLow.map(v => v / mL), chg: oChg.map(v => v / mC), nov: oNov.map(v => v / mN) } : undefined;

    return {
      tempo: Math.round(g.top.bpm * 100) / 100,
      beats: beats.map(t => Math.round(t * 1e4) / 1e4),
      strength,
      segStarts,                       // 各テンポ区間の最初の拍の番号(先頭は 0)
      tempi,                           // 区間ごとの { start(秒), bpm }
      mode: segments.length > 1 ? 'segments' : 'grid',
      duration: Math.round(T * 100) / 100,
      phaseBiasMs: Math.round(biasSum / segments.length * 1000),
      candidates: g.results.map(r => Math.round(r.bpm * 100) / 100),
      features,
    };
  }

  return { analyzeBeats, onsetEnvelope, rmsEnvelope, lowpass };
});
