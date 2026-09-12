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
  function onsetEnvelope(y, sr, nFft, hop) {
    const fft = makeFFT(nFft);
    const win = new Float64Array(nFft);
    for (let i = 0; i < nFft; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / nFft);
    const fb = melFilterbank(sr, nFft, 128);
    const nFrames = 1 + Math.floor(y.length / hop);
    const half = nFft >> 1;
    const re = new Float64Array(nFft), im = new Float64Array(nFft);
    let prev = null;
    const env = new Float32Array(nFrames);
    const mel = new Float64Array(fb.length);
    for (let t = 0; t < nFrames; t++) {
      const start = t * hop - half;
      for (let i = 0; i < nFft; i++) {
        let idx = start + i;
        if (idx < 0) idx = -idx; else if (idx >= y.length) idx = 2 * (y.length - 1) - idx;   // 反射パディング
        re[i] = (idx >= 0 && idx < y.length ? y[idx] : 0) * win[i]; im[i] = 0;
      }
      fft(re, im);
      for (let m = 0; m < fb.length; m++) {
        const w = fb[m]; let s = 0;
        for (let k = 0; k <= half; k++) if (w[k]) s += w[k] * (re[k] * re[k] + im[k] * im[k]);
        mel[m] = 10 * Math.log10(Math.max(s, 1e-10));
      }
      if (prev) {
        let s = 0;
        for (let m = 0; m < fb.length; m++) s += Math.max(0, mel[m] - prev[m]);
        env[t] = s / fb.length;
      }
      prev = prev || new Float64Array(fb.length);
      prev.set(mel);
    }
    return env;
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
    const onset = onsetEnvelope(y, sr, nFft, hop);

    const envAt = (env, t) => { const f = Math.min(Math.max(Math.round(t * fps), 1), env.length - 2); return Math.max(env[f - 1], env[f], env[f + 1]); };
    function gridScore(bpm, off, a, b) {
      a = a || 0; b = b === undefined ? T : b;
      const per = 60 / bpm; let s = 0, n = 0;
      for (let t = off; t < b; t += per) if (t >= a) { s += envAt(onset, t); n++; }
      return n ? s / n : 0;
    }
    function refine(bpm0) {
      let best = [-1, bpm0, 0];
      for (let bpm = bpm0 * 0.97; bpm < bpm0 * 1.03; bpm += 0.1) {
        const per = 60 / bpm;
        for (let off = 0; off < per; off += per / 40) { const sc = gridScore(bpm, off); if (sc > best[0]) best = [sc, bpm, off]; }
      }
      const b1 = best[1], o1 = best[2];
      for (let bpm = b1 - 0.15; bpm < b1 + 0.15; bpm += 0.01) {
        const per = 60 / bpm;
        for (let off = o1 - per / 40; off < o1 + per / 40; off += 0.003) { const sc = gridScore(bpm, off); if (sc > best[0]) best = [sc, bpm, off]; }
      }
      return best;
    }

    // 1. テンポ候補: 包絡(局所平均を引く)の自己相関のピーク
    const sm = new Float32Array(onset.length);
    { const w = Math.round(fps * 2); let s = 0; const q = [];
      for (let i = 0; i < onset.length; i++) { q.push(onset[i]); s += onset[i]; if (q.length > w) s -= q.shift(); sm[i] = onset[i] - s / q.length; } }
    const lagMin = Math.floor(60 / 220 * fps), lagMax = Math.ceil(60 / 60 * fps);
    const ac = new Float64Array(lagMax + 1);
    for (let lag = lagMin; lag <= lagMax; lag++) { let s = 0; for (let i = lag; i < sm.length; i++) s += sm[i] * sm[i - lag]; ac[lag] = s / (sm.length - lag); }
    const peaks = [];
    for (let lag = lagMin + 1; lag < lagMax; lag++) if (ac[lag] > ac[lag - 1] && ac[lag] >= ac[lag + 1]) peaks.push([ac[lag], 60 * fps / lag]);
    peaks.sort((a, b) => b[0] - a[0]);
    const cands = [];
    const addCand = (bpm) => { if (bpm >= 60 && bpm <= 220 && cands.every(c => Math.abs(bpm - c) / c > 0.08)) cands.push(bpm); };
    for (const [, bpm] of peaks) { addCand(bpm); if (cands.length >= 6) break; }
    // 本当のテンポの自己相関が半分のテンポより弱く出る曲があるので、上位候補の倍と半分も必ず候補に入れる
    for (const c of cands.slice(0, 4)) { addCand(c * 2); addCand(c / 2); }
    if (!cands.length) cands.push(120);

    // 2. 候補ごとにグリッドを詰めて、コントラスト(拍上 − 半拍)× 自己相関の強さ × 120BPM 中心の事前分布で選ぶ
    //    (自己相関を掛けるのは、166BPM の曲で 2/3 の 110BPM が僅差で勝つことがあったため。110 の自己相関はほぼ 0)
    const ac0 = ac[0] || (() => { let s = 0; for (let i = 0; i < sm.length; i++) s += sm[i] * sm[i]; return s / sm.length; })();
    const acAt = (bpm) => { const l = 60 / bpm * fps; let m = -1; for (const lag of [Math.floor(l), Math.ceil(l)]) if (lag >= lagMin && lag <= lagMax) m = Math.max(m, ac[lag] / ac0); return Math.max(m, 0.02); };
    const results = [];
    for (const c of cands) {
      const [sc, bpm, off] = refine(c);
      const half = gridScore(bpm, off + 30 / bpm);
      const contrast = sc - half;
      const prior = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 120) / 0.8, 2));
      const acw = acAt(bpm);
      results.push({ score: contrast * prior * acw, contrast, bpm, off });
      log(`候補 ${c.toFixed(1)} → ${bpm.toFixed(2)} BPM コントラスト ${contrast.toFixed(2)} 自己相関 ${acw.toFixed(2)} 評価 ${(contrast * prior * acw).toFixed(3)}`);
    }
    results.sort((a, b) => b.score - a.score);
    // 倍/半分の候補が並んでいるときは、速いほうの拍にも本物の立ち上がりが乗っているか(コントラスト)で決める。
    // 166 と 83 の評価が僅差で 83 に転ぶことがあったため。速いほうのコントラストが遅いほうの 0.8 倍以上なら速いほう
    let top = results[0];
    for (const r of results.slice(1)) {
      const ratio = r.bpm / top.bpm;
      if (Math.abs(ratio - 2) < 0.06 && r.contrast >= top.contrast * 0.8) { log(`倍テンポ ${r.bpm.toFixed(2)} を採用`); top = r; }
      else if (Math.abs(ratio - 0.5) < 0.015 && top.contrast < r.contrast * 0.8) { log(`半分のテンポ ${r.bpm.toFixed(2)} を採用`); top = r; }
    }
    let { bpm, off } = top;
    let per = 60 / bpm;

    // 3. 20秒窓ごとの位相ずれ
    const win = 20, centers = [], shifts = [];
    for (let a = 0; a <= Math.max(T - win, 0) + 1e-9; a += win / 2) {
      const b = Math.min(a + win, T);
      const base = gridScore(bpm, off, a, b);
      let best = [-1, 0];
      for (let d = -per / 2; d < per / 2; d += 0.004) { const sc = gridScore(bpm, off + d, a, b); if (sc > best[0]) best = [sc, d]; }
      centers.push((a + b) / 2); shifts.push(best[0] < base * 1.03 ? 0 : best[1]);
      if (b >= T) break;
    }
    if (shifts.length >= 3) {
      const med = shifts.map((_, i) => { const s = shifts.slice(Math.max(0, i - 2), i + 3).slice().sort((x, y) => x - y); return s[s.length >> 1]; });
      for (let i = 0; i < shifts.length; i++) if (Math.abs(shifts[i] - med[i]) > per * 0.2) shifts[i] = med[i];
    }
    const maxShift = shifts.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

    // 4. 立ち上がり基準の位相補正
    const hopE = 64, fpsE = sr / hopE;
    const rms = rmsEnvelope(y, 256, hopE);
    const denv = new Float32Array(rms.length);
    for (let i = 1; i < rms.length; i++) denv[i] = Math.max(0, rms[i] - rms[i - 1]);
    function energyScore(o) { let s = 0, n = 0; for (let t = o; t < T; t += per) { const f = Math.min(Math.max(Math.round(t * fpsE), 0), denv.length - 1); s += denv[f]; n++; } return n ? s / n : 0; }
    let eb = [-1, 0];
    for (let d = -per / 3; d < per / 3; d += 0.001) { const sc = energyScore(off + d); if (sc > eb[0]) eb = [sc, d]; }
    const bias = eb[1];
    off += bias;

    const interp = (x) => {
      if (centers.length < 2) return shifts[0] || 0;
      if (x <= centers[0]) return shifts[0];
      if (x >= centers[centers.length - 1]) return shifts[shifts.length - 1];
      let i = 1; while (centers[i] < x) i++;
      const w = (x - centers[i - 1]) / (centers[i] - centers[i - 1]);
      return shifts[i - 1] * (1 - w) + shifts[i] * w;
    };
    let beats = [];
    for (let t = off; t < T; t += per) beats.push(t + interp(t));
    const mode = maxShift <= per * 0.25 ? 'grid' : 'grid-warp';
    beats = beats.filter(t => t >= 0 && t < T);

    // 拍の強さ(1拍目推定用): 拍直後 50ms のエネルギー増加(全帯域 + 200Hz以下)
    const rmsLow = rmsEnvelope(lowpass(y, sr, 200), 256, hopE);
    const denvLow = new Float32Array(rmsLow.length);
    for (let i = 1; i < rmsLow.length; i++) denvLow[i] = Math.max(0, rmsLow[i] - rmsLow[i - 1]);
    const riseAt = (env, t) => { const f0 = Math.max(0, Math.round((t - 0.01) * fpsE)), f1 = Math.min(env.length - 1, Math.round((t + 0.05) * fpsE)); let m = 0; for (let f = f0; f <= f1; f++) m = Math.max(m, env[f]); return m; };
    const oAll = beats.map(t => riseAt(denv, t)), oLow = beats.map(t => riseAt(denvLow, t));
    const mA = Math.max(...oAll, 1e-9), mL = Math.max(...oLow, 1e-9);
    const strength = beats.map((_, i) => Math.round((0.5 * oAll[i] / mA + 0.5 * oLow[i] / mL) * 1e4) / 1e4);

    return {
      tempo: Math.round(bpm * 100) / 100,
      beats: beats.map(t => Math.round(t * 1e4) / 1e4),
      strength, mode, duration: Math.round(T * 100) / 100,
      phaseBiasMs: Math.round(bias * 1000),
      candidates: results.map(r => Math.round(r.bpm * 100) / 100),
    };
  }

  return { analyzeBeats, onsetEnvelope, rmsEnvelope, lowpass };
});
