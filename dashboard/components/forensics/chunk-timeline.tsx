'use client'

import { motion } from 'framer-motion'
import { Pause, Play, Rows3 } from 'lucide-react'
import { useRef, useState } from 'react'
import { GlowIcon } from '@/components/glow-icon'
import type { TimelineChunk } from '@/lib/forensics'
import { cn } from '@/lib/utils'

function chunkStyle(chunk: TimelineChunk) {
  if (chunk.fake_probability == null || /silence/i.test(chunk.verdict)) {
    return { bar: 'bg-slate-600/60 border-slate-500/40', glow: 'rgba(100,116,139,0.6)', text: 'text-slate-300' }
  }
  if (/synthetic|clone/i.test(chunk.verdict)) {
    return { bar: 'bg-neon-crimson/80 border-neon-crimson', glow: 'rgba(255,0,85,0.9)', text: 'text-neon-crimson' }
  }
  return { bar: 'bg-neon-emerald/80 border-neon-emerald', glow: 'rgba(0,255,102,0.85)', text: 'text-neon-emerald' }
}

function fmtTime(s: number) {
  if (!Number.isFinite(s)) return '0:00'
  const m = Math.floor(s / 60)
  return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}`
}

export function ChunkTimeline({ chunks, audioUrl }: { chunks: TimelineChunk[]; audioUrl: string | null }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const [playing, setPlaying] = useState(false)
  const [current, setCurrent] = useState(0)
  const [hovered, setHovered] = useState<number | null>(null)

  const total = Math.max(audioRef.current?.duration || 0, chunks.at(-1)?.end_seconds ?? 0, 1)
  const progress = Math.min(100, (current / total) * 100)

  function seek(seconds: number) {
    setCurrent(seconds)
    const audio = audioRef.current
    if (audio && audioUrl) {
      audio.currentTime = seconds
      void audio.play()
    }
  }

  function toggle() {
    const audio = audioRef.current
    if (!audio || !audioUrl) return
    if (audio.paused) void audio.play()
    else audio.pause()
  }

  return (
    <section aria-labelledby="timeline-title" className="glass flex flex-col gap-5 rounded-2xl p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <GlowIcon icon={Rows3} label="Chunk timeline" tone="violet" size="sm" />
          <div>
            <h3 id="timeline-title" className="font-semibold text-white">2-Second Chunk Timeline</h3>
            <p className="text-xs text-muted-foreground">Hover a window for details · click to seek playback</p>
          </div>
        </div>
        <ul className="flex flex-wrap gap-3 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <li className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-neon-crimson" />Synthetic/Clone</li>
          <li className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-neon-emerald" />Authentic</li>
          <li className="flex items-center gap-1.5"><span className="size-2.5 rounded-sm bg-slate-500" />Silence</li>
        </ul>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={toggle}
          disabled={!audioUrl}
          aria-label={playing ? 'Pause audio' : 'Play audio'}
          className="flex size-10 shrink-0 items-center justify-center rounded-full border border-neon-cyan/60 bg-neon-cyan/10 text-neon-cyan transition-all duration-300 ease-out hover:-translate-y-1 hover:scale-115 hover:shadow-[0_0_24px_rgba(0,240,255,0.9)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-cyan/70 disabled:opacity-40 disabled:hover:translate-y-0 disabled:hover:scale-100 disabled:hover:shadow-none"
        >
          {playing ? <Pause className="size-4" aria-hidden="true" /> : <Play className="size-4" aria-hidden="true" />}
        </button>
        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/10">
          <div
            className="absolute inset-y-0 left-0 rounded-full bg-gradient-to-r from-neon-cyan to-neon-violet shadow-[0_0_12px_rgba(0,240,255,0.8)]"
            style={{ width: `${progress}%` }}
          />
        </div>
        <span className="w-24 text-right font-mono text-xs tabular-nums text-muted-foreground">
          {fmtTime(current)} / {fmtTime(total)}
        </span>
        {audioUrl && (
          <audio
            ref={audioRef}
            src={audioUrl}
            preload="metadata"
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
            className="hidden"
          />
        )}
      </div>
      {!audioUrl && (
        <p className="-mt-2 text-xs text-muted-foreground">Sample report · upload a file to enable playback.</p>
      )}

      <div className="relative pb-24">
        <div className="relative flex h-28 items-end gap-1 sm:gap-1.5" role="list" aria-label="Analysis windows">
          {chunks.map((chunk) => {
            const style = chunkStyle(chunk)
            const pct = chunk.fake_probability == null ? null : chunk.fake_probability * (chunk.fake_probability <= 1 ? 100 : 1)
            const height = pct == null ? 18 : Math.max(14, pct)
            const active = current >= chunk.start_seconds && current < chunk.end_seconds
            return (
              <div key={chunk.chunk_id} role="listitem" className="relative flex h-full flex-1 items-end">
                <motion.button
                  type="button"
                  onMouseEnter={() => setHovered(chunk.chunk_id)}
                  onMouseLeave={() => setHovered(null)}
                  onFocus={() => setHovered(chunk.chunk_id)}
                  onBlur={() => setHovered(null)}
                  onClick={() => seek(chunk.start_seconds)}
                  whileHover={{ scaleY: 1.08, y: -6, scaleX: 1.12 }}
                  transition={{ type: 'spring', stiffness: 420, damping: 18 }}
                  aria-label={`${chunk.time_window}: ${chunk.verdict}${pct == null ? '' : `, ${pct.toFixed(1)}% fake probability`}`}
                  className={cn(
                    'w-full origin-bottom rounded-md border outline-none transition-shadow duration-300 focus-visible:ring-2 focus-visible:ring-white/70',
                    style.bar,
                    active && 'ring-2 ring-neon-cyan',
                  )}
                  style={{
                    height: `${height}%`,
                    boxShadow: hovered === chunk.chunk_id || active ? `0 0 22px ${style.glow}` : `0 0 8px ${style.glow.replace(/[\d.]+\)$/, '0.3)')}`,
                  }}
                />
                {hovered === chunk.chunk_id && (
                  <motion.div
                    role="tooltip"
                    initial={{ opacity: 0, y: -4, scale: 0.92 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    className="pointer-events-none absolute left-1/2 top-full z-30 mt-3 w-40 -translate-x-1/2 rounded-xl border border-neon-cyan/40 bg-void-2/90 p-3 text-left shadow-[0_0_24px_-4px_rgba(0,240,255,0.6)] backdrop-blur-xl"
                  >
                    <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Chunk #{chunk.chunk_id}</p>
                    <p className="font-mono text-sm text-white">{chunk.time_window}</p>
                    <p className={cn('mt-1 font-mono text-lg font-semibold tabular-nums', style.text)}>
                      {pct == null ? '—' : `${pct.toFixed(1)}%`}
                    </p>
                    <p className={cn('text-xs font-medium', style.text)}>{chunk.verdict}</p>
                  </motion.div>
                )}
              </div>
            )
          })}
        </div>
        <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground">
          <span>0.0s</span>
          <span>{(chunks.at(-1)?.end_seconds ?? 0).toFixed(1)}s</span>
        </div>
      </div>
    </section>
  )
}
