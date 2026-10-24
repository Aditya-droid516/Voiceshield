'use client'

import { motion } from 'framer-motion'
import {
  Binary,
  FileSearch,
  Info,
  Layers,
  Music2,
  Repeat,
  ScanSearch,
  Scissors,
  SlidersHorizontal,
  Tags,
  type LucideIcon,
} from 'lucide-react'
import { GlowIcon, type NeonTone } from '@/components/glow-icon'
import type { MetadataInspection, TechnicalSpecs } from '@/lib/forensics'
import { cn } from '@/lib/utils'

const KIND_UI: Record<string, { icon: LucideIcon; tone: NeonTone }> = {
  editor: { icon: Scissors, tone: 'crimson' },
  converter: { icon: Repeat, tone: 'amber' },
  pro_audio: { icon: SlidersHorizontal, tone: 'violet' },
  info: { icon: Info, tone: 'cyan' },
}

const SOURCE_ICON: Record<string, LucideIcon> = { tags: Tags, raw_bytes: Binary, wav_chunk: Layers }

const CONFIDENCE_TONE: Record<string, string> = {
  high: 'border-neon-crimson/60 bg-neon-crimson/10 text-neon-crimson',
  medium: 'border-neon-amber/60 bg-neon-amber/10 text-neon-amber',
  low: 'border-neon-cyan/50 bg-neon-cyan/10 text-neon-cyan',
  none: 'border-neon-emerald/60 bg-neon-emerald/10 text-neon-emerald',
}

function Flag({ label, value, danger }: { label: string; value: string; danger: string }) {
  return (
    <motion.div
      whileHover={{ y: -5, scale: 1.04 }}
      transition={{ type: 'spring', stiffness: 380, damping: 20 }}
      className="flex flex-col gap-2 rounded-xl border border-white/10 bg-white/5 p-3 transition-colors duration-300 hover:border-neon-cyan/60"
    >
      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
      <span className={cn('w-fit rounded-full border px-2.5 py-0.5 font-mono text-xs font-semibold uppercase tracking-wider', danger)}>
        {value}
      </span>
    </motion.div>
  )
}

function boolBadge(v?: boolean) {
  return v ? CONFIDENCE_TONE.high : CONFIDENCE_TONE.none
}

export function MetadataInspector({ data }: { data?: MetadataInspection }) {
  if (!data) return null
  const specs: TechnicalSpecs = data.technical_specs ?? data
  const findings = data.findings ?? []
  const confidence = (data.edit_confidence ?? 'none').toLowerCase()

  const specRows: [string, string][] = [
    ['Format', specs.format ?? '—'],
    ['Duration', specs.duration_seconds != null ? `${specs.duration_seconds.toFixed(2)} s` : '—'],
    ['Sample rate', specs.sample_rate_hz != null ? `${specs.sample_rate_hz.toLocaleString()} Hz` : '—'],
    ['Channels', specs.channels != null ? String(specs.channels) : '—'],
    ['Bitrate', specs.bitrate_kbps != null ? `${specs.bitrate_kbps} kbps` : '—'],
  ]

  return (
    <section aria-labelledby="metadata-title" className="glass flex flex-col gap-5 rounded-2xl p-5">
      <div className="flex items-center gap-3">
        <GlowIcon icon={FileSearch} label="Metadata inspector" tone="cyan" size="sm" />
        <div>
          <h3 id="metadata-title" className="font-semibold text-white">Forensic Metadata Inspector</h3>
          <p className="text-xs text-muted-foreground">Container tags, raw byte signatures and WAV chunks</p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <Flag label="Potentially edited" value={data.potentially_edited ? 'Yes' : 'No'} danger={boolBadge(data.potentially_edited)} />
        <Flag label="Edit confidence" value={confidence} danger={CONFIDENCE_TONE[confidence] ?? CONFIDENCE_TONE.low} />
        <Flag label="Re-encoded suspected" value={data.re_encoded_suspected ? 'Yes' : 'No'} danger={boolBadge(data.re_encoded_suspected)} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <h4 className="font-mono text-[11px] uppercase tracking-[0.2em] text-neon-violet">Findings · {findings.length}</h4>
          {findings.length === 0 ? (
            <p className="rounded-xl border border-white/10 bg-white/5 p-4 text-sm text-muted-foreground">
              No editor, converter or pro-audio signatures found.
            </p>
          ) : (
            <ul className="flex flex-col gap-2">
              {findings.map((f, i) => {
                const kind = KIND_UI[f.kind] ?? KIND_UI.info
                const SourceIcon = SOURCE_ICON[f.source] ?? ScanSearch
                return (
                  <motion.li
                    key={`${f.source}-${f.detail}-${i}`}
                    whileHover={{ x: 4, scale: 1.02 }}
                    transition={{ type: 'spring', stiffness: 380, damping: 22 }}
                    className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 p-3 transition-colors duration-300 hover:border-neon-cyan/50"
                  >
                    <GlowIcon icon={kind.icon} label={f.kind.replace('_', ' ')} tone={kind.tone} size="sm" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-white">{f.detail}</p>
                      <p className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                        <SourceIcon className="size-3" aria-hidden="true" />
                        {f.source} · {f.kind}
                      </p>
                    </div>
                  </motion.li>
                )
              })}
            </ul>
          )}
        </div>

        <div className="flex flex-col gap-3">
          <h4 className="font-mono text-[11px] uppercase tracking-[0.2em] text-neon-violet">Technical specs</h4>
          <dl className="grid grid-cols-2 gap-2">
            {specRows.map(([k, v]) => (
              <motion.div
                key={k}
                whileHover={{ y: -4, scale: 1.04 }}
                transition={{ type: 'spring', stiffness: 380, damping: 20 }}
                className="rounded-xl border border-white/10 bg-white/5 p-3 transition-colors duration-300 hover:border-neon-cyan/60"
              >
                <dt className="font-mono text-[10px] uppercase tracking-[0.16em] text-muted-foreground">{k}</dt>
                <dd className="truncate font-mono text-sm text-white tabular-nums">{v}</dd>
              </motion.div>
            ))}
          </dl>
          {specs.wav_chunks && specs.wav_chunks.length > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              <Music2 className="size-3.5 text-neon-cyan" aria-hidden="true" />
              {specs.wav_chunks.map((c) => (
                <span key={c} className="rounded-md border border-neon-cyan/30 bg-neon-cyan/5 px-2 py-0.5 font-mono text-[11px] text-neon-cyan">
                  {c.trim() || c}
                </span>
              ))}
            </div>
          )}
          {specs.wav_info_tags && Object.keys(specs.wav_info_tags).length > 0 && (
            <ul className="flex flex-col gap-1 rounded-xl border border-white/10 bg-white/5 p-3 font-mono text-xs">
              {Object.entries(specs.wav_info_tags).map(([k, v]) => (
                <li key={k} className="flex gap-2">
                  <span className="text-neon-violet">{k}</span>
                  <span className="truncate text-white/80">{v}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {data.forensic_note && (
        <p className="flex items-start gap-2 rounded-xl border border-neon-amber/30 bg-neon-amber/5 p-3 text-xs leading-relaxed text-neon-amber/90">
          <Info className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          {data.forensic_note}
        </p>
      )}
    </section>
  )
}
