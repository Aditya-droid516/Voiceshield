'use client'

import { motion } from 'framer-motion'
import { Activity, BarChart3, FileAudio, Fingerprint, HardDrive, RotateCcw, Target } from 'lucide-react'
import { GlowIcon } from '@/components/glow-icon'
import { useAnalyzeFile } from '@/hooks/use-analyze-file'
import { ANALYZE_FILE_URL, formatBytes, toPercent } from '@/lib/forensics'
import { ChunkTimeline } from './chunk-timeline'
import { Dropzone } from './dropzone'
import { MetadataInspector } from './metadata-inspector'
import { RadialGauge } from './radial-gauge'
import { VerdictHero } from './verdict-hero'

function InfoCard({ icon, label, value }: { icon: typeof FileAudio; label: string; value: string }) {
  return (
    <motion.div
      whileHover={{ y: -6, scale: 1.03 }}
      transition={{ type: 'spring', stiffness: 380, damping: 20 }}
      className="glass flex min-w-0 items-center gap-3 rounded-xl p-3 transition-colors duration-300 hover:border-neon-cyan/70"
    >
      <GlowIcon icon={icon} label={label} tone="cyan" size="sm" />
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
        <p className="truncate font-mono text-sm text-white" title={value}>{value}</p>
      </div>
    </motion.div>
  )
}

export function FileForensicsStudio() {
  const { result, file, audioUrl, loading, isSample, analyze, loadSample, reset } = useAnalyzeFile()

  const info = {
    original: result?.file_info?.original_filename ?? result?.original_filename ?? file?.name ?? '—',
    assigned: result?.file_info?.filename_assigned ?? result?.filename_assigned ?? '—',
    size: result?.file_info?.size_bytes ?? result?.size_bytes ?? file?.size,
  }

  return (
    <section aria-labelledby="forensics-title" className="glass relative w-full rounded-3xl p-5 sm:p-8">
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-neon-violet">Feature 02 · /api/v2/analyze-file</p>
            <h1 id="forensics-title" className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl">
              File Forensics Studio
            </h1>
          </div>
          <p className="max-w-md text-pretty text-sm leading-relaxed text-muted-foreground">
            Upload a recording to score every 2-second window for synthetic voice artifacts and inspect its metadata
            for signs of editing.
          </p>
        </div>

        {!result && <Dropzone onFile={analyze} loading={loading} fileName={file?.name} fileSize={file?.size} />}

        {!result && !loading && (
          <div className="flex flex-wrap items-center justify-center gap-3">
            <button
              type="button"
              onClick={loadSample}
              className="rounded-lg border border-neon-violet/40 bg-neon-violet/10 px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-neon-violet transition-all duration-300 hover:-translate-y-1 hover:scale-105 hover:border-neon-violet hover:shadow-[0_0_20px_rgba(176,38,255,0.7)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-violet/70"
            >
              Load sample report
            </button>
          </div>
        )}

        {result && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col gap-6">
            <VerdictHero verdict={result.overall_verdict} />

            <div className="grid gap-3 sm:grid-cols-3">
              <InfoCard icon={FileAudio} label="Original filename" value={info.original} />
              <InfoCard icon={Fingerprint} label="Assigned ID" value={info.assigned} />
              <InfoCard icon={HardDrive} label="Size" value={formatBytes(info.size)} />
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <RadialGauge
                label="Avg confidence"
                percent={toPercent(result.average_confidence_score)}
                icon={Target}
                caption="Mean fake probability across voiced chunks"
              />
              <RadialGauge
                label="Max chunk score"
                percent={toPercent(result.max_chunk_score)}
                icon={Activity}
                caption="Highest single-window probability"
              />
              <RadialGauge
                label="Fake chunk ratio"
                percent={toPercent(result.fake_chunk_ratio)}
                icon={BarChart3}
                caption="Share of windows flagged Synthetic/Clone"
              />
            </div>

            {result.timeline_analysis && result.timeline_analysis.length > 0 && (
              <ChunkTimeline chunks={result.timeline_analysis} audioUrl={audioUrl} />
            )}

            <MetadataInspector data={result.metadata_inspection} />

            <div className="flex flex-wrap items-center justify-between gap-3">
              <p className="truncate font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground/70">
                {isSample ? 'Sample report · no backend' : `Endpoint · ${ANALYZE_FILE_URL}`}
              </p>
              <button
                type="button"
                onClick={reset}
                className="flex items-center gap-2 rounded-xl border border-neon-cyan/50 bg-neon-cyan/10 px-4 py-2 font-mono text-xs uppercase tracking-[0.2em] text-neon-cyan transition-all duration-300 ease-out hover:-translate-y-1.5 hover:scale-105 hover:shadow-[0_0_24px_rgba(0,240,255,0.8)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-cyan/70"
              >
                <RotateCcw className="size-3.5" aria-hidden="true" />
                Analyze another file
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </section>
  )
}
