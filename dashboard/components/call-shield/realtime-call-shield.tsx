'use client'

import { motion } from 'framer-motion'
import { Activity, AlertTriangle, Gauge, Radio, ShieldCheck, Timer, Waves, Zap } from 'lucide-react'
import { GlowIcon, type NeonTone } from '@/components/glow-icon'
import { useRealtimeCall } from '@/hooks/use-realtime-call'
import { cn } from '@/lib/utils'
import { MicButton } from './mic-button'
import { NeonMeter } from './neon-meter'
import { Oscilloscope } from './oscilloscope'
import { ThreatBanner } from './threat-banner'
import { ThreatGraph } from './threat-graph'

const PHASE_LABEL = {
  idle: 'Standby',
  connecting: 'Opening',
  configuring: 'Configuring',
  streaming: 'Live',
  error: 'Offline',
} as const

function StatCard({ icon, label, value, tone }: { icon: typeof Radio; label: string; value: string; tone: NeonTone }) {
  return (
    <motion.div
      whileHover={{ y: -6, scale: 1.04 }}
      transition={{ type: 'spring', stiffness: 380, damping: 20 }}
      className="glass flex items-center gap-3 rounded-xl p-3 transition-colors duration-300 hover:border-neon-cyan/70"
    >
      <GlowIcon icon={icon} label={label} tone={tone} size="sm" />
      <div className="min-w-0">
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
        <p className="truncate font-mono text-sm font-medium text-white tabular-nums">{value}</p>
      </div>
    </motion.div>
  )
}

export function RealtimeCallShield() {
  const call = useRealtimeCall()
  const { hud, isThreat, isActive, phase } = call
  const analyzing = hud.status === 'analyzed'
  const silence = hud.status === 'silence'
  const waveColor = isThreat ? '#FF0055' : analyzing ? '#B026FF' : '#00F0FF'

  const statusTone: NeonTone = isThreat ? 'crimson' : hud.smoothedScore >= 0.4 && analyzing ? 'amber' : isActive ? 'emerald' : 'cyan'
  const statusText = isActive ? (silence ? hud.alert || 'Secure' : analyzing ? hud.alert || 'Secure' : PHASE_LABEL[phase]) : 'Standby'

  return (
    <>
      <div
        aria-hidden="true"
        className={cn(
          'pointer-events-none fixed inset-0 z-40 border-4 border-transparent transition-opacity duration-300',
          isThreat ? 'threat-pulse opacity-100' : 'opacity-0',
        )}
      />

      <section
        aria-labelledby="call-shield-title"
        className={cn(
          'glass relative w-full overflow-hidden rounded-3xl p-5 transition-colors duration-500 sm:p-8',
          isThreat && 'threat-pulse border-2',
        )}
      >
        <div className="flex flex-col gap-6">
          <ThreatBanner show={isThreat} alert={hud.alert} score={hud.smoothedScore} />

          <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.3em] text-neon-violet">Feature 03 · /ws/realtime-call</p>
              <h1 id="call-shield-title" className="text-balance text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Real-Time Call Shield
              </h1>
            </div>
            <p className="max-w-md text-pretty text-sm leading-relaxed text-muted-foreground">
              Streams 16-bit PCM from your microphone to the WavLM detector and scores every ~0.5s hop for synthetic
              voice signatures.
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div className="glass flex flex-col items-center justify-center gap-4 rounded-2xl px-4 py-8">
              <MicButton
                phase={phase}
                isActive={isActive}
                isThreat={isThreat}
                analyzing={analyzing}
                onToggle={isActive ? call.stop : call.start}
              />

              <div className="flex flex-col items-center gap-2 text-center" aria-live="polite">
                <span
                  className={cn(
                    'inline-flex items-center gap-2 rounded-full border px-4 py-1.5 font-mono text-xs uppercase tracking-[0.2em]',
                    statusTone === 'crimson' && 'border-neon-crimson/70 bg-neon-crimson/15 text-neon-crimson',
                    statusTone === 'amber' && 'border-neon-amber/60 bg-neon-amber/10 text-neon-amber',
                    statusTone === 'emerald' && 'border-neon-emerald/60 bg-neon-emerald/10 text-neon-emerald',
                    statusTone === 'cyan' && 'border-neon-cyan/50 bg-neon-cyan/10 text-neon-cyan',
                  )}
                >
                  <span className="relative flex size-2">
                    {isActive && <span className="absolute inline-flex size-full animate-ping rounded-full bg-current opacity-75" />}
                    <span className="relative inline-flex size-2 rounded-full bg-current" />
                  </span>
                  <span className="max-w-[16rem] truncate">{statusText}</span>
                </span>
                <p className="text-sm text-muted-foreground">
                  {isActive
                    ? silence
                      ? 'Line is quiet — monitoring for voice activity.'
                      : analyzing
                        ? `Analyzing a rolling ${hud.windowSeconds ?? '–'}s window.`
                        : 'Establishing a secure channel…'
                    : 'Press the shield to begin live call protection.'}
                </p>
                {call.error && (
                  <p role="alert" className="flex max-w-sm items-start gap-2 text-left text-xs text-neon-amber">
                    <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
                    {call.error}
                  </p>
                )}
                {!isActive && (
                  <button
                    type="button"
                    onClick={call.startDemo}
                    className="mt-1 rounded-lg border border-neon-violet/40 bg-neon-violet/10 px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.2em] text-neon-violet transition-all duration-300 hover:-translate-y-1 hover:scale-105 hover:border-neon-violet hover:shadow-[0_0_20px_rgba(176,38,255,0.7)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neon-violet/70"
                  >
                    Run simulated feed
                  </button>
                )}
                {call.demo && isActive && (
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-neon-violet">Simulated frames · no backend</p>
                )}
              </div>
            </div>

            <div className="flex min-w-0 flex-col gap-4">
              <div
                className={cn(
                  'glass relative h-36 overflow-hidden rounded-2xl transition-all duration-500',
                  silence && isActive && !isThreat && 'shadow-[0_0_40px_-6px_rgba(0,240,255,0.6)]',
                )}
              >
                <div className="absolute left-3 top-3 z-10 flex items-center gap-2">
                  <GlowIcon icon={Waves} label="Oscilloscope" tone={isThreat ? 'crimson' : analyzing ? 'violet' : 'cyan'} size="sm" />
                  <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Voice input</span>
                </div>
                {silence && isActive && !isThreat && (
                  <motion.span
                    aria-hidden="true"
                    className="absolute inset-0 rounded-2xl bg-neon-cyan/10"
                    animate={{ opacity: [0.1, 0.5, 0.1] }}
                    transition={{ duration: 2.6, repeat: Infinity, ease: 'easeInOut' }}
                  />
                )}
                <Oscilloscope
                  analyserRef={call.analyserRef}
                  color={waveColor}
                  active={isActive}
                  synthetic={call.demo && analyzing}
                />
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <NeonMeter label="Live threat score" value={hud.liveScore} icon={Zap} active={isActive && analyzing} />
                <NeonMeter label="Smoothed score" value={hud.smoothedScore} icon={Gauge} active={isActive && analyzing} />
              </div>

              <div className="glass relative h-44 overflow-hidden rounded-2xl">
                <div className="absolute left-3 top-3 z-10 flex flex-wrap items-center gap-3">
                  <GlowIcon icon={Activity} label="Threat timeline" tone={isThreat ? 'crimson' : 'cyan'} size="sm" />
                  <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    <span className="h-0.5 w-4 rounded bg-neon-cyan" /> Smoothed
                  </span>
                  <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    <span className="h-0.5 w-4 rounded bg-neon-violet" /> Live
                  </span>
                </div>
                <div className="absolute inset-x-0 bottom-0 top-12">
                  <ThreatGraph history={call.history} isThreat={isThreat} />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <StatCard icon={Radio} label="Socket" value={PHASE_LABEL[phase]} tone={isActive ? 'emerald' : 'cyan'} />
                <StatCard
                  icon={Waves}
                  label="Sample rate"
                  value={call.sampleRate ? `${call.sampleRate.toLocaleString()} Hz` : '16,000 Hz'}
                  tone="violet"
                />
                <StatCard
                  icon={Timer}
                  label="Window"
                  value={hud.windowSeconds != null ? `${hud.windowSeconds}s` : '—'}
                  tone="amber"
                />
                <StatCard
                  icon={ShieldCheck}
                  label="PCM frames"
                  value={call.demo ? 'simulated' : call.framesSent.toLocaleString()}
                  tone={isThreat ? 'crimson' : 'cyan'}
                />
              </div>
            </div>
          </div>

          <p className="truncate font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground/70">
            Endpoint · {call.wsUrl}
          </p>
        </div>
      </section>
    </>
  )
}
